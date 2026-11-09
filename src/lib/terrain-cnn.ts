import { spawn } from "node:child_process";
import path from "node:path";
import { fromFile, type GeoTIFFImage } from "geotiff";
import proj4 from "proj4";
import type { BoostedFloodRiskInput } from "./boosted-model";

const CALIFORNIA_ALBERS = "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +datum=NAD83 +units=m +no_defs";
const SIZE = 64;
const PIXELS = SIZE * SIZE;
const CHANNELS = 9;
const NO_DATA = -9999;

type Images = { terrain: GeoTIFFImage; dem: GeoTIFFImage; imperviousness: GeoTIFFImage };
type CnnResult = {
  onePercentPatternScore: number;
  classification: "higher-risk pattern" | "lower-risk pattern";
  imputedFeatures: string[];
  topographyCoverage: number;
  imperviousnessCoverage: number;
};

let imagesPromise: Promise<Images> | undefined;

async function loadImages(): Promise<Images> {
  const root = process.cwd();
  const files = [
    "terrain/runtime/terrain-features-30m.tif",
    "data/cache/terrain-cnn/dem-30m.tif",
    "data/cache/terrain-cnn/imperviousness-30m.tif",
  ];
  const images = await Promise.all(files.map(async (file) =>
    (await fromFile(path.join(root, file))).getImage()));
  const [terrain, dem, imperviousness] = images;
  if (terrain.getSamplesPerPixel() !== 5 || dem.getSamplesPerPixel() !== 1 ||
      imperviousness.getSamplesPerPixel() !== 1) {
    throw new Error("CNN raster channel counts do not match the training grids");
  }
  const referenceOrigin = terrain.getOrigin();
  const referenceResolution = terrain.getResolution();
  for (const image of [dem, imperviousness]) {
    const origin = image.getOrigin();
    const resolution = image.getResolution();
    if (image.getWidth() !== terrain.getWidth() || image.getHeight() !== terrain.getHeight() ||
        origin.some((value, index) => Math.abs(value - referenceOrigin[index]) > 1e-6) ||
        resolution.some((value, index) => Math.abs(value - referenceResolution[index]) > 1e-9)) {
      throw new Error("CNN raster grids are not aligned");
    }
  }
  return { terrain, dem, imperviousness };
}

function getImages() {
  imagesPromise ??= loadImages().catch((error) => {
    imagesPromise = undefined;
    throw error;
  });
  return imagesPromise;
}

async function readWindow(image: GeoTIFFImage, column: number, row: number): Promise<Float32Array> {
  const bands = image.getSamplesPerPixel();
  const data = new Float32Array(bands * PIXELS);
  data.fill(NO_DATA);
  const left = column - SIZE / 2;
  const top = row - SIZE / 2;
  const x0 = Math.max(0, left);
  const y0 = Math.max(0, top);
  const x1 = Math.min(image.getWidth(), left + SIZE);
  const y1 = Math.min(image.getHeight(), top + SIZE);
  if (x1 <= x0 || y1 <= y0) return data;
  const read = await image.readRasters({ window: [x0, y0, x1, y1], interleave: true });
  const values = read as ArrayLike<number>;
  const width = x1 - x0;
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const source = ((y - y0) * width + x - x0) * bands;
      const destination = (y - top) * SIZE + x - left;
      for (let channel = 0; channel < bands; channel++) {
        data[channel * PIXELS + destination] = values[source + channel];
      }
    }
  }
  return data;
}

export async function sampleCnnPatch(longitude: number, latitude: number, elevationMeters: number) {
  const { terrain, dem, imperviousness } = await getImages();
  const [x, y] = proj4("EPSG:4326", CALIFORNIA_ALBERS, [longitude, latitude]);
  const [originX, originY] = terrain.getOrigin();
  const [resolutionX, resolutionY] = terrain.getResolution();
  const column = Math.floor((x - originX) / resolutionX);
  const row = Math.floor((y - originY) / resolutionY);
  if (column < 0 || row < 0 || column >= terrain.getWidth() || row >= terrain.getHeight()) {
    return null;
  }
  const [terrainPixels, demPixels, imperviousnessPixels] = await Promise.all([
    readWindow(terrain, column, row),
    readWindow(dem, column, row),
    readWindow(imperviousness, column, row),
  ]);
  const patch = new Float32Array(CHANNELS * PIXELS);
  for (let pixel = 0; pixel < PIXELS; pixel++) {
    const topographyValid = Number.isFinite(demPixels[pixel]) && demPixels[pixel] !== NO_DATA &&
      [0, 1, 2, 3, 4].every((channel) => {
        const value = terrainPixels[channel * PIXELS + pixel];
        return Number.isFinite(value) && value !== NO_DATA;
      });
    if (topographyValid) {
      for (let channel = 0; channel < 5; channel++) {
        patch[channel * PIXELS + pixel] = terrainPixels[channel * PIXELS + pixel];
      }
      patch[3 * PIXELS + pixel] = Math.log1p(Math.max(0, patch[3 * PIXELS + pixel]));
      patch[5 * PIXELS + pixel] = demPixels[pixel] - elevationMeters;
      patch[7 * PIXELS + pixel] = 1;
    }
    const imperviousValue = imperviousnessPixels[pixel];
    if (Number.isFinite(imperviousValue) && imperviousValue >= 0 && imperviousValue <= 100) {
      patch[6 * PIXELS + pixel] = imperviousValue;
      patch[8 * PIXELS + pixel] = 1;
    }
  }
  if (patch[7 * PIXELS + 32 * SIZE + 32] !== 1) return null;
  return patch;
}

function pointFeatures(input: BoostedFloodRiskInput): Array<number | null> {
  return [
    input.elevationMeters,
    input.imperviousnessPercent,
    input.rainfallMillimeters,
    input.nearestWaterAvailable,
    input.nearestWaterDistanceMeters === null ? null : Math.log1p(Math.max(0, input.nearestWaterDistanceMeters)),
    input.slopeDegrees,
    input.localElevation250Meters,
    input.localElevation500Meters,
    Math.log1p(Math.max(0, input.flowAccumulationSquareMeters)),
    input.heightAboveDrainageMeters,
  ];
}

function runPython(patch: Float32Array, features: Array<number | null>): Promise<CnnResult> {
  const root = process.cwd();
  const python = path.join(root, ".venv-model/bin/python");
  const script = path.join(root, "scripts/predict-terrain-cnn.py");
  const input = JSON.stringify({
    patchBase64: Buffer.from(patch.buffer, patch.byteOffset, patch.byteLength).toString("base64"),
    features,
  });
  return new Promise((resolve, reject) => {
    const child = spawn(python, [script], { cwd: root, stdio: ["pipe", "pipe", "pipe"] });
    let output = "";
    let errors = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, 30_000);
    child.stdout.setEncoding("utf8").on("data", (chunk: string) => { output += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk: string) => { errors += chunk; });
    child.stdin.on("error", () => {});
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut || code !== 0) {
        reject(new Error(timedOut ? "CNN inference timed out" : errors.trim() || "CNN inference failed"));
        return;
      }
      try {
        const result = JSON.parse(output) as CnnResult;
        if (!Number.isFinite(result.onePercentPatternScore) ||
            result.onePercentPatternScore < 0 || result.onePercentPatternScore > 1) {
          throw new Error("Invalid CNN score");
        }
        resolve(result);
      } catch (error) {
        reject(error);
      }
    });
    child.stdin.end(input);
  });
}

export async function predictTerrainCnn(longitude: number, latitude: number, input: BoostedFloodRiskInput) {
  const patch = await sampleCnnPatch(longitude, latitude, input.elevationMeters);
  if (patch === null) return null;
  return runPython(patch, pointFeatures(input));
}
