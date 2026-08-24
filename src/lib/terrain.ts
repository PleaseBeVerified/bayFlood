import { access } from "node:fs/promises";
import path from "node:path";
import {
  fromFile,
  fromUrl,
  type GeoTIFFImage,
} from "geotiff";
import proj4 from "proj4";

const WGS84 = "EPSG:4326";
const CALIFORNIA_ALBERS =
  "+proj=aea +lat_0=0 +lon_0=-120 +lat_1=34 +lat_2=40.5 " +
  "+x_0=0 +y_0=-4000000 +datum=NAD83 +units=m +no_defs +type=crs";
const RUNTIME_COG_PATH = path.join(
  process.cwd(),
  "terrain",
  "runtime",
  "terrain-features-30m.tif",
);

const terrainFiles = {
  slope_degrees: "slope-degrees.tif",
  local_elevation_250m: "local-elevation-250m.tif",
  local_elevation_500m: "local-elevation-500m.tif",
  flow_accumulation_m2: "flow-accumulation-m2.tif",
  height_above_drainage_m: "height-above-drainage-m.tif",
} as const;
const terrainFeatureNames = Object.keys(terrainFiles) as Array<
  keyof typeof terrainFiles
>;

export type TerrainFeatures = Record<keyof typeof terrainFiles, number | null>;

type TerrainRaster = {
  image: GeoTIFFImage;
  bounds: [number, number, number, number];
  noData: number | null;
};

type TerrainSource =
  | { mode: "stack"; raster: TerrainRaster }
  | {
      mode: "separate";
      rasters: Record<keyof typeof terrainFiles, TerrainRaster>;
    };

let sourcePromise: Promise<TerrainSource> | undefined;

function rasterFromImage(image: GeoTIFFImage): TerrainRaster {
  return {
    image,
    bounds: image.getBoundingBox() as [number, number, number, number],
    noData: image.getGDALNoData(),
  };
}

async function loadStackFromUrl(url: string): Promise<TerrainSource> {
  const parsed = new URL(url);
  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    throw new Error("TERRAIN_COG_URL must use http or https.");
  }
  const tiff = await fromUrl(parsed.toString());
  const image = await tiff.getImage();
  if (image.getSamplesPerPixel() !== terrainFeatureNames.length) {
    throw new Error(
      `Runtime terrain COG must contain ${terrainFeatureNames.length} bands.`,
    );
  }
  return { mode: "stack", raster: rasterFromImage(image) };
}

async function fileExists(file: string) {
  try {
    await access(file);
    return true;
  } catch {
    return false;
  }
}

async function loadLocalStack(): Promise<TerrainSource | null> {
  if (!(await fileExists(RUNTIME_COG_PATH))) return null;
  const tiff = await fromFile(RUNTIME_COG_PATH);
  const image = await tiff.getImage();
  if (image.getSamplesPerPixel() !== terrainFeatureNames.length) {
    throw new Error(
      `Runtime terrain COG must contain ${terrainFeatureNames.length} bands.`,
    );
  }
  return { mode: "stack", raster: rasterFromImage(image) };
}

async function loadSeparateLocalRasters(): Promise<TerrainSource> {
  const entries = await Promise.all(
    Object.entries(terrainFiles).map(async ([feature, file]) => {
      const tiff = await fromFile(path.join(process.cwd(), "terrain", file));
      return [feature, rasterFromImage(await tiff.getImage())];
    }),
  );
  return {
    mode: "separate",
    rasters: Object.fromEntries(entries) as Record<
      keyof typeof terrainFiles,
      TerrainRaster
    >,
  };
}

async function loadTerrainSource(): Promise<TerrainSource> {
  const remoteUrl = process.env.TERRAIN_COG_URL?.trim();
  if (remoteUrl) return loadStackFromUrl(remoteUrl);
  return (await loadLocalStack()) ?? loadSeparateLocalRasters();
}

function getTerrainSource() {
  sourcePromise ??= loadTerrainSource();
  return sourcePromise;
}

function rasterWindow(
  raster: TerrainRaster,
  easting: number,
  northing: number,
): [number, number, number, number] | null {
  const [west, south, east, north] = raster.bounds;
  if (
    easting < west ||
    easting >= east ||
    northing <= south ||
    northing > north
  ) {
    return null;
  }

  const x = Math.floor(
    ((easting - west) / (east - west)) * raster.image.getWidth(),
  );
  const y = Math.floor(
    ((north - northing) / (north - south)) * raster.image.getHeight(),
  );
  return [x, y, x + 1, y + 1];
}

function validRasterValue(value: number, noData: number | null) {
  return Number.isFinite(value) && value !== noData ? value : null;
}

async function sampleStack(
  raster: TerrainRaster,
  easting: number,
  northing: number,
): Promise<TerrainFeatures> {
  const window = rasterWindow(raster, easting, northing);
  if (!window) {
    return Object.fromEntries(
      terrainFeatureNames.map((feature) => [feature, null]),
    ) as TerrainFeatures;
  }
  const values = await raster.image.readRasters({
    window,
    samples: terrainFeatureNames.map((_, index) => index),
    interleave: true,
  });
  return Object.fromEntries(
    terrainFeatureNames.map((feature, index) => [
      feature,
      validRasterValue(Number((values as ArrayLike<number>)[index]), raster.noData),
    ]),
  ) as TerrainFeatures;
}

async function sampleSingleRaster(
  raster: TerrainRaster,
  easting: number,
  northing: number,
) {
  const window = rasterWindow(raster, easting, northing);
  if (!window) return null;
  const values = await raster.image.readRasters({ window, interleave: true });
  return validRasterValue(Number((values as ArrayLike<number>)[0]), raster.noData);
}

export async function sampleTerrainFeatures(
  longitude: number,
  latitude: number,
): Promise<TerrainFeatures> {
  const [easting, northing] = proj4(WGS84, CALIFORNIA_ALBERS, [
    longitude,
    latitude,
  ]);
  const source = await getTerrainSource();
  if (source.mode === "stack") {
    return sampleStack(source.raster, easting, northing);
  }
  const entries = await Promise.all(
    Object.entries(source.rasters).map(async ([feature, raster]) => [
      feature,
      await sampleSingleRaster(raster, easting, northing),
    ]),
  );
  return Object.fromEntries(entries) as TerrainFeatures;
}
