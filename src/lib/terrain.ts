import path from "node:path";
import { fromFile, type GeoTIFFImage } from "geotiff";
import proj4 from "proj4";

const CALIFORNIA_ALBERS = "+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +datum=NAD83 +units=m +no_defs";
const TERRAIN_FILE = path.join(process.cwd(), "terrain/runtime/terrain-features-30m.tif");

export type TerrainFeatures = {
  slopeDegrees: number;
  localElevation250Meters: number;
  localElevation500Meters: number;
  flowAccumulationSquareMeters: number;
  heightAboveDrainageMeters: number;
};

let imagePromise: Promise<GeoTIFFImage> | undefined;

function getImage() {
  imagePromise ??= fromFile(TERRAIN_FILE).then((tiff) => tiff.getImage());
  return imagePromise;
}

export async function sampleTerrain(longitude: number, latitude: number): Promise<TerrainFeatures | null> {
  const image = await getImage();
  const [x, y] = proj4("EPSG:4326", CALIFORNIA_ALBERS, [longitude, latitude]);
  const [originX, originY] = image.getOrigin();
  const [resolutionX, resolutionY] = image.getResolution();
  const column = Math.floor((x - originX) / resolutionX);
  const row = Math.floor((y - originY) / resolutionY);
  if (column < 0 || row < 0 || column >= image.getWidth() || row >= image.getHeight()) {
    return null;
  }

  const raster = await image.readRasters({
    window: [column, row, column + 1, row + 1],
    interleave: true,
  });
  const values = Array.from(raster as ArrayLike<number>);
  const noData = image.getGDALNoData();
  if (values.length !== 5 || values.some((value) => !Number.isFinite(value) || value === noData)) {
    return null;
  }
  // Calibration CSV values were sampled from this raster and rounded to 0.001.
  const [slopeDegrees, localElevation250Meters, localElevation500Meters,
    flowAccumulationSquareMeters, heightAboveDrainageMeters] = values.map(
    (value) => Math.round(value * 1_000) / 1_000,
  );
  return {
    slopeDegrees,
    localElevation250Meters,
    localElevation500Meters,
    flowAccumulationSquareMeters,
    heightAboveDrainageMeters,
  };
}
