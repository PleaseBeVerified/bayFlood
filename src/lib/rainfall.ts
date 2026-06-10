import path from "node:path";
import { fromFile, type GeoTIFFImage } from "geotiff";

type RainfallRaster = {
  image: GeoTIFFImage;
  bounds: [number, number, number, number];
  noData: number | null;
};

let rasterPromise: Promise<RainfallRaster> | undefined;

async function loadRaster() {
  const tiff = await fromFile(
    path.join(process.cwd(), "rainfall", "rainfall_data.tif"),
  );
  const image = await tiff.getImage();

  return {
    image,
    bounds: image.getBoundingBox() as [number, number, number, number],
    noData: image.getGDALNoData(),
  };
}

function getRaster() {
  rasterPromise ??= loadRaster();
  return rasterPromise;
}

export async function sampleRainfall(longitude: number, latitude: number) {
  const { image, bounds, noData } = await getRaster();
  const [west, south, east, north] = bounds;

  if (
    longitude < west ||
    longitude > east ||
    latitude < south ||
    latitude > north
  ) {
    return null;
  }

  const width = image.getWidth();
  const height = image.getHeight();
  const x = Math.max(
    0,
    Math.min(width - 1, Math.floor(((longitude - west) / (east - west)) * width)),
  );
  const y = Math.max(
    0,
    Math.min(height - 1, Math.floor(((north - latitude) / (north - south)) * height)),
  );
  const rasters = await image.readRasters({
    window: [x, y, x + 1, y + 1],
    interleave: true,
  });
  const value = Number((rasters as ArrayLike<number>)[0]);

  if (!Number.isFinite(value) || value === noData) return null;

  // NOAA Atlas 14 stores these grid values in thousandths of an inch.
  return value / 1_000;
}
