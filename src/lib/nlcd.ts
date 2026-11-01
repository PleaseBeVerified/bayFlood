import path from "node:path";
import { fromFile, type GeoTIFFImage } from "geotiff";
import proj4 from "proj4";

type NlcdRaster = {
  image: GeoTIFFImage;
  origin: [number, number];
  resolution: [number, number];
  noData: number | null;
};

const NLCD_ALBERS =
  "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs";
const toNlcdAlbers = proj4("WGS84", NLCD_ALBERS);
let rasterPromise: Promise<NlcdRaster> | undefined;

async function loadRaster(): Promise<NlcdRaster> {
  const tiff = await fromFile(
    path.join(process.cwd(), "data", "nlcd", "fractional-impervious-2025-full.tiff"),
  );
  const image = await tiff.getImage();
  const [originX, originY] = image.getOrigin();
  const [resolutionX, resolutionY] = image.getResolution();
  return {
    image,
    origin: [originX, originY],
    resolution: [resolutionX, resolutionY],
    noData: image.getGDALNoData(),
  };
}

function getRaster() {
  rasterPromise ??= loadRaster();
  return rasterPromise;
}

export async function sampleImperviousness(
  longitude: number,
  latitude: number,
) {
  const raster = await getRaster();
  const [x, y] = toNlcdAlbers.forward([longitude, latitude]);
  const column = Math.floor((x - raster.origin[0]) / raster.resolution[0]);
  const row = Math.floor((y - raster.origin[1]) / raster.resolution[1]);

  if (
    column < 0 ||
    column >= raster.image.getWidth() ||
    row < 0 ||
    row >= raster.image.getHeight()
  ) {
    return null;
  }

  const pixels = await raster.image.readRasters({
    window: [column, row, column + 1, row + 1],
    interleave: true,
  });
  const value = Number((pixels as ArrayLike<number>)[0]);
  return Number.isFinite(value) && value !== raster.noData ? value : null;
}
