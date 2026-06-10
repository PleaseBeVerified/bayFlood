import { readdir } from "node:fs/promises";
import path from "node:path";
import { fromFile, type GeoTIFFImage } from "geotiff";

type ElevationTile = {
  image: GeoTIFFImage;
  bounds: [number, number, number, number];
  noData: number | null;
};

let tilesPromise: Promise<ElevationTile[]> | undefined;

async function loadTiles() {
  const elevationDirectory = path.join(process.cwd(), "elevation");
  const files = (await readdir(elevationDirectory)).filter((file) =>
    /\.tiff?$/i.test(file),
  );

  return Promise.all(
    files.map(async (file) => {
      const tiff = await fromFile(path.join(elevationDirectory, file));
      const image = await tiff.getImage();

      return {
        image,
        bounds: image.getBoundingBox() as [number, number, number, number],
        noData: image.getGDALNoData(),
      };
    }),
  );
}

function getTiles() {
  tilesPromise ??= loadTiles();
  return tilesPromise;
}

export async function sampleElevation(longitude: number, latitude: number) {
  const tiles = await getTiles();
  const tile = tiles.find(({ bounds }) => {
    const [west, south, east, north] = bounds;
    return (
      longitude >= west &&
      longitude <= east &&
      latitude >= south &&
      latitude <= north
    );
  });

  if (!tile) return null;

  const [west, south, east, north] = tile.bounds;
  const width = tile.image.getWidth();
  const height = tile.image.getHeight();
  const x = Math.max(
    0,
    Math.min(width - 1, Math.floor(((longitude - west) / (east - west)) * width)),
  );
  const y = Math.max(
    0,
    Math.min(height - 1, Math.floor(((north - latitude) / (north - south)) * height)),
  );
  const radius = 2;
  const window: [number, number, number, number] = [
    Math.max(0, x - radius),
    Math.max(0, y - radius),
    Math.min(width, x + radius + 1),
    Math.min(height, y + radius + 1),
  ];
  const rasters = await tile.image.readRasters({ window, interleave: true });
  const values = Array.from(rasters as ArrayLike<number>)
    .filter((value) => Number.isFinite(value) && value !== tile.noData)
    .sort((a, b) => a - b);

  if (values.length === 0) return null;

  return values[Math.floor(values.length / 2)];
}
