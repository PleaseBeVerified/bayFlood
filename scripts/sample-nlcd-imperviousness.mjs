import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import proj4 from "proj4";
import { fromFile } from "geotiff";
import { parseCsv, serializeCsv } from "./lib/csv.mjs";
import { buildImperviousnessTemplate, imperviousnessHeaders } from "./lib/dnn-scenarios.mjs";

const option = (name, fallback) =>
  process.argv.find((argument) => argument.startsWith(`--${name}=`))?.split("=").slice(1).join("=") ??
  fallback;

if (process.argv.includes("--help")) {
  console.log(`Usage:
  node scripts/sample-nlcd-imperviousness.mjs [options]

Options:
  --raster=PATH     Annual NLCD Fractional Impervious Surface GeoTIFF
  --calibration=CSV  Physical calibration locations (default)
  --input=CSV        Optional location-keyed imperviousness template CSV
  --output=PATH      Completed DNN imperviousness CSV`);
  process.exit(0);
}

const rasterPath = option("raster", "data/nlcd/fractional-impervious-2025-full.tiff");
const inputPath = option("input", null);
const calibrationPath = option("calibration", "data/physical-hazard-calibration.csv");
const outputPath = option("output", "data/dnn-imperviousness.csv");

const nlcdAlbers = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs";
const toNlcdAlbers = proj4("WGS84", nlcdAlbers);

const template = inputPath
  ? parseCsv(await readFile(inputPath, "utf8")).rows
  : buildImperviousnessTemplate(parseCsv(await readFile(calibrationPath, "utf8")).rows);
const tiff = await fromFile(rasterPath);
const image = await tiff.getImage();
const [originX, originY] = image.getOrigin();
const [resolutionX, resolutionY] = image.getResolution();
const width = image.getWidth();
const height = image.getHeight();
const noDataValue = image.getGDALNoData();
const pixels = await image.readRasters({ interleave: true });

const missing = [];
const rows = template.map((row) => {
  const latitude = Number(row.lat);
  const longitude = Number(row.lng);
  const [x, y] = toNlcdAlbers.forward([longitude, latitude]);
  const column = Math.floor((x - originX) / resolutionX);
  const rasterRow = Math.floor((y - originY) / resolutionY);
  const value = pixels[rasterRow * width + column];

  if (
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude) ||
    column < 0 ||
    column >= width ||
    rasterRow < 0 ||
    rasterRow >= height ||
    !Number.isFinite(value) ||
    value === noDataValue
  ) {
    missing.push(row.location_id);
    return null;
  }

  return {
    location_id: row.location_id,
    lat: latitude,
    lng: longitude,
    county: row.county,
    current_imperviousness_pct: value,
    source: "USGS Annual NLCD Fractional Impervious Surface (2025)",
  };
});

if (missing.length > 0) {
  throw new Error(
    `${missing.length} of ${template.length} template locations are outside the raster or have no data. ` +
      "Download a raster that covers the full model area before building the DNN table.",
  );
}

await mkdir(path.dirname(outputPath), { recursive: true });
await writeFile(outputPath, serializeCsv(rows, imperviousnessHeaders));
console.log(`Wrote ${rows.length} sampled imperviousness rows to ${outputPath}.`);
