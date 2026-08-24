import { readFile, writeFile } from "node:fs/promises";
import { sampleElevation } from "../src/lib/elevation.ts";
import { lookupFema } from "../src/lib/fema.ts";
import { sampleRainfall } from "../src/lib/rainfall.ts";
import { lookupNearbyWater } from "../src/lib/water.ts";

const inputPath = "data/training-labels.csv";
const outputPath = "data/training-table.csv";

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const nextCharacter = line[index + 1];

    if (character === '"' && quoted && nextCharacter === '"') {
      value += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += character;
    }
  }

  values.push(value);
  return values;
}

function parseCsv(contents) {
  const lines = contents
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);

  if (lines.length === 0) return [];

  const headers = parseCsvLine(lines[0]);

  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""]),
    );
  }).map((row, index) => ({ ...row, rowNumber: index + 2 }));
}

function serializeCsvValue(value) {
  if (value === null || value === undefined) return "";

  const text = String(value);
  if (/[",\n\r]/.test(text)) {
    return `"${text.replaceAll('"', '""')}"`;
  }

  return text;
}

function serializeCsv(rows, headers) {
  return [
    headers.join(","),
    ...rows.map((row) =>
      headers.map((header) => serializeCsvValue(row[header])).join(","),
    ),
  ].join("\n") + "\n";
}

function parseNumber(value, fieldName, rowNumber) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`Row ${rowNumber}: ${fieldName} must be a number.`);
  }
  return number;
}

function floodYear(labelRow, flooded) {
  const explicitYear = String(labelRow.flood_year ?? "").trim();
  if (flooded === 0) {
    if (explicitYear) {
      throw new Error(
        `Row ${labelRow.rowNumber}: flood_year must be empty when flooded is 0.`,
      );
    }
    return "";
  }

  const inferredYear = String(labelRow.notes ?? "").match(
    /\b(?:19|20)\d{2}\b/,
  )?.[0];
  const value = explicitYear || inferredYear || "";

  if (!value) return "";

  const year = Number(value);
  const currentYear = new Date().getUTCFullYear();
  if (!Number.isInteger(year) || year < 1900 || year > currentYear) {
    throw new Error(`Row ${labelRow.rowNumber}: flood_year is invalid.`);
  }
  return year;
}

async function buildTrainingRow(labelRow) {
  const latitude = parseNumber(labelRow.lat, "lat", labelRow.rowNumber);
  const longitude = parseNumber(labelRow.lng, "lng", labelRow.rowNumber);
  const flooded = parseNumber(labelRow.flooded, "flooded", labelRow.rowNumber);

  if (flooded !== 0 && flooded !== 1) {
    throw new Error(`Row ${labelRow.rowNumber}: flooded must be 0 or 1.`);
  }

  const [elevationMeters, fema, rainfallInches, nearbyWater] = await Promise.all([
    sampleElevation(longitude, latitude),
    lookupFema(longitude, latitude),
    sampleRainfall(longitude, latitude),
    lookupNearbyWater(longitude, latitude),
  ]);

  if (elevationMeters === null) {
    throw new Error(
      `Row ${labelRow.rowNumber}: no elevation tile covers ${latitude}, ${longitude}.`,
    );
  }

  return {
    id: labelRow.id,
    lat: latitude,
    lng: longitude,
    flooded,
    source: labelRow.source,
    flood_year: floodYear(labelRow, flooded),
    address: labelRow.address ?? "",
    reference_score: labelRow.reference_score ?? "",
    reference_label: labelRow.reference_label ?? "",
    source_url: labelRow.source_url ?? "",
    elevation_m: Math.round(elevationMeters * 10) / 10,
    elevation_ft: Math.round(elevationMeters * 3.28084),
    fema_zone: fema.floodZone.zone ?? "",
    fema_subtype: fema.floodZone.subtype ?? "",
    sfha:
      fema.floodZone.sfha === null || fema.floodZone.sfha === undefined
        ? ""
        : fema.floodZone.sfha === "T"
          ? 1
          : 0,
    distance_sfha_m: fema.distanceToSfhaMeters ?? "",
    rainfall_100yr_24h_in:
      rainfallInches === null ? "" : Math.round(rainfallInches * 100) / 100,
    nearest_water_status: nearbyWater.status,
    nearest_water_distance_m: nearbyWater.distanceMeters ?? "",
    nearest_water_kind: nearbyWater.kind ?? "",
    nearest_water_name: nearbyWater.name ?? "",
    notes: labelRow.notes,
  };
}

async function main() {
  const labels = parseCsv(await readFile(inputPath, "utf8"));

  if (labels.length === 0) {
    throw new Error(`${inputPath} has no rows.`);
  }

  const rows = [];
  for (const label of labels) {
    rows.push(await buildTrainingRow(label));
    console.log(`Built ${rows.length}/${labels.length}: ${label.id}`);
  }

  const headers = [
    "id",
    "lat",
    "lng",
    "flooded",
    "source",
    "flood_year",
    "address",
    "reference_score",
    "reference_label",
    "source_url",
    "elevation_m",
    "elevation_ft",
    "fema_zone",
    "fema_subtype",
    "sfha",
    "distance_sfha_m",
    "rainfall_100yr_24h_in",
    "nearest_water_status",
    "nearest_water_distance_m",
    "nearest_water_kind",
    "nearest_water_name",
    "notes",
  ];

  await writeFile(outputPath, serializeCsv(rows, headers));
  console.log(`Wrote ${rows.length} rows to ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
