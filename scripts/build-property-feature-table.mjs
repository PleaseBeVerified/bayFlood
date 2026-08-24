import { readFile, writeFile } from "node:fs/promises";
import { sampleElevation } from "../src/lib/elevation.ts";
import { lookupFema } from "../src/lib/fema.ts";
import { sampleRainfall } from "../src/lib/rainfall.ts";
import { lookupNearbyWater } from "../src/lib/water.ts";
import { serializeCsv, validateObservationTable } from "./lib/property-observations.mjs";

const inputPath = process.argv.find((argument) => argument.startsWith("--input="))?.slice(8)
  ?? "data/property-observations.csv";
const outputPath = process.argv.find((argument) => argument.startsWith("--output="))?.slice(9)
  ?? "data/property-observation-features.csv";

export const featureHeaders = [
  "property_id", "parcel_id", "building_id", "address", "lat", "lng", "county",
  "observation_start", "observation_end", "event_date", "event_source",
  "non_event_evidence", "verification_status", "coordinate_precision_m", "source_url",
  "elevation_m", "elevation_ft", "fema_zone", "fema_subtype", "sfha",
  "distance_sfha_m", "rainfall_100yr_24h_in", "nearest_water_status",
  "nearest_water_distance_m", "nearest_water_kind", "nearest_water_name",
  "notes",
];

async function buildFeature(row) {
  const [elevationMeters, fema, rainfallInches, nearbyWater] = await Promise.all([
    sampleElevation(row.lng, row.lat), lookupFema(row.lng, row.lat),
    sampleRainfall(row.lng, row.lat), lookupNearbyWater(row.lng, row.lat),
  ]);
  if (elevationMeters === null) throw new Error(`No elevation tile covers ${row.property_id}.`);
  return {
    ...row,
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
    rainfall_100yr_24h_in: rainfallInches === null ? "" : Math.round(rainfallInches * 100) / 100,
    nearest_water_status: nearbyWater.status,
    nearest_water_distance_m: nearbyWater.distanceMeters ?? "",
    nearest_water_kind: nearbyWater.kind ?? "",
    nearest_water_name: nearbyWater.name ?? "",
  };
}

async function main() {
  const validation = validateObservationTable(await readFile(inputPath, "utf8"));
  if (validation.errors.length) throw new Error(validation.errors.join("\n"));
  const rows = [];
  for (const row of validation.verifiedRows) {
    rows.push(await buildFeature(row));
    console.log(`Built ${rows.length}/${validation.verifiedRows.length}: ${row.property_id}`);
  }
  await writeFile(outputPath, serializeCsv(rows, featureHeaders));
  console.log(`Wrote ${rows.length} verified properties to ${outputPath}.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
