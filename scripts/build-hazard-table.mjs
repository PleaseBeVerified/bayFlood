import { readFile, writeFile } from "node:fs/promises";
import { expandHazardPeriods, parseCsv, serializeCsv } from "./lib/property-observations.mjs";

const inputPath = process.argv.find((argument) => argument.startsWith("--input="))?.slice(8)
  ?? "data/property-observation-features.csv";
const outputPath = process.argv.find((argument) => argument.startsWith("--output="))?.slice(9)
  ?? "data/property-hazard-table.csv";

const periodHeaders = [
  "property_id", "calendar_year", "time_index", "period_start", "period_end",
  "exposure_days", "exposure_fraction", "event",
];

async function main() {
  const { headers, rows } = parseCsv(await readFile(inputPath, "utf8"));
  const required = ["property_id", "observation_start", "observation_end", "event_date"];
  const missing = required.filter((header) => !headers.includes(header));
  if (missing.length) throw new Error(`Missing feature columns: ${missing.join(", ")}.`);
  const staticHeaders = headers.filter((header) => !periodHeaders.includes(header));
  const hazardRows = rows.flatMap((row) =>
    expandHazardPeriods(row).map((period) => ({ ...row, ...period })),
  );
  await writeFile(outputPath, serializeCsv(hazardRows, [...periodHeaders, ...staticHeaders]));
  console.log(`Wrote ${hazardRows.length} property-years to ${outputPath}.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
