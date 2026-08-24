import { readFile } from "node:fs/promises";
import { validateObservationTable } from "./lib/property-observations.mjs";

const inputPath = process.argv.find((argument) => argument.startsWith("--input="))?.slice(8)
  ?? "data/property-observations.csv";

async function main() {
  const result = validateObservationTable(await readFile(inputPath, "utf8"));
  console.log(`Rows: ${result.rows.length}`);
  console.log(`Verified: ${result.verifiedRows.length}`);
  for (const warning of result.warnings) console.warn(`Warning: ${warning}`);
  for (const error of result.errors) console.error(`Error: ${error}`);
  if (result.errors.length > 0) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
