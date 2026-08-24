import { readFile } from "node:fs/promises";
import { sampleElevation } from "../src/lib/elevation.ts";
import { lookupFema } from "../src/lib/fema.ts";
import {
  modelRiskPresentation,
  predictFloodClaimSimilarity,
} from "../src/lib/flood-model.ts";
import { sampleRainfall } from "../src/lib/rainfall.ts";
import { lookupNearbyWater } from "../src/lib/water.ts";

const inputPath = "data/firststreet-test-cases.csv";
const elevatedRiskMinimum = 4;

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
  const lines = contents.split(/\r?\n/).filter((line) => line.trim());
  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(
      headers.map((header, index) => [header, values[index] ?? ""]),
    );
  });
}

async function evaluateCase(testCase) {
  const latitude = Number(testCase.lat);
  const longitude = Number(testCase.lng);
  const floodFactor = Number(testCase.flood_factor);
  const [elevationMeters, fema, rainfallInches, nearbyWater] =
    await Promise.all([
      sampleElevation(longitude, latitude),
      lookupFema(longitude, latitude),
      sampleRainfall(longitude, latitude),
      lookupNearbyWater(longitude, latitude),
    ]);

  if (elevationMeters === null) {
    throw new Error(`No elevation tile covers ${testCase.address}.`);
  }

  const prediction = predictFloodClaimSimilarity({
    elevation_m: Math.round(elevationMeters * 10) / 10,
    sfha:
      fema.floodZone.sfha === null || fema.floodZone.sfha === undefined
        ? null
        : fema.floodZone.sfha === "T"
          ? 1
          : 0,
    distance_sfha_m: fema.distanceToSfhaMeters,
    rainfall_100yr_24h_in:
      rainfallInches === null ? null : Math.round(rainfallInches * 100) / 100,
    nearest_water_distance_m: nearbyWater.distanceMeters,
    fema_zone: fema.floodZone.zone ?? "Unknown",
    fema_subtype: fema.floodZone.subtype ?? "Unknown",
    nearest_water_kind: nearbyWater.kind ?? "Unknown",
  });
  const expectedElevated = floodFactor >= elevatedRiskMinimum;
  const predictedElevated = prediction.classification === "claim-like";

  return {
    address: testCase.address,
    floodFactor,
    referenceLabel: testCase.risk_label,
    modelScore: prediction.score,
    modelLevel: modelRiskPresentation(prediction.probability).level,
    passed: expectedElevated === predictedElevated,
  };
}

async function main() {
  const testCases = parseCsv(await readFile(inputPath, "utf8"));
  const results = [];

  for (const testCase of testCases) {
    results.push(await evaluateCase(testCase));
  }

  for (const result of results) {
    console.log(
      `${result.passed ? "PASS" : "FAIL"} ${result.address}: ` +
        `First Street ${result.floodFactor}/10 (${result.referenceLabel}); ` +
        `model ${result.modelScore}/100 (${result.modelLevel})`,
    );
  }

  const passed = results.filter((result) => result.passed).length;
  console.log(`Matched ${passed}/${results.length} First Street risk classes.`);
  if (passed !== results.length) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
