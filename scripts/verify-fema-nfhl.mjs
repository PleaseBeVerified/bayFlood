import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { lookupFema } from "../src/lib/fema.ts";
import {
  FEMA_SOURCES,
  fetchOfficialAttributes,
  fetchOfficialPoint,
} from "./lib/fema-nfhl.mjs";

const root = process.cwd();
const snapshot = JSON.parse(
  await readFile(path.join(root, "data", "fema-nfhl-snapshot.json"), "utf8"),
);
const snapshotSources = new Map(snapshot.sources.map((source) => [source.dfirmId, source]));
const comparedFields = ["DFIRM_ID", "FLD_AR_ID", "FLD_ZONE", "ZONE_SUBTY", "SFHA_TF", "SOURCE_CIT"];
let failures = 0;

function sha256(contents) {
  return createHash("sha256").update(contents).digest("hex");
}

function normalizedAttributes(properties) {
  return Object.fromEntries(comparedFields.map((field) => [field, properties[field] ?? null]));
}

function hazardPriority(properties) {
  const zone = properties?.FLD_ZONE?.toUpperCase() ?? "";
  if (["V", "VE"].includes(zone)) return 5;
  if (properties?.SFHA_TF === "T") return 4;
  if (properties?.ZONE_SUBTY?.toUpperCase().includes("0.2 PCT")) return 3;
  if (zone === "X") return 2;
  if (zone === "D") return 1;
  return 0;
}

function comparableZone(properties) {
  return {
    zone: properties?.FLD_ZONE ?? null,
    subtype: properties?.ZONE_SUBTY ?? null,
    sfha: properties?.SFHA_TF ?? null,
  };
}

const pmtilesContents = await readFile(path.join(root, snapshot.pmtiles.file));
if (sha256(pmtilesContents) !== snapshot.pmtiles.sha256) {
  failures += 1;
  console.log("PMTiles FAILED: checksum does not match the FEMA snapshot");
} else {
  console.log("PMTiles checksum matches the FEMA snapshot");
}

for (const source of FEMA_SOURCES) {
  process.stdout.write(`Verifying ${source.name}... `);
  const localContents = await readFile(path.join(root, "FEMAdata", source.file), "utf8");
  const localCollection = JSON.parse(localContents);
  const recorded = snapshotSources.get(source.dfirmId);
  const officialAttributes = await fetchOfficialAttributes(source.dfirmId);
  const localById = new Map(
    localCollection.features.map((feature) => [feature.properties.FLD_AR_ID, feature.properties]),
  );
  const officialById = new Map(
    officialAttributes.map((properties) => [properties.FLD_AR_ID, properties]),
  );
  const issues = [];

  if (!recorded) issues.push("missing snapshot metadata");
  if (recorded?.sha256 !== sha256(localContents)) issues.push("snapshot checksum mismatch");
  if (recorded?.featureCount !== localCollection.features.length) issues.push("snapshot count mismatch");
  if (localById.size !== officialById.size) {
    issues.push(`feature count ${localById.size} local / ${officialById.size} official`);
  }

  for (const [id, official] of officialById) {
    const local = localById.get(id);
    if (!local) {
      issues.push(`missing ${id}`);
      continue;
    }
    if (JSON.stringify(normalizedAttributes(local)) !== JSON.stringify(normalizedAttributes(official))) {
      issues.push(`changed attributes for ${id}`);
    }
  }
  for (const id of localById.keys()) {
    if (!officialById.has(id)) issues.push(`removed upstream ${id}`);
  }

  if (issues.length > 0) {
    failures += issues.length;
    console.log(`FAILED\n  ${issues.slice(0, 10).join("\n  ")}`);
    if (issues.length > 10) console.log(`  ...and ${issues.length - 10} more`);
  } else {
    console.log(`${localById.size.toLocaleString()} features match`);
  }
}

const samplePoints = [
  { name: "San Francisco", longitude: -122.4194, latitude: 37.7749 },
  { name: "SFO", longitude: -122.388186, latitude: 37.614532 },
  { name: "Palo Alto", longitude: -122.1601873, latitude: 37.4512686 },
  { name: "Oakland", longitude: -122.2712, latitude: 37.8044 },
  { name: "San Jose", longitude: -121.8863, latitude: 37.3382 },
  { name: "San Rafael", longitude: -122.5311, latitude: 37.9735 },
  { name: "Napa", longitude: -122.2869, latitude: 38.2975 },
  { name: "Fairfield", longitude: -122.04, latitude: 38.2494 },
  { name: "Santa Rosa", longitude: -122.7144, latitude: 38.4405 },
  { name: "Carquinez boundary", longitude: -122.4, latitude: 38 },
  { name: "Eastern Contra Costa boundary", longitude: -121.579105, latitude: 37.859116 },
];

for (const point of samplePoints) {
  const [local, officialFeatures] = await Promise.all([
    lookupFema(point.longitude, point.latitude),
    fetchOfficialPoint(point.longitude, point.latitude),
  ]);
  const official = officialFeatures.sort((a, b) => hazardPriority(b) - hazardPriority(a))[0];
  const officialZone = comparableZone(official);
  if (JSON.stringify(local.floodZone) !== JSON.stringify(officialZone)) {
    failures += 1;
    console.log(
      `Point FAILED: ${point.name}\n  local ${JSON.stringify(local.floodZone)}\n  official ${JSON.stringify(officialZone)}`,
    );
  } else {
    console.log(`Point matched: ${point.name} (${officialZone.zone ?? "not mapped"})`);
  }
}

if (failures > 0) {
  console.error(`FEMA verification failed with ${failures} issue(s).`);
  process.exitCode = 1;
} else {
  console.log(`FEMA verification passed for all ${FEMA_SOURCES.length} sources and ${samplePoints.length} points.`);
}
