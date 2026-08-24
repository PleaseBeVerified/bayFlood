import { createHash } from "node:crypto";
import { access, mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawn } from "node:child_process";
import {
  FEMA_NFHL_LAYER_URL,
  FEMA_SOURCES,
  fetchOfficialGeoJson,
} from "./lib/fema-nfhl.mjs";

const root = process.cwd();
const femaDirectory = path.join(root, "FEMAdata");
const publicDataDirectory = path.join(root, "public", "data");
const snapshotPath = path.join(root, "data", "fema-nfhl-snapshot.json");
const pmtilesPath = path.join(publicDataDirectory, "bay-area-flood-zones.pmtiles");
const temporaryDirectory = path.join(femaDirectory, ".nfhl-sync-work");

function sha256(contents) {
  return createHash("sha256").update(contents).digest("hex");
}

function geometryBounds(collection) {
  const bounds = [Infinity, Infinity, -Infinity, -Infinity];

  function visit(coordinates) {
    if (
      Array.isArray(coordinates) &&
      coordinates.length >= 2 &&
      typeof coordinates[0] === "number" &&
      typeof coordinates[1] === "number"
    ) {
      bounds[0] = Math.min(bounds[0], coordinates[0]);
      bounds[1] = Math.min(bounds[1], coordinates[1]);
      bounds[2] = Math.max(bounds[2], coordinates[0]);
      bounds[3] = Math.max(bounds[3], coordinates[1]);
      return;
    }
    for (const value of coordinates ?? []) visit(value);
  }

  for (const feature of collection.features) visit(feature.geometry?.coordinates);
  if (!bounds.every(Number.isFinite)) throw new Error("FEMA collection has no valid geometry");
  return bounds;
}

function validateCollection(collection, source) {
  const ids = new Set();
  let nullGeometryCount = 0;
  for (const feature of collection.features) {
    if (feature.properties?.DFIRM_ID !== source.dfirmId) {
      throw new Error(`${source.name} contains a feature from another DFIRM`);
    }
    const id = feature.properties?.FLD_AR_ID;
    if (!id || ids.has(id)) {
      throw new Error(`${source.name} contains a missing or duplicate FLD_AR_ID`);
    }
    ids.add(id);
    if (!feature.geometry) {
      nullGeometryCount += 1;
    } else if (!["Polygon", "MultiPolygon"].includes(feature.geometry.type)) {
      throw new Error(`${source.name} contains unsupported ${feature.geometry.type} geometry`);
    }
  }
  return nullGeometryCount;
}

function run(command, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: root, stdio: "inherit" });
    child.on("error", reject);
    child.on("exit", (code) => {
      if (code === 0) resolve();
      else reject(new Error(`${command} exited with status ${code}`));
    });
  });
}

await mkdir(femaDirectory, { recursive: true });
await mkdir(publicDataDirectory, { recursive: true });
await mkdir(temporaryDirectory, { recursive: true });

try {
  const downloaded = [];
  for (const source of FEMA_SOURCES) {
    const temporaryPath = path.join(temporaryDirectory, source.file);
    let collection;
    try {
      await access(temporaryPath);
      process.stdout.write(`Checking staged ${source.name} (${source.dfirmId})... `);
      collection = JSON.parse(await readFile(temporaryPath, "utf8"));
    } catch {
      process.stdout.write(`Downloading ${source.name} (${source.dfirmId})... `);
      collection = await fetchOfficialGeoJson(source.dfirmId, {
        pageSize: source.geometryPageSize,
        byObjectId: source.geometryByObjectId,
      });
    }
    const nullGeometryCount = validateCollection(collection, source);
    const contents = `${JSON.stringify(collection)}\n`;
    await writeFile(temporaryPath, contents);
    downloaded.push({
      ...source,
      path: temporaryPath,
      featureCount: collection.features.length,
      nullGeometryCount,
      bounds: geometryBounds(collection),
      sha256: sha256(contents),
    });
    console.log(
      `${collection.features.length.toLocaleString()} features` +
        (nullGeometryCount ? ` (${nullGeometryCount} with no FEMA geometry)` : ""),
    );
  }

  const temporaryPmtiles = path.join(temporaryDirectory, "bay-area-flood-zones.pmtiles");
  console.log("Building browser PMTiles archive...");
  await run("tippecanoe", [
    "--force",
    "-o",
    temporaryPmtiles,
    "-l",
    "flood_zones",
    "-zg",
    "--drop-densest-as-needed",
    "--extend-zooms-if-still-dropping",
    "--detect-shared-borders",
    ...downloaded.map((source) => source.path),
  ]);

  const pmtilesContents = await readFile(temporaryPmtiles);
  const snapshot = {
    schemaVersion: 1,
    source: {
      organization: "Federal Emergency Management Agency",
      dataset: "National Flood Hazard Layer - Flood Hazard Zones",
      layerUrl: FEMA_NFHL_LAYER_URL,
      fetchedAt: new Date().toISOString(),
      outputSpatialReference: "EPSG:4326",
    },
    sources: downloaded.map((source) => ({
      name: source.name,
      file: source.file,
      dfirmId: source.dfirmId,
      featureCount: source.featureCount,
      nullGeometryCount: source.nullGeometryCount,
      bounds: source.bounds,
      sha256: source.sha256,
    })),
    pmtiles: {
      file: "public/data/bay-area-flood-zones.pmtiles",
      sha256: sha256(pmtilesContents),
    },
  };

  for (const source of downloaded) {
    await rename(source.path, path.join(femaDirectory, source.file));
  }
  await rename(temporaryPmtiles, pmtilesPath);
  await writeFile(snapshotPath, `${JSON.stringify(snapshot, null, 2)}\n`);
  console.log(`FEMA NFHL snapshot updated at ${snapshot.source.fetchedAt}`);
  await rm(temporaryDirectory, { recursive: true, force: true });
} catch (error) {
  console.error(`FEMA sync stopped. Verified staged downloads remain in ${temporaryDirectory}.`);
  throw error;
}
