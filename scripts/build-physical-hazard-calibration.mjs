import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { sampleElevation } from "../src/lib/elevation.ts";
import { sampleRainfall } from "../src/lib/rainfall.ts";
import { sampleTerrainFeatures } from "../src/lib/terrain.ts";
import { lookupNearbyWater } from "../src/lib/water.ts";
import { parseCsv, serializeCsv } from "./lib/property-observations.mjs";

const countArgument = process.argv.find((argument) => argument.startsWith("--count="));
const outputArgument = process.argv.find((argument) => argument.startsWith("--output="));
const hazardArgument = process.argv.find((argument) => argument.startsWith("--hazard="));
const partitionsArgument = process.argv.find((argument) =>
  argument.startsWith("--partitions="),
);
const excludeArgument = process.argv.find((argument) =>
  argument.startsWith("--exclude="),
);
const requestedCount = Number(countArgument?.slice(8) ?? 10_000);
const outputPath = outputArgument?.slice(9) ?? "data/physical-hazard-calibration.csv";
const requestedHazard = hazardArgument?.slice(9) ?? null;
const requestedPartitions = new Set(
  (partitionsArgument?.slice(13) ?? "training,validation,test")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean),
);
const excludePath = excludeArgument?.slice(10) ?? null;
const snapshotPath = "data/fema-nfhl-snapshot.json";
const seed = 2_026_081_0;
const hazardClasses = new Set(["one_percent", "point_two_percent", "minimal"]);
const partitionNames = new Set(["training", "validation", "test"]);
const sampleIdPrefix = requestedHazard
  ? `physical-${requestedHazard.replaceAll("_percent", "").replaceAll("_", "-")}-extra`
  : "physical";

if (process.argv.includes("--help")) {
  console.log(`Usage:
  npm run calibration:physical-table -- [options]

Options:
  --count=N                 Number of new rows to generate (minimum 300)
  --hazard=CLASS            one_percent, point_two_percent, or minimal
  --partitions=NAMES        Comma-separated training,validation,test block groups
  --exclude=CSV             Skip six-decimal coordinates already in this CSV
  --output=CSV              Output CSV path`);
  process.exit(0);
}

if (!Number.isInteger(requestedCount) || requestedCount < 300) {
  throw new Error("--count must be an integer of at least 300.");
}
if (requestedHazard !== null && !hazardClasses.has(requestedHazard)) {
  throw new Error(
    "--hazard must be one_percent, point_two_percent, or minimal.",
  );
}
if (
  requestedPartitions.size === 0 ||
  [...requestedPartitions].some((value) => !partitionNames.has(value))
) {
  throw new Error(
    "--partitions must contain training, validation, and/or test.",
  );
}

function mulberry32(initialSeed) {
  let state = initialSeed;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4_294_967_296;
  };
}

function flattenPositions(geometry) {
  if (geometry.type === "Polygon") return geometry.coordinates.flat();
  return geometry.coordinates.flat(2);
}

function geometryBounds(geometry) {
  const positions = flattenPositions(geometry);
  let west = Number.POSITIVE_INFINITY;
  let south = Number.POSITIVE_INFINITY;
  let east = Number.NEGATIVE_INFINITY;
  let north = Number.NEGATIVE_INFINITY;
  for (const [longitude, latitude] of positions) {
    west = Math.min(west, longitude);
    south = Math.min(south, latitude);
    east = Math.max(east, longitude);
    north = Math.max(north, latitude);
  }
  return [west, south, east, north];
}

function pointInBounds(longitude, latitude, bounds) {
  return (
    longitude >= bounds[0] &&
    longitude <= bounds[2] &&
    latitude >= bounds[1] &&
    latitude <= bounds[3]
  );
}

function pointInRing(longitude, latitude, ring) {
  let inside = false;
  for (
    let index = 0, previous = ring.length - 1;
    index < ring.length;
    previous = index, index += 1
  ) {
    const [x, y] = ring[index];
    const [previousX, previousY] = ring[previous];
    if (
      y > latitude !== previousY > latitude &&
      longitude <
        ((previousX - x) * (latitude - y)) / (previousY - y) + x
    ) {
      inside = !inside;
    }
  }
  return inside;
}

function pointInPolygon(longitude, latitude, polygon) {
  if (!pointInRing(longitude, latitude, polygon[0])) return false;
  return !polygon.slice(1).some((ring) => pointInRing(longitude, latitude, ring));
}

function pointInGeometry(longitude, latitude, geometry) {
  if (geometry.type === "Polygon") {
    return pointInPolygon(longitude, latitude, geometry.coordinates);
  }
  return geometry.coordinates.some((polygon) =>
    pointInPolygon(longitude, latitude, polygon),
  );
}

function hazardClass(properties) {
  const zone = String(properties.FLD_ZONE ?? "").toUpperCase().trim();
  const subtype = String(properties.ZONE_SUBTY ?? "").toUpperCase().trim();
  if (
    properties.SFHA_TF === "T" ||
    ["A", "AE", "A1-30", "AO", "AH", "AR", "A99", "V", "VE", "V1-30"].includes(
      zone,
    )
  ) {
    return "one_percent";
  }
  if (subtype.includes("0.2 PCT") || subtype.includes("MODERATE FLOOD HAZARD")) {
    return "point_two_percent";
  }
  if (zone === "X" && subtype.includes("MINIMAL FLOOD HAZARD")) {
    return "minimal";
  }
  return null;
}

function hazardPriority(value) {
  if (value === "one_percent") return 3;
  if (value === "point_two_percent") return 2;
  if (value === "minimal") return 1;
  return 0;
}

function containingHazardClass(longitude, latitude, features) {
  let selected = null;
  for (const feature of features) {
    if (
      hazardPriority(feature.hazardClass) <= hazardPriority(selected) ||
      !pointInBounds(longitude, latitude, feature.bounds)
    ) {
      continue;
    }
    if (pointInGeometry(longitude, latitude, feature.geometry)) {
      selected = feature.hazardClass;
    }
  }
  return selected;
}

function surfaceArea(bounds) {
  const middleLatitude = (bounds[1] + bounds[3]) / 2;
  return (
    (bounds[2] - bounds[0]) *
    (bounds[3] - bounds[1]) *
    Math.cos((middleLatitude * Math.PI) / 180)
  );
}

function stringHash(value) {
  let hash = 2_166_136_261;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16_777_619);
  }
  return hash >>> 0;
}

function spatialPartition(longitude, latitude) {
  const latitudeTile = Math.floor(latitude / 0.05);
  const longitudeTile = Math.floor(longitude / 0.05);
  const bucket = stringHash(`${latitudeTile},${longitudeTile}`) % 20;
  if (bucket < 14) return "training";
  if (bucket < 17) return "validation";
  return "test";
}

function selectWeightedFeature(features, totalWeight, random) {
  let selection = random() * totalWeight;
  for (const feature of features) {
    selection -= feature.samplingWeight;
    if (selection <= 0) return feature;
  }
  return features.at(-1);
}

function allocateCounts(sources, totalCount) {
  const totalArea = sources.reduce(
    (sum, source) => sum + surfaceArea(source.bounds),
    0,
  );
  const allocations = sources.map((source) => ({
    ...source,
    exactCount: (surfaceArea(source.bounds) / totalArea) * totalCount,
  }));
  let assigned = 0;
  for (const allocation of allocations) {
    allocation.count = Math.floor(allocation.exactCount);
    assigned += allocation.count;
  }
  allocations
    .sort(
      (a, b) =>
        b.exactCount - b.count - (a.exactCount - a.count),
    )
    .slice(0, totalCount - assigned)
    .forEach((allocation) => {
      allocation.count += 1;
    });
  return allocations.sort((a, b) => a.name.localeCompare(b.name));
}

async function sampleSource(
  source,
  random,
  count,
  completedBefore,
  excludedCoordinates,
) {
  const collection = JSON.parse(
    await readFile(path.join("FEMAdata", source.file), "utf8"),
  );
  const features = collection.features.flatMap((feature) => {
    if (!feature.geometry) return [];
    const selectedClass = hazardClass(feature.properties ?? {});
    if (!selectedClass) return [];
    return [{
      hazardClass: selectedClass,
      geometry: feature.geometry,
      bounds: geometryBounds(feature.geometry),
    }];
  });
  const targetedFeatures = requestedHazard
    ? features
        .filter((feature) => feature.hazardClass === requestedHazard)
        .map((feature) => ({
          ...feature,
          samplingWeight: Math.max(surfaceArea(feature.bounds), 1e-12),
        }))
    : [];
  const totalTargetedWeight = targetedFeatures.reduce(
    (sum, feature) => sum + feature.samplingWeight,
    0,
  );
  const rows = [];
  const seen = new Set();
  const maximumAttempts = Math.max(20_000, count * 250);

  for (let attempt = 0; rows.length < count && attempt < maximumAttempts; attempt += 1) {
    const targetedFeature = requestedHazard
      ? selectWeightedFeature(targetedFeatures, totalTargetedWeight, random)
      : null;
    if (requestedHazard && !targetedFeature) break;
    const samplingBounds = targetedFeature?.bounds ?? source.bounds;
    const longitude =
      samplingBounds[0] + random() * (samplingBounds[2] - samplingBounds[0]);
    const latitude =
      samplingBounds[1] + random() * (samplingBounds[3] - samplingBounds[1]);
    const coordinateKey = `${latitude.toFixed(6)},${longitude.toFixed(6)}`;
    if (
      seen.has(coordinateKey) ||
      excludedCoordinates.has(coordinateKey) ||
      !requestedPartitions.has(spatialPartition(longitude, latitude))
    ) {
      continue;
    }
    seen.add(coordinateKey);

    const selectedClass = containingHazardClass(longitude, latitude, features);
    if (
      !selectedClass ||
      (requestedHazard !== null && selectedClass !== requestedHazard)
    ) {
      continue;
    }

    const [elevationMeters, rainfallInches, nearbyWater, terrain] = await Promise.all([
      sampleElevation(longitude, latitude),
      sampleRainfall(longitude, latitude),
      lookupNearbyWater(longitude, latitude),
      sampleTerrainFeatures(longitude, latitude),
    ]);
    if (
      elevationMeters === null ||
      rainfallInches === null ||
      Object.values(terrain).some((value) => value === null)
    ) {
      continue;
    }

    rows.push({
      sample_id: `${sampleIdPrefix}-${String(completedBefore + rows.length + 1).padStart(5, "0")}`,
      county: source.name,
      lat: latitude.toFixed(6),
      lng: longitude.toFixed(6),
      mapped_hazard_class: selectedClass,
      elevation_m: (Math.round(elevationMeters * 10) / 10).toFixed(1),
      rainfall_100yr_24h_in: (Math.round(rainfallInches * 100) / 100).toFixed(2),
      nearest_water_status: nearbyWater.status,
      nearest_water_distance_m: nearbyWater.distanceMeters ?? "",
      nearest_water_kind: nearbyWater.kind ?? "Unknown",
      slope_degrees: terrain.slope_degrees.toFixed(3),
      local_elevation_250m: terrain.local_elevation_250m.toFixed(3),
      local_elevation_500m: terrain.local_elevation_500m.toFixed(3),
      flow_accumulation_m2: terrain.flow_accumulation_m2.toFixed(1),
      height_above_drainage_m: terrain.height_above_drainage_m.toFixed(3),
      label_source: "FEMA NFHL mapped hazard class",
    });
    excludedCoordinates.add(coordinateKey);
    if ((completedBefore + rows.length) % 100 === 0) {
      console.log(`Sampled ${completedBefore + rows.length}/${requestedCount}.`);
    }
  }

  if (rows.length < count) {
    console.warn(
      `${source.name}: requested ${count}, collected ${rows.length} covered samples.`,
    );
  }
  return rows;
}

async function main() {
  const snapshot = JSON.parse(await readFile(snapshotPath, "utf8"));
  const sources = snapshot.sources.filter(
    (source) => !source.name.toLowerCase().includes("san joaquin"),
  );
  const allocations = allocateCounts(sources, requestedCount);
  const random = mulberry32(seed);
  const rows = [];
  const excludedCoordinates = new Set();
  if (excludePath) {
    const excludedRows = parseCsv(await readFile(excludePath, "utf8")).rows;
    for (const row of excludedRows) {
      const latitude = Number(row.lat);
      const longitude = Number(row.lng);
      if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
        excludedCoordinates.add(
          `${latitude.toFixed(6)},${longitude.toFixed(6)}`,
        );
      }
    }
    console.log(
      `Excluding ${excludedCoordinates.size} existing coordinates from ${excludePath}.`,
    );
  }

  for (const source of allocations) {
    console.log(`Sampling ${source.name}: ${source.count} requested.`);
    rows.push(
      ...(await sampleSource(
        source,
        random,
        source.count,
        rows.length,
        excludedCoordinates,
      )),
    );
  }

  const headers = [
    "sample_id",
    "county",
    "lat",
    "lng",
    "mapped_hazard_class",
    "elevation_m",
    "rainfall_100yr_24h_in",
    "nearest_water_status",
    "nearest_water_distance_m",
    "nearest_water_kind",
    "slope_degrees",
    "local_elevation_250m",
    "local_elevation_500m",
    "flow_accumulation_m2",
    "height_above_drainage_m",
    "label_source",
  ];
  await writeFile(outputPath, serializeCsv(rows, headers));
  const counts = Object.fromEntries(
    ["one_percent", "point_two_percent", "minimal"].map((hazard) => [
      hazard,
      rows.filter((row) => row.mapped_hazard_class === hazard).length,
    ]),
  );
  console.log(
    `Wrote ${rows.length} new rows to ${outputPath}.`,
    counts,
    requestedHazard ? { requestedHazard, partitions: [...requestedPartitions] } : {},
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
