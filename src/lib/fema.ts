import { readFile } from "node:fs/promises";
import path from "node:path";
import type { FloodZoneInput } from "@/lib/flood-guidance";

type Position = [number, number];
type PolygonCoordinates = Position[][];
type MultiPolygonCoordinates = PolygonCoordinates[];
type FemaFeature = {
  properties: {
    FLD_ZONE?: string | null;
    ZONE_SUBTY?: string | null;
    SFHA_TF?: string | null;
  };
  geometry:
    | { type: "Polygon"; coordinates: PolygonCoordinates }
    | { type: "MultiPolygon"; coordinates: MultiPolygonCoordinates }
    | null;
};

type FemaCollection = {
  features: FemaFeature[];
};

type CountySource = {
  file: string;
  bounds: [number, number, number, number];
};

type FemaSnapshot = {
  sources: Array<{
    file: string;
    bounds: CountySource["bounds"];
  }>;
};

export type FemaLookup = {
  floodZone: FloodZoneInput;
  distanceToSfhaMeters: number | null;
};

const SOURCE_PADDING_DEGREES = 0.02;

const collectionPromises = new Map<string, Promise<FemaCollection>>();
let sourcePromise: Promise<CountySource[]> | undefined;

function loadCountySources() {
  if (!sourcePromise) {
    sourcePromise = readFile(
      path.join(process.cwd(), "data", "fema-nfhl-snapshot.json"),
      "utf8",
    ).then((contents) => {
      const snapshot = JSON.parse(contents) as FemaSnapshot;
      return snapshot.sources.map(({ file, bounds }) => ({
        file,
        bounds: [
          bounds[0] - SOURCE_PADDING_DEGREES,
          bounds[1] - SOURCE_PADDING_DEGREES,
          bounds[2] + SOURCE_PADDING_DEGREES,
          bounds[3] + SOURCE_PADDING_DEGREES,
        ] as CountySource["bounds"],
      }));
    });
  }
  return sourcePromise;
}

function loadCollection(file: string) {
  let collectionPromise = collectionPromises.get(file);
  if (!collectionPromise) {
    collectionPromise = readFile(path.join(process.cwd(), "FEMAdata", file), "utf8")
      .then((contents) => JSON.parse(contents) as FemaCollection);
    collectionPromises.set(file, collectionPromise);
  }
  return collectionPromise;
}

function pointInRing(point: Position, ring: Position[]) {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [x, y] = ring[index];
    const [previousX, previousY] = ring[previous];
    if (distanceToSegmentMeters(point, ring[previous], ring[index]) <= 0.01) {
      return "boundary";
    }
    const intersects =
      y > point[1] !== previousY > point[1] &&
      point[0] <
        ((previousX - x) * (point[1] - y)) / (previousY - y) + x;
    if (intersects) inside = !inside;
  }
  return inside ? "inside" : "outside";
}

function pointInPolygon(point: Position, polygon: Position[][]) {
  const exterior = pointInRing(point, polygon[0]);
  if (exterior === "outside") return false;
  if (exterior === "boundary") return true;

  for (const hole of polygon.slice(1)) {
    const relation = pointInRing(point, hole);
    if (relation === "boundary") return true;
    if (relation === "inside") return false;
  }
  return true;
}

function distanceToSegmentMeters(point: Position, start: Position, end: Position) {
  const latitudeScale = 111_320;
  const longitudeScale = latitudeScale * Math.cos((point[1] * Math.PI) / 180);
  const pointX = point[0] * longitudeScale;
  const pointY = point[1] * latitudeScale;
  const startX = start[0] * longitudeScale;
  const startY = start[1] * latitudeScale;
  const endX = end[0] * longitudeScale;
  const endY = end[1] * latitudeScale;
  const dx = endX - startX;
  const dy = endY - startY;
  const lengthSquared = dx * dx + dy * dy;
  const projection =
    lengthSquared === 0
      ? 0
      : Math.max(
          0,
          Math.min(1, ((pointX - startX) * dx + (pointY - startY) * dy) / lengthSquared),
        );
  return Math.hypot(
    pointX - (startX + projection * dx),
    pointY - (startY + projection * dy),
  );
}

function distanceToFeatureMeters(point: Position, feature: FemaFeature) {
  if (!feature.geometry) return Number.POSITIVE_INFINITY;
  const polygons =
    feature.geometry.type === "Polygon"
      ? [feature.geometry.coordinates]
      : feature.geometry.coordinates;

  if (polygons.some((polygon) => pointInPolygon(point, polygon))) {
    return 0;
  }

  let minimum = Number.POSITIVE_INFINITY;
  for (const polygon of polygons) {
    for (const ring of polygon) {
      for (let index = 1; index < ring.length; index += 1) {
        minimum = Math.min(
          minimum,
          distanceToSegmentMeters(point, ring[index - 1], ring[index]),
        );
      }
    }
  }
  return minimum;
}

function isInsideBounds(point: Position, bounds: CountySource["bounds"]) {
  return (
    point[0] >= bounds[0] &&
    point[0] <= bounds[2] &&
    point[1] >= bounds[1] &&
    point[1] <= bounds[3]
  );
}

function hazardPriority(feature: FemaFeature) {
  const zone = feature.properties.FLD_ZONE?.toUpperCase() ?? "";
  if (["V", "VE"].includes(zone)) return 5;
  if (feature.properties.SFHA_TF === "T") return 4;
  if (feature.properties.ZONE_SUBTY?.toUpperCase().includes("0.2 PCT")) return 3;
  if (zone === "X") return 2;
  if (zone === "D") return 1;
  return 0;
}

export async function lookupFema(longitude: number, latitude: number): Promise<FemaLookup> {
  const point: Position = [longitude, latitude];
  const countySources = await loadCountySources();
  const sources = countySources.filter((source) => isInsideBounds(point, source.bounds));
  const collections = await Promise.all(sources.map((source) => loadCollection(source.file)));
  let containingFeature: FemaFeature | null = null;
  let distanceToSfhaMeters = Number.POSITIVE_INFINITY;

  for (const collection of collections) {
    for (const feature of collection.features) {
      const distance = distanceToFeatureMeters(point, feature);
      if (
        distance === 0 &&
        (!containingFeature || hazardPriority(feature) > hazardPriority(containingFeature))
      ) {
        containingFeature = feature;
      }
      if (feature.properties.SFHA_TF === "T") {
        distanceToSfhaMeters = Math.min(distanceToSfhaMeters, distance);
      }
    }
  }

  return {
    floodZone: {
      zone: containingFeature?.properties.FLD_ZONE ?? null,
      subtype: containingFeature?.properties.ZONE_SUBTY ?? null,
      sfha: containingFeature?.properties.SFHA_TF ?? null,
    },
    distanceToSfhaMeters: Number.isFinite(distanceToSfhaMeters)
      ? Math.round(distanceToSfhaMeters)
      : null,
  };
}
