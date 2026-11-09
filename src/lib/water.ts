import { readFile } from "node:fs/promises";
import path from "node:path";

const SEARCH_RADIUS = 5_000;

type Position = [number, number];

type GeoJsonGeometry = {
  type: "LineString" | "MultiLineString" | "Polygon" | "MultiPolygon";
  coordinates: Position[] | Position[][] | Position[][][];
};

type GeoJsonFeature = {
  properties?: {
    GNIS_Name?: string | null;
    FType?: number | null;
    featuretypelabel?: string | null;
  };
  geometry?: GeoJsonGeometry | null;
};

type GeoJsonCollection = {
  features: GeoJsonFeature[];
};

type LocalWaterFeature = {
  name: string | null;
  kind: string;
  geometry: GeoJsonGeometry;
  bounds: [number, number, number, number];
};

export type NearbyWater = {
  status: "found" | "not_found" | "unavailable";
  distanceMeters: number | null;
  name: string | null;
  kind: string | null;
  searchRadiusMeters: number;
};

let localWaterFeaturesPromise: Promise<LocalWaterFeature[]> | undefined;

function waterKind(featureType?: number | null, featureTypeLabel?: string | null) {
  if (featureTypeLabel === "River" || featureTypeLabel === "Drainageway") return "stream/river";
  if (featureTypeLabel === "Channel Line") return "stream/river";
  if (featureTypeLabel === "Canal") return "canal/ditch";
  if (featureTypeLabel === "Waterbody Connector") return "waterbody";
  if (featureType === 336) return "canal/ditch";
  if (featureType === 343) return "dam/weir";
  if (featureType === 364) return "foreshore";
  if (featureType === 390) return "waterbody";
  if (featureType === 428) return "pipeline";
  if (featureType === 431) return "rapids";
  if (featureType === 434) return "reef";
  if (featureType === 436) return "reservoir";
  if (featureType === 445) return "sea/ocean";
  if (featureType === 450) return "sink/rise";
  if (featureType === 460) return "stream/river";
  if (featureType === 466) return "swamp/marsh";
  if (featureType === 558) return "artificial path";
  if (featureType === 566) return "coastline";
  return "water";
}

function flattenPositions(geometry: GeoJsonGeometry): Position[] {
  if (geometry.type === "LineString") return geometry.coordinates as Position[];
  if (geometry.type === "MultiLineString") return (geometry.coordinates as Position[][]).flat();
  if (geometry.type === "Polygon") return (geometry.coordinates as Position[][]).flat();
  return (geometry.coordinates as Position[][][]).flat(2);
}

function boundsForGeometry(geometry: GeoJsonGeometry): [number, number, number, number] {
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

async function loadLocalWaterFeatures() {
  const files = [
    "bay_area_flowlines.geojson",
    "bay_area_water_bodies.geojson",
    "NHDArea.geojson",
    "NHDLine.geojson",
    "santa_cruz_north_3dhp_flowlines.geojson",
    "tracy_lathrop_stockton_3dhp_flowlines.geojson",
  ];
  const collections = await Promise.all(
    files.map(async (file) => {
      const contents = await readFile(path.join(process.cwd(), "hydrography", file), "utf8");
      return JSON.parse(contents) as GeoJsonCollection;
    }),
  );

  return collections.flatMap((collection) =>
    collection.features.flatMap((feature) => {
      if (!feature.geometry) return [];

      return {
        name: feature.properties?.GNIS_Name ?? null,
        kind: waterKind(feature.properties?.FType, feature.properties?.featuretypelabel),
        geometry: feature.geometry,
        bounds: boundsForGeometry(feature.geometry),
      };
    }),
  );
}

function getLocalWaterFeatures() {
  localWaterFeaturesPromise ??= loadLocalWaterFeatures();
  return localWaterFeaturesPromise;
}

function distanceToPointMeters(
  longitude: number,
  latitude: number,
  targetLongitude: number,
  targetLatitude: number,
) {
  const latitudeScale = 111_320;
  const longitudeScale = latitudeScale * Math.cos((latitude * Math.PI) / 180);

  return Math.hypot(
    (longitude - targetLongitude) * longitudeScale,
    (latitude - targetLatitude) * latitudeScale,
  );
}

function boundsDistanceMeters(
  longitude: number,
  latitude: number,
  bounds: [number, number, number, number],
) {
  const [west, south, east, north] = bounds;
  const nearestLongitude = Math.max(west, Math.min(east, longitude));
  const nearestLatitude = Math.max(south, Math.min(north, latitude));
  return distanceToPointMeters(longitude, latitude, nearestLongitude, nearestLatitude);
}

function distanceToSegment(
  longitude: number,
  latitude: number,
  start: Position,
  end: Position,
) {
  const latitudeScale = 111_320;
  const longitudeScale = latitudeScale * Math.cos((latitude * Math.PI) / 180);
  const px = longitude * longitudeScale;
  const py = latitude * latitudeScale;
  const x1 = start[0] * longitudeScale;
  const y1 = start[1] * latitudeScale;
  const x2 = end[0] * longitudeScale;
  const y2 = end[1] * latitudeScale;
  const dx = x2 - x1;
  const dy = y2 - y1;
  const lengthSquared = dx * dx + dy * dy;
  const amount =
    lengthSquared === 0
      ? 0
      : Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lengthSquared));

  return Math.hypot(px - (x1 + amount * dx), py - (y1 + amount * dy));
}

function pointInRing(longitude: number, latitude: number, ring: Position[]) {
  let inside = false;

  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [x, y] = ring[index];
    const [previousX, previousY] = ring[previous];
    const intersects =
      y > latitude !== previousY > latitude &&
      longitude < ((previousX - x) * (latitude - y)) / (previousY - y) + x;

    if (intersects) inside = !inside;
  }

  return inside;
}

function pointInPolygon(longitude: number, latitude: number, polygon: Position[][]) {
  if (!pointInRing(longitude, latitude, polygon[0])) return false;
  return !polygon.slice(1).some((ring) => pointInRing(longitude, latitude, ring));
}

function distanceToLineCoordinates(longitude: number, latitude: number, line: Position[]) {
  let minimum = Number.POSITIVE_INFINITY;

  for (let index = 1; index < line.length; index += 1) {
    minimum = Math.min(
      minimum,
      distanceToSegment(longitude, latitude, line[index - 1], line[index]),
    );
  }

  return minimum;
}

function distanceToPolygonCoordinates(
  longitude: number,
  latitude: number,
  polygon: Position[][],
) {
  if (pointInPolygon(longitude, latitude, polygon)) return 0;
  return Math.min(...polygon.map((ring) => distanceToLineCoordinates(longitude, latitude, ring)));
}

function distanceToGeometry(longitude: number, latitude: number, geometry: GeoJsonGeometry) {
  if (geometry.type === "LineString") {
    return distanceToLineCoordinates(longitude, latitude, geometry.coordinates as Position[]);
  }

  if (geometry.type === "MultiLineString") {
    return Math.min(
      ...(geometry.coordinates as Position[][]).map((line) =>
        distanceToLineCoordinates(longitude, latitude, line),
      ),
    );
  }

  if (geometry.type === "Polygon") {
    return distanceToPolygonCoordinates(longitude, latitude, geometry.coordinates as Position[][]);
  }

  return Math.min(
    ...(geometry.coordinates as Position[][][]).map((polygon) =>
      distanceToPolygonCoordinates(longitude, latitude, polygon),
    ),
  );
}

export async function lookupNearbyWater(
  longitude: number,
  latitude: number,
): Promise<NearbyWater> {
  try {
    const features = await getLocalWaterFeatures();
    let nearest: NearbyWater = {
      status: "not_found",
      distanceMeters: null,
      name: null,
      kind: null,
      searchRadiusMeters: SEARCH_RADIUS,
    };

    for (const feature of features) {
      if (boundsDistanceMeters(longitude, latitude, feature.bounds) > SEARCH_RADIUS) {
        continue;
      }

      const distance = distanceToGeometry(longitude, latitude, feature.geometry);
      if (distance > SEARCH_RADIUS) continue;

      if (nearest.distanceMeters === null || distance < nearest.distanceMeters) {
        nearest = {
          status: "found",
          distanceMeters: Math.round(distance),
          name: feature.name,
          kind: feature.kind,
          searchRadiusMeters: SEARCH_RADIUS,
        };
      }
    }

    return nearest;
  } catch (error) {
    console.error("Nearby water lookup failed", error);
    return {
      status: "unavailable",
      distanceMeters: null,
      name: null,
      kind: null,
      searchRadiusMeters: SEARCH_RADIUS,
    };
  }
}
