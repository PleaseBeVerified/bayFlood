import { readFile } from "node:fs/promises";
import path from "node:path";

type Position = [number, number];
type Bounds = [number, number, number, number];
type FloodFeature = {
  properties: { FLD_ZONE?: string | null; ZONE_SUBTY?: string | null; SFHA_TF?: string | null };
  geometry: { type: "Polygon"; coordinates: Position[][] } | null;
};
type FloodCollection = { features: FloodFeature[] };

const SOURCES: Array<{ file: string; bounds: Bounds }> = [
  { file: "alameda_county_flood_zones.geojson", bounds: [-122.37386, 37.45396, -121.4691, 37.90581] },
  { file: "contra_costa_county_flood_zones.geojson", bounds: [-122.44138, 37.71848, -121.53429, 38.10452] },
  { file: "marin_county.geojson", bounds: [-123.028, 37.81317, -122.41874, 38.32117] },
  { file: "napa_county.geojson", bounds: [-122.64682, 38.15487, -122.06141, 38.86421] },
  { file: "san_francisco_county_flood_zones.geojson", bounds: [-123.11112, 37.59222, -122.28149, 37.92982] },
  { file: "san_joaquin_boundary_flood_zones.geojson", bounds: [-121.58488, 37.48191, -120.91705, 38.30013] },
  { file: "san_mateo_county_flood_zones.geojson", bounds: [-122.52681, 37.10553, -122.08171, 37.70832] },
  { file: "santa_clara_flood_zones.geojson", bounds: [-122.20267, 36.89312, -121.20831, 37.48464] },
  { file: "solano_county.geojson", bounds: [-122.40706, 38.03191, -121.59333, 38.54031] },
  { file: "sonoma_county_flood_zones.geojson", bounds: [-123.5374, 38.06946, -122.34971, 38.85266] },
];

const collections = new Map<string, Promise<FloodCollection>>();

function loadCollection(file: string) {
  let pending = collections.get(file);
  if (!pending) {
    pending = readFile(path.join(process.cwd(), "FEMAdata", file), "utf8")
      .then((contents) => JSON.parse(contents) as FloodCollection)
      .catch((error) => {
        collections.delete(file);
        throw error;
      });
    collections.set(file, pending);
  }
  return pending;
}

function insideBounds([longitude, latitude]: Position, [west, south, east, north]: Bounds) {
  const padding = 0.001;
  return longitude >= west - padding && longitude <= east + padding
    && latitude >= south - padding && latitude <= north + padding;
}

function pointInRing([x, y]: Position, ring: Position[]) {
  let inside = false;
  for (let index = 0, previous = ring.length - 1; index < ring.length; previous = index++) {
    const [ax, ay] = ring[previous];
    const [bx, by] = ring[index];
    const cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax);
    if (Math.abs(cross) < 1e-10 && x >= Math.min(ax, bx) && x <= Math.max(ax, bx)
      && y >= Math.min(ay, by) && y <= Math.max(ay, by)) return true;
    if ((ay > y) !== (by > y) && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
  }
  return inside;
}

function pointInFeature(point: Position, feature: FloodFeature) {
  if (feature.geometry?.type !== "Polygon") return false;
  const [exterior, ...holes] = feature.geometry.coordinates;
  return pointInRing(point, exterior) && !holes.some((hole) => pointInRing(point, hole));
}

export type MappedFloodZone = {
  zone: string | null;
  subtype: string | null;
  sfha: string | null;
};

export function mappedAnnualChance(zone: MappedFloodZone): number | null {
  const code = zone.zone?.toUpperCase().trim() ?? "";
  const subtype = zone.subtype?.toUpperCase().trim() ?? "";
  if (zone.sfha?.toUpperCase() === "T" || /^(A|V)(\d|$|E|H|O|R|99)/.test(code)) return 0.01;
  if (code === "X" && /^1[- ]PCT (DEPTH LESS THAN 1 FOOT|DRAINAGE AREA LESS THAN 1 SQUARE MILE)$/.test(subtype)) return 0.01;
  if (code === "X" && subtype.includes("0.2 PCT ANNUAL CHANCE")) return 0.002;
  return null;
}

export function isMinimalHazardZone(zone: MappedFloodZone): boolean {
  return zone.zone?.toUpperCase().trim() === "X"
    && zone.subtype?.toUpperCase().trim() === "AREA OF MINIMAL FLOOD HAZARD";
}

export function cumulativeEventChance(annualChance: number, years: 10 | 20) {
  return Math.round((1 - (1 - annualChance) ** years) * 1_000) / 10;
}

export function cumulativeEventUpperBound(annualChance: number, years: 10 | 20) {
  return Math.ceil((1 - (1 - annualChance) ** years) * 1_000) / 10;
}

export async function lookupMappedFloodZone(longitude: number, latitude: number) {
  const point: Position = [longitude, latitude];
  const candidates = SOURCES.filter((source) => insideBounds(point, source.bounds));
  if (candidates.length === 0) return null;
  const results = await Promise.allSettled(candidates.map((source) => loadCollection(source.file)));
  let selected: MappedFloodZone | null = null;
  let bestChance = -1;
  for (const result of results) {
    if (result.status !== "fulfilled") continue;
    for (const feature of result.value.features) {
      if (!pointInFeature(point, feature)) continue;
      const zone = {
        zone: feature.properties.FLD_ZONE ?? null,
        subtype: feature.properties.ZONE_SUBTY ?? null,
        sfha: feature.properties.SFHA_TF ?? null,
      };
      const chance = mappedAnnualChance(zone) ?? 0;
      if (!selected || chance > bestChance) {
        selected = zone;
        bestChance = chance;
      }
    }
  }
  return selected;
}
