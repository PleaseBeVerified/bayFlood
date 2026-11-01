import { lookupMappedFloodZone, mappedAnnualChance, type MappedFloodZone } from "./fema-event-benchmark.ts";

export const FEMA_NFHL_QUERY_URL =
  "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28/query";

export type LiveFemaResult =
  | { status: "matched"; zone: MappedFloodZone }
  | { status: "no_match" | "unavailable" };

export type FemaZoneResult = {
  zone: MappedFloodZone | null;
  source: "local_snapshot" | "live_nfhl" | "none";
  liveStatus: "matched" | "no_match" | "unavailable";
};

function nullableString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export async function lookupLiveFemaFloodZone(
  longitude: number,
  latitude: number,
  fetcher: typeof fetch = fetch,
): Promise<LiveFemaResult> {
  const url = new URL(FEMA_NFHL_QUERY_URL);
  url.searchParams.set("f", "json");
  url.searchParams.set("where", "1=1");
  url.searchParams.set("geometry", `${longitude},${latitude}`);
  url.searchParams.set("geometryType", "esriGeometryPoint");
  url.searchParams.set("inSR", "4326");
  url.searchParams.set("spatialRel", "esriSpatialRelIntersects");
  url.searchParams.set("outFields", "FLD_ZONE,ZONE_SUBTY,SFHA_TF");
  url.searchParams.set("returnGeometry", "false");
  url.searchParams.set("resultRecordCount", "10");

  try {
    const response = await fetcher(url, {
      headers: { Accept: "application/json", "User-Agent": "Mozilla/5.0" },
      cache: "no-store",
      signal: AbortSignal.timeout(7000),
    });
    if (!response.ok) return { status: "unavailable" };

    const payload: unknown = await response.json();
    if (typeof payload !== "object" || payload === null || !("features" in payload)
      || !Array.isArray(payload.features)) return { status: "unavailable" };
    if (payload.features.length === 0) return { status: "no_match" };

    let selected: MappedFloodZone | null = null;
    let bestChance = -1;
    for (const feature of payload.features) {
      const attributes = feature?.attributes;
      if (typeof attributes !== "object" || attributes === null) continue;
      const zone: MappedFloodZone = {
        zone: nullableString(attributes.FLD_ZONE),
        subtype: nullableString(attributes.ZONE_SUBTY),
        sfha: nullableString(attributes.SFHA_TF),
      };
      if (zone.zone === null) continue;
      const chance = mappedAnnualChance(zone) ?? 0;
      if (selected === null || chance > bestChance) {
        selected = zone;
        bestChance = chance;
      }
    }
    return selected === null ? { status: "unavailable" } : { status: "matched", zone: selected };
  } catch {
    return { status: "unavailable" };
  }
}

export async function lookupFemaZoneWithLocalFallback(
  longitude: number,
  latitude: number,
  localLookup: typeof lookupMappedFloodZone = lookupMappedFloodZone,
  liveLookup: typeof lookupLiveFemaFloodZone = lookupLiveFemaFloodZone,
): Promise<FemaZoneResult> {
  const live = await liveLookup(longitude, latitude);
  if (live.status === "matched") {
    return { zone: live.zone, source: "live_nfhl", liveStatus: "matched" };
  }
  const localZone = await localLookup(longitude, latitude);
  if (localZone !== null) {
    return { zone: localZone, source: "local_snapshot", liveStatus: live.status };
  }
  return { zone: null, source: "none", liveStatus: live.status };
}
