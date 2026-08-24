export const FEMA_NFHL_LAYER_URL =
  "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28";

export const FEMA_SOURCES = [
  { name: "Alameda County", file: "alameda_county_flood_zones.geojson", dfirmId: "06001C" },
  { name: "Contra Costa County", file: "contra_costa_county_flood_zones.geojson", dfirmId: "06013C" },
  { name: "Marin County", file: "marin_county.geojson", dfirmId: "06041C" },
  { name: "Napa County", file: "napa_county.geojson", dfirmId: "06055C" },
  { name: "San Francisco County", file: "san_francisco_county_flood_zones.geojson", dfirmId: "060298" },
  { name: "San Mateo County", file: "san_mateo_county_flood_zones.geojson", dfirmId: "06081C" },
  { name: "Santa Clara County", file: "santa_clara_flood_zones.geojson", dfirmId: "06085C" },
  { name: "Solano County", file: "solano_county.geojson", dfirmId: "06095C" },
  {
    name: "San Joaquin County boundary supplement",
    file: "san_joaquin_boundary_flood_zones.geojson",
    dfirmId: "06077C",
    geometryPageSize: 250,
  },
  {
    name: "Sonoma County",
    file: "sonoma_county_flood_zones.geojson",
    dfirmId: "06097C",
    geometryPageSize: 25,
    geometryByObjectId: true,
  },
];

const PAGE_SIZE = 500;
const RETRYABLE_STATUSES = new Set([429, 500, 502, 503, 504]);

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function query(params) {
  const url = new URL(`${FEMA_NFHL_LAYER_URL}/query`);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, String(value));
  }

  for (let attempt = 1; attempt <= 5; attempt += 1) {
    let response;
    try {
      response = await fetch(url, {
        headers: { "user-agent": "Bay-Area-Flood-Risk-FEMA-Audit/1.0" },
        signal: AbortSignal.timeout(120_000),
      });
    } catch (error) {
      if (attempt === 5) throw error;
      await sleep(1_000 * 2 ** (attempt - 1));
      continue;
    }
    if (response.ok) {
      const result = await response.json();
      if (!result.error) return result;
      if (attempt === 5) throw new Error(`FEMA NFHL error: ${result.error.message}`);
    } else if (!RETRYABLE_STATUSES.has(response.status) || attempt === 5) {
      throw new Error(`FEMA NFHL request failed (${response.status} ${response.statusText})`);
    }
    await sleep(1_000 * 2 ** (attempt - 1));
  }
  throw new Error("FEMA NFHL request failed after retries");
}

export async function fetchOfficialCount(dfirmId) {
  const result = await query({
    f: "json",
    where: `DFIRM_ID='${dfirmId}'`,
    returnCountOnly: true,
  });
  return result.count;
}

export async function fetchOfficialGeoJson(
  dfirmId,
  { pageSize = PAGE_SIZE, byObjectId = false } = {},
) {
  const expectedCount = await fetchOfficialCount(dfirmId);
  const features = [];

  if (byObjectId) {
    const idResult = await query({
      f: "json",
      where: `DFIRM_ID='${dfirmId}'`,
      returnIdsOnly: true,
    });
    const objectIds = [...idResult.objectIds].sort((a, b) => a - b);
    for (let offset = 0; offset < objectIds.length; offset += pageSize) {
      const page = await query({
        f: "geojson",
        objectIds: objectIds.slice(offset, offset + pageSize).join(","),
        outFields: "*",
        returnGeometry: true,
        outSR: 4326,
      });
      features.push(...page.features);
    }
  } else {
    for (let offset = 0; offset < expectedCount; offset += pageSize) {
      const page = await query({
        f: "geojson",
        where: `DFIRM_ID='${dfirmId}'`,
        outFields: "*",
        returnGeometry: true,
        outSR: 4326,
        orderByFields: "OBJECTID",
        resultOffset: offset,
        resultRecordCount: pageSize,
      });
      features.push(...page.features);
    }
  }

  if (features.length !== expectedCount) {
    throw new Error(
      `Expected ${expectedCount} ${dfirmId} features from FEMA, received ${features.length}`,
    );
  }

  features.sort((a, b) => (a.properties.OBJECTID ?? 0) - (b.properties.OBJECTID ?? 0));
  return { type: "FeatureCollection", features };
}

export async function fetchOfficialAttributes(dfirmId) {
  const expectedCount = await fetchOfficialCount(dfirmId);
  const attributes = [];

  for (let offset = 0; offset < expectedCount; offset += PAGE_SIZE) {
    const page = await query({
      f: "json",
      where: `DFIRM_ID='${dfirmId}'`,
      outFields: "DFIRM_ID,FLD_AR_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,SOURCE_CIT",
      returnGeometry: false,
      orderByFields: "OBJECTID",
      resultOffset: offset,
      resultRecordCount: PAGE_SIZE,
    });
    attributes.push(...page.features.map((feature) => feature.attributes));
  }

  if (attributes.length !== expectedCount) {
    throw new Error(
      `Expected ${expectedCount} ${dfirmId} attributes from FEMA, received ${attributes.length}`,
    );
  }
  return attributes;
}

export async function fetchOfficialPoint(longitude, latitude) {
  const result = await query({
    f: "json",
    where: "1=1",
    geometry: JSON.stringify({
      x: longitude,
      y: latitude,
      spatialReference: { wkid: 4326 },
    }),
    geometryType: "esriGeometryPoint",
    inSR: 4326,
    spatialRel: "esriSpatialRelIntersects",
    outFields: "DFIRM_ID,FLD_AR_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,SOURCE_CIT",
    returnGeometry: false,
  });
  return result.features.map((feature) => feature.attributes);
}
