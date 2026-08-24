import areaSignalArtifact from "../../public/data/fema-area-claim-signal.json" with {
  type: "json",
};

type AreaSignalRecord = {
  county: string;
  claims: number;
  policyYears: number;
  claimsPer1000PolicyYears: number;
  percentile: number;
  level: "Very low" | "Low" | "Moderate" | "High" | "Very high";
  asOfYear: number;
};

type AreaSignalArtifact = {
  interpretation: string;
  source: string;
  areas: Record<string, AreaSignalRecord>;
};

export type FemaAreaClaimSignal =
  | {
      status: "matched";
      censusBlockGroup: string;
      county: string;
      claims: number;
      policyYears: number;
      claimsPer1000PolicyYears: number;
      percentile: number;
      level: AreaSignalRecord["level"];
      asOfYear: number;
      source: string;
      interpretation: string;
    }
  | {
      status: "no_nfip_area_rate";
      censusBlockGroup: string;
      source: string;
      interpretation: string;
    }
  | {
      status: "unavailable";
      censusBlockGroup: null;
      source: string;
      interpretation: string;
    };

const artifact = areaSignalArtifact as AreaSignalArtifact;
const censusCoordinateEndpoint =
  "https://geocoding.geo.census.gov/geocoder/geographies/coordinates";

export function extractCensusBlockGroup(payload: unknown) {
  const result = payload as {
    result?: {
      geographies?: Record<string, Array<{ GEOID?: string }>>;
    };
  };
  const block = result.result?.geographies?.["2020 Census Blocks"]?.[0];
  const geoid = String(block?.GEOID ?? "");
  return geoid.length >= 12 ? geoid.slice(0, 12) : null;
}

export function getFemaAreaClaimSignal(
  censusBlockGroup: string,
): FemaAreaClaimSignal {
  const signal = artifact.areas[censusBlockGroup];
  if (!signal) {
    return {
      status: "no_nfip_area_rate",
      censusBlockGroup,
      source: artifact.source,
      interpretation:
        "Census geography was found, but the public NFIP data has no usable policy-rate denominator for this block group. This does not mean the property is flood-free.",
    };
  }

  return {
    status: "matched",
    censusBlockGroup,
    ...signal,
    source: artifact.source,
    interpretation: artifact.interpretation,
  };
}

export async function lookupFemaAreaClaimSignal(
  longitude: number,
  latitude: number,
  fetcher: typeof fetch = fetch,
): Promise<FemaAreaClaimSignal> {
  const url = new URL(censusCoordinateEndpoint);
  url.searchParams.set("x", String(longitude));
  url.searchParams.set("y", String(latitude));
  url.searchParams.set("benchmark", "Public_AR_Current");
  url.searchParams.set("vintage", "Current_Current");
  url.searchParams.set("format", "json");

  try {
    const response = await fetcher(url, {
      headers: { "user-agent": "BayFlood-area-signal/1.0" },
      signal: AbortSignal.timeout(10_000),
    });
    if (!response.ok) {
      throw new Error(`Census geography lookup returned ${response.status}.`);
    }
    const censusBlockGroup = extractCensusBlockGroup(await response.json());
    if (!censusBlockGroup) {
      throw new Error("Census geography response did not contain a block group.");
    }
    return getFemaAreaClaimSignal(censusBlockGroup);
  } catch {
    return {
      status: "unavailable",
      censusBlockGroup: null,
      source: artifact.source,
      interpretation:
        "The neighborhood claim-history lookup is temporarily unavailable. The physical exposure score and FEMA map result do not depend on this lookup.",
    };
  }
}
