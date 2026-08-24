import { mkdir, readFile, writeFile } from "node:fs/promises";
import { coordinateKey, extractCensusGeography } from "./lib/census-area.mjs";
import { parseCsv, serializeCsv } from "./lib/property-observations.mjs";

const inputPath = process.argv.find((argument) => argument.startsWith("--input="))?.slice(8)
  ?? "data/firststreet-property-candidates.csv";
const outputPath = process.argv.find((argument) => argument.startsWith("--output="))?.slice(9)
  ?? "data/property-area-features.csv";
const ratesPath = "data/fema-area-rates.csv";
const cachePath = "data/cache/census-coordinate-geographies.json";
const censusSource = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates";

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function lookupCensusGeography(latitude, longitude) {
  const url = new URL(censusSource);
  url.searchParams.set("x", longitude);
  url.searchParams.set("y", latitude);
  url.searchParams.set("benchmark", "Public_AR_Current");
  url.searchParams.set("vintage", "Current_Current");
  url.searchParams.set("format", "json");
  for (let attempt = 1; attempt <= 5; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: { "user-agent": "BayFlood-Census-geography-join/1.0" },
        signal: AbortSignal.timeout(45_000),
      });
      if (response.ok) return extractCensusGeography(await response.json());
      if (attempt === 5 || (response.status < 500 && response.status !== 429)) {
        throw new Error(`Census lookup failed: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      if (attempt === 5) throw error;
    }
    await sleep(Math.min(10_000, 1000 * 2 ** (attempt - 1)));
  }
  return null;
}

async function main() {
  const input = parseCsv(await readFile(inputPath, "utf8"));
  const rateRows = parseCsv(await readFile(ratesPath, "utf8")).rows;
  const rates = new Map(rateRows.map((row) => [row.census_geoid, row]));
  const cache = JSON.parse(await readFile(cachePath, "utf8").catch(() => "{}"));
  const pending = input.rows.filter((row) => !cache[coordinateKey(row.lat, row.lng)]);

  for (let offset = 0; offset < pending.length; offset += 5) {
    const batch = pending.slice(offset, offset + 5);
    const results = await Promise.all(batch.map(async (row) => ({
      key: coordinateKey(row.lat, row.lng),
      geography: await lookupCensusGeography(row.lat, row.lng),
    })));
    for (const result of results) cache[result.key] = result.geography;
    await mkdir("data/cache", { recursive: true });
    await writeFile(cachePath, `${JSON.stringify(cache, null, 2)}\n`);
    console.log(`Resolved ${Math.min(offset + batch.length, pending.length)}/${pending.length} uncached coordinates.`);
    await sleep(250);
  }

  const joined = input.rows.map((row) => {
    const geography = cache[coordinateKey(row.lat, row.lng)] ?? null;
    const rate = geography ? rates.get(geography.census_geoid) : null;
    return {
      ...row,
      census_block_geoid: geography?.census_block_geoid ?? "",
      census_geoid: geography?.census_geoid ?? "",
      census_tract_geoid: geography?.census_tract_geoid ?? "",
      census_county_fips: geography?.county_fips ?? "",
      area_claim_count: rate?.claim_count ?? "",
      area_policy_years: rate?.policy_years ?? "",
      area_smoothed_claims_per_1000_policy_years:
        rate?.smoothed_claims_per_1000_policy_years ?? "",
      area_signal_percentile_0_100: rate?.signal_percentile_0_100 ?? "",
      area_historical_claim_signal: rate?.historical_claim_signal ?? "",
      area_signal_as_of_year: rate?.as_of_year ?? "",
      area_signal_interpretation: rate?.interpretation
        ?? "Area signal unavailable; not a property flood probability",
      area_match_status: rate ? "matched" : geography ? "no_nfip_area_rate" : "census_lookup_failed",
      census_source_url: censusSource,
    };
  });
  const extraHeaders = [
    "census_block_geoid", "census_geoid", "census_tract_geoid", "census_county_fips", "area_claim_count",
    "area_policy_years", "area_smoothed_claims_per_1000_policy_years",
    "area_signal_percentile_0_100", "area_historical_claim_signal",
    "area_signal_as_of_year", "area_signal_interpretation", "area_match_status",
    "census_source_url",
  ];
  const headers = [...input.headers, ...extraHeaders.filter((header) => !input.headers.includes(header))];
  await writeFile(outputPath, serializeCsv(joined, headers));
  console.log(`Wrote ${joined.length} property-area joins to ${outputPath}.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
