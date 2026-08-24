import { mkdir, writeFile } from "node:fs/promises";
import { requiredHeaders, serializeCsv } from "./lib/property-observations.mjs";
import {
  apiRoot,
  bayAreaCounties,
  buildClaimCandidate,
  buildPolicyCandidate,
  candidateIsMappable,
  claimsSelect,
  policiesSelect,
} from "./lib/openfema-candidates.mjs";

const outputPath = "data/openfema-observation-candidates.csv";
const reportJsonPath = "reports/openfema-candidate-audit.json";
const reportMarkdownPath = "reports/openfema-candidate-audit.md";
const pageSize = 250;
const retryableStatuses = new Set([429, 500, 502, 503, 504]);
const maxArgument = process.argv.find((argument) => argument.startsWith("--max-per-county="));
const maxPerCounty = process.argv.includes("--all")
  ? Number.POSITIVE_INFINITY
  : Number(maxArgument?.slice(17) ?? 250);

if (!Number.isInteger(maxPerCounty) && maxPerCounty !== Number.POSITIVE_INFINITY) {
  throw new Error("--max-per-county must be a positive integer.");
}
if (maxPerCounty <= 0) throw new Error("--max-per-county must be a positive integer.");

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function fetchJson(url) {
  for (let attempt = 1; attempt <= 6; attempt += 1) {
    let response;
    try {
      response = await fetch(url, {
        headers: { "user-agent": "BayFlood-public-training-candidate-generator/1.0" },
        signal: AbortSignal.timeout(120_000),
      });
    } catch (error) {
      if (attempt === 6) throw error;
      const delay = Math.min(30_000, 2000 * 2 ** (attempt - 1));
      console.warn(`  FEMA connection failed; retrying in ${delay / 1000}s (attempt ${attempt}/6).`);
      await sleep(delay);
      continue;
    }
    if (response.ok) return response.json();
    if (!retryableStatuses.has(response.status) || attempt === 6) {
      const detail = (await response.text()).slice(0, 1000);
      throw new Error(`OpenFEMA request failed: ${response.status} ${response.statusText}\n${detail}`);
    }
    const delay = Math.min(30_000, 2000 * 2 ** (attempt - 1));
    console.warn(`  FEMA returned ${response.status}; retrying in ${delay / 1000}s (attempt ${attempt}/6).`);
    await sleep(delay);
  }
  throw new Error("OpenFEMA request failed after retries.");
}

async function fetchCountyRecords({ entity, filter, select, maximum, label }) {
  const rows = [];
  while (rows.length < maximum) {
    const top = Number.isFinite(maximum)
      ? Math.min(pageSize, maximum - rows.length)
      : pageSize;
    const url = new URL(`${apiRoot}/${entity}`);
    url.searchParams.set("$filter", filter);
    url.searchParams.set("$select", select);
    url.searchParams.set("$top", String(top));
    url.searchParams.set("$skip", String(rows.length));
    console.log(`  Downloading ${label}: records ${rows.length + 1}-${rows.length + top}...`);
    const result = await fetchJson(url);
    const page = result[entity] ?? [];
    rows.push(...page);
    console.log(`  Received ${rows.length} ${label} records.`);
    if (page.length < top) break;
    await sleep(250);
  }
  return rows;
}

function auditMarkdown(report) {
  const countyRows = report.counties.map((county) =>
    `| ${county.county} | ${county.claims} | ${county.policyExposures} | ${county.skippedWithoutCoordinates} |`,
  ).join("\n");
  return `# OpenFEMA training-candidate audit\n\n` +
    `Generated: ${report.generatedAt}\n\n` +
    `These are **automated coarse-area candidates**, not verified property labels. ` +
    `OpenFEMA removes street addresses and rounds public coordinates, so the generator marks every row \`pending\`.\n\n` +
    `| County | Claim candidates | Policy-exposure candidates | Skipped without coordinates |\n` +
    `|---|---:|---:|---:|\n${countyRows}\n\n` +
    `Total candidates: ${report.totalCandidates}\n\n` +
    `Verified properties added: 0\n\n` +
    `The policy rows represent insurance exposure only. They cannot be called non-flood properties unless a separate, parcel-resolved source confirms the outcome.\n`;
}

async function main() {
  const candidates = [];
  const counties = [];

  for (const county of bayAreaCounties) {
    console.log(`Starting ${county.name} County...`);
    const claims = await fetchCountyRecords({
      entity: "NfipClaims",
      filter: `state eq 'CA' and countyCode eq '${county.fips}'`,
      select: claimsSelect,
      maximum: maxPerCounty,
      label: `${county.name} claims`,
    });
    const nextCountyFips = String(Number(county.fips) + 1).padStart(5, "0");
    const policies = await fetchCountyRecords({
      entity: "NfipPolicies",
      filter: `propertyState eq 'CA' and censusGeoid ge '${county.fips}0000000' and censusGeoid lt '${nextCountyFips}0000000'`,
      select: policiesSelect,
      maximum: maxPerCounty,
      label: `${county.name} policy exposures`,
    });
    const transformed = [
      ...claims.map((record) => buildClaimCandidate(record, county)),
      ...policies.map((record) => buildPolicyCandidate(record, county)),
    ];
    const mappable = transformed.filter(candidateIsMappable);
    candidates.push(...mappable);
    counties.push({
      county: county.name,
      claims: mappable.filter((row) => row.record_type === "claim").length,
      policyExposures: mappable.filter((row) => row.record_type === "policy_exposure").length,
      skippedWithoutCoordinates: transformed.length - mappable.length,
    });
    console.log(`${county.name}: ${claims.length} claims, ${policies.length} policy exposures.`);
  }

  candidates.sort((a, b) => String(a.property_id).localeCompare(String(b.property_id)));
  const extraHeaders = [
    "record_type", "openfema_record_id", "census_geoid", "reported_zip_code",
    "flood_zone", "policy_count", "total_claim_payment", "building_coverage",
    "contents_coverage", "as_of_date",
  ];
  const report = {
    generatedAt: new Date().toISOString(),
    apiVersion: "OpenFEMA NfipClaims v3 and NfipPolicies v3",
    maximumPerCountyPerDataset: Number.isFinite(maxPerCounty) ? maxPerCounty : "all",
    totalCandidates: candidates.length,
    verifiedPropertiesAdded: 0,
    counties,
    limitations: [
      "Street addresses and parcel identifiers are not present in public OpenFEMA records.",
      "Latitude and longitude are rounded to one decimal degree and are unsuitable for parcel assignment.",
      "Public policy and claim rows cannot be reliably joined into an exact property history.",
      "All generated rows remain pending and are excluded from hazard training.",
    ],
  };
  await mkdir("reports", { recursive: true });
  await Promise.all([
    writeFile(outputPath, serializeCsv(candidates, [...requiredHeaders, ...extraHeaders])),
    writeFile(reportJsonPath, `${JSON.stringify(report, null, 2)}\n`),
    writeFile(reportMarkdownPath, auditMarkdown(report)),
  ]);
  console.log(`Wrote ${candidates.length} pending candidates to ${outputPath}.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
