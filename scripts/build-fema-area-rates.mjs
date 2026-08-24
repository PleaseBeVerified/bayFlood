import { mkdir, readFile, stat, writeFile } from "node:fs/promises";
import { DuckDBConnection } from "@duckdb/node-api";
import { parseCsv } from "./lib/property-observations.mjs";

const claimsPath = "data/cache/openfema/NfipClaimsV3.parquet";
const policiesPath = "data/cache/openfema/NfipPoliciesV3.parquet";
const areaYearPath = "data/fema-area-year-rates.csv";
const areaPath = "data/fema-area-rates.csv";
const validationPath = "reports/fema-area-signal-validation.csv";
const reportPath = "reports/fema-area-rate-report.json";
const reportMarkdownPath = "reports/fema-area-rate-report.md";
const publicJsonPath = "public/data/fema-area-claim-signal.json";
const latestCompleteYear = new Date().getUTCFullYear() - 1;
const firstExposureYear = 2009;
const priorPolicyYears = 100;
const countyFips = ["06001", "06013", "06041", "06055", "06075", "06081", "06085", "06095", "06097"];
const countyListSql = countyFips.map((value) => `'${value}'`).join(",");

function sqlPath(path) {
  return path.replaceAll("'", "''");
}

function countyNameCase(column) {
  return `CASE ${column}
    WHEN '06001' THEN 'Alameda'
    WHEN '06013' THEN 'Contra Costa'
    WHEN '06041' THEN 'Marin'
    WHEN '06055' THEN 'Napa'
    WHEN '06075' THEN 'San Francisco'
    WHEN '06081' THEN 'San Mateo'
    WHEN '06085' THEN 'Santa Clara'
    WHEN '06095' THEN 'Solano'
    WHEN '06097' THEN 'Sonoma'
    ELSE 'Unknown' END`;
}

async function ensureCache(path) {
  const file = await stat(path).catch(() => null);
  if (!file || file.size < 1024) {
    throw new Error(`Missing complete OpenFEMA cache file: ${path}. Run npm run training:area-cache first.`);
  }
}

async function scalarRows(connection, sql) {
  const reader = await connection.runAndReadAll(sql);
  return reader.getRowObjectsJson();
}

async function main() {
  await Promise.all([ensureCache(claimsPath), ensureCache(policiesPath)]);
  await Promise.all([mkdir("reports", { recursive: true }), mkdir("public/data", { recursive: true })]);
  const connection = await DuckDBConnection.create();
  console.log("Aggregating complete Bay Area policy exposure...");
  await connection.run(`
    CREATE OR REPLACE TEMP TABLE policy_exposure AS
    WITH filtered AS (
      SELECT
        substr(censusGeoid, 1, 12) AS census_geoid,
        substr(censusGeoid, 1, 5) AS county_fips,
        greatest(policyEffectiveDate, DATE '${firstExposureYear}-01-01') AS start_date,
        least(policyTerminationDate, DATE '${latestCompleteYear + 1}-01-01') AS end_date,
        coalesce(policyCount, 1)::DOUBLE AS policy_count
      FROM read_parquet('${sqlPath(policiesPath)}')
      WHERE substr(censusGeoid, 1, 5) IN (${countyListSql})
        AND length(censusGeoid) >= 12
        AND policyEffectiveDate IS NOT NULL
        AND policyTerminationDate IS NOT NULL
        AND policyTerminationDate > policyEffectiveDate
        AND policyEffectiveDate < DATE '${latestCompleteYear + 1}-01-01'
        AND policyTerminationDate > DATE '${firstExposureYear}-01-01'
    ), expanded AS (
      SELECT filtered.*, years.year
      FROM filtered,
      LATERAL range(
        year(start_date)::INTEGER,
        year(end_date - INTERVAL '1 day')::INTEGER + 1
      ) AS years(year)
      WHERE end_date > start_date
    )
    SELECT
      census_geoid,
      county_fips,
      year,
      round(sum(
        policy_count *
        date_diff('day', greatest(start_date, make_date(year, 1, 1)), least(end_date, make_date(year + 1, 1, 1)))::DOUBLE /
        date_diff('day', make_date(year, 1, 1), make_date(year + 1, 1, 1))
      ), 6) AS policy_years
    FROM expanded
    GROUP BY census_geoid, county_fips, year;
  `);

  console.log("Aggregating complete Bay Area claims...");
  await connection.run(`
    CREATE OR REPLACE TEMP TABLE claim_events AS
    SELECT
      substr(censusGeoid, 1, 12) AS census_geoid,
      substr(censusGeoid, 1, 5) AS county_fips,
      yearOfLoss::INTEGER AS year,
      sum(coalesce(policyCount, 1))::BIGINT AS claim_count,
      count(*)::BIGINT AS claim_records,
      round(sum(
        coalesce(amountPaidOnBuildingClaim, 0) +
        coalesce(amountPaidOnContentsClaim, 0) +
        coalesce(amountPaidOnIncreasedCostOfComplianceClaim, 0)
      ), 2) AS total_claim_payment
    FROM read_parquet('${sqlPath(claimsPath)}')
    WHERE substr(censusGeoid, 1, 5) IN (${countyListSql})
      AND length(censusGeoid) >= 12
      AND yearOfLoss BETWEEN ${firstExposureYear} AND ${latestCompleteYear}
    GROUP BY census_geoid, county_fips, year;
  `);

  console.log("Building leakage-safe area-year signals and validation splits...");
  await connection.run(`
    CREATE OR REPLACE TEMP TABLE area_year_final AS
    WITH keys AS (
      SELECT census_geoid, county_fips, year FROM policy_exposure
      UNION
      SELECT census_geoid, county_fips, year FROM claim_events
    ), raw AS (
      SELECT
        keys.census_geoid,
        keys.county_fips,
        ${countyNameCase("keys.county_fips")} AS county,
        keys.year,
        coalesce(claim_events.claim_count, 0)::BIGINT AS claim_count,
        coalesce(claim_events.claim_records, 0)::BIGINT AS claim_records,
        coalesce(claim_events.total_claim_payment, 0)::DOUBLE AS total_claim_payment,
        coalesce(policy_exposure.policy_years, 0)::DOUBLE AS policy_years
      FROM keys
      LEFT JOIN policy_exposure USING (census_geoid, county_fips, year)
      LEFT JOIN claim_events USING (census_geoid, county_fips, year)
    ), area_history AS (
      SELECT *,
        sum(claim_count) OVER (
          PARTITION BY census_geoid ORDER BY year
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )::DOUBLE AS prior_area_claims,
        sum(policy_years) OVER (
          PARTITION BY census_geoid ORDER BY year
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        )::DOUBLE AS prior_area_policy_years
      FROM raw
    ), county_year AS (
      SELECT county_fips, year, sum(claim_count)::DOUBLE AS claims, sum(policy_years)::DOUBLE AS exposure
      FROM raw GROUP BY county_fips, year
    ), county_history AS (
      SELECT *,
        sum(claims) OVER (
          PARTITION BY county_fips ORDER BY year
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_county_claims,
        sum(exposure) OVER (
          PARTITION BY county_fips ORDER BY year
          ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
        ) AS prior_county_policy_years
      FROM county_year
    ), global_year AS (
      SELECT year, sum(claim_count)::DOUBLE AS claims, sum(policy_years)::DOUBLE AS exposure
      FROM raw GROUP BY year
    ), global_history AS (
      SELECT *,
        sum(claims) OVER (ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_global_claims,
        sum(exposure) OVER (ORDER BY year ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS prior_global_policy_years
      FROM global_year
    ), joined AS (
      SELECT
        area_history.*,
        county_history.prior_county_claims,
        county_history.prior_county_policy_years,
        global_history.prior_global_claims,
        global_history.prior_global_policy_years,
        coalesce(
          county_history.prior_county_claims / nullif(county_history.prior_county_policy_years, 0),
          global_history.prior_global_claims / nullif(global_history.prior_global_policy_years, 0)
        ) AS prior_reference_rate
      FROM area_history
      LEFT JOIN county_history USING (county_fips, year)
      LEFT JOIN global_history USING (year)
    )
    SELECT
      census_geoid,
      county_fips,
      county,
      year,
      claim_count,
      claim_records,
      total_claim_payment,
      round(policy_years, 6) AS policy_years,
      round(1000 * claim_count / nullif(policy_years, 0), 6) AS observed_claims_per_1000_policy_years,
      coalesce(prior_area_claims, 0) AS prior_area_claims,
      round(coalesce(prior_area_policy_years, 0), 6) AS prior_area_policy_years,
      round(1000 * (
        coalesce(prior_area_claims, 0) + ${priorPolicyYears} * prior_reference_rate
      ) / nullif(coalesce(prior_area_policy_years, 0) + ${priorPolicyYears}, 0), 6)
        AS lagged_smoothed_claims_per_1000_policy_years,
      CASE
        WHEN year <= 2019 THEN 'train'
        WHEN year <= 2022 THEN 'validation'
        ELSE 'test'
      END AS time_split,
      county_fips AS spatial_fold,
      'OpenFEMA NFIP claims and policies v3' AS source
    FROM joined
    ORDER BY census_geoid, year;
  `);

  await connection.run(`
    CREATE OR REPLACE TEMP TABLE area_summary AS
    WITH totals AS (
      SELECT
        census_geoid,
        county_fips,
        county,
        min(year)::INTEGER AS first_year,
        max(year)::INTEGER AS last_year,
        sum(claim_count)::BIGINT AS claim_count,
        round(sum(policy_years), 6) AS policy_years,
        round(sum(total_claim_payment), 2) AS total_claim_payment
      FROM area_year_final
      GROUP BY census_geoid, county_fips, county
    ), county_totals AS (
      SELECT county_fips, sum(claim_count)::DOUBLE AS claims, sum(policy_years)::DOUBLE AS exposure
      FROM totals GROUP BY county_fips
    ), rates AS (
      SELECT
        totals.*,
        round(1000 * totals.claim_count / nullif(totals.policy_years, 0), 6) AS observed_claims_per_1000_policy_years,
        round(1000 * (
          totals.claim_count + ${priorPolicyYears} * county_totals.claims / nullif(county_totals.exposure, 0)
        ) / nullif(totals.policy_years + ${priorPolicyYears}, 0), 6)
          AS smoothed_claims_per_1000_policy_years
      FROM totals
      LEFT JOIN county_totals USING (county_fips)
    ), ranked AS (
      SELECT *, percent_rank() OVER (ORDER BY smoothed_claims_per_1000_policy_years) AS signal_percentile
      FROM rates
    )
    SELECT *,
      round(100 * signal_percentile, 2) AS signal_percentile_0_100,
      CASE
        WHEN signal_percentile < 0.2 THEN 'Very low'
        WHEN signal_percentile < 0.4 THEN 'Low'
        WHEN signal_percentile < 0.6 THEN 'Moderate'
        WHEN signal_percentile < 0.8 THEN 'High'
        ELSE 'Very high'
      END AS historical_claim_signal,
      ${latestCompleteYear}::INTEGER AS as_of_year,
      ${priorPolicyYears}::INTEGER AS smoothing_prior_policy_years,
      'Area-level historical NFIP claim signal; not a property flood probability' AS interpretation
    FROM ranked
    ORDER BY census_geoid;
  `);

  await connection.run(`
    CREATE OR REPLACE TEMP TABLE validation_summary AS
    SELECT
      time_split,
      county_fips,
      county,
      count(*)::BIGINT AS area_years,
      round(sum(policy_years), 3) AS policy_years,
      sum(claim_count)::BIGINT AS observed_claims,
      round(sum(policy_years * lagged_smoothed_claims_per_1000_policy_years / 1000), 3) AS expected_claims,
      round(
        sum(claim_count) /
        nullif(sum(policy_years * lagged_smoothed_claims_per_1000_policy_years / 1000), 0),
        4
      ) AS observed_to_expected_ratio
    FROM area_year_final
    WHERE policy_years > 0 AND lagged_smoothed_claims_per_1000_policy_years IS NOT NULL
    GROUP BY time_split, county_fips, county
    ORDER BY time_split, county_fips;
  `);

  await Promise.all([
    connection.run(`COPY area_year_final TO '${sqlPath(areaYearPath)}' (HEADER, DELIMITER ',')`),
    connection.run(`COPY area_summary TO '${sqlPath(areaPath)}' (HEADER, DELIMITER ',')`),
    connection.run(`COPY validation_summary TO '${sqlPath(validationPath)}' (HEADER, DELIMITER ',')`),
  ]);

  const [counts] = await scalarRows(connection, `
    SELECT
      (SELECT sum(claim_count)::BIGINT FROM claim_events) AS claims,
      (SELECT round(sum(policy_years), 2) FROM policy_exposure) AS policy_years,
      (SELECT count(*)::BIGINT FROM area_summary) AS census_block_groups,
      (SELECT count(*)::BIGINT FROM area_year_final) AS area_years;
  `);
  const validation = await scalarRows(connection, `
    SELECT
      time_split,
      sum(area_years)::BIGINT AS area_years,
      round(sum(policy_years), 2) AS policy_years,
      sum(observed_claims)::BIGINT AS observed_claims,
      round(sum(expected_claims), 2) AS expected_claims,
      round(sum(observed_claims) / nullif(sum(expected_claims), 0), 4) AS observed_to_expected_ratio
    FROM validation_summary GROUP BY time_split ORDER BY time_split;
  `);
  connection.closeSync();

  const areaRows = parseCsv(await readFile(areaPath, "utf8")).rows;
  const publicSignals = Object.fromEntries(areaRows.map((row) => [row.census_geoid, {
    county: row.county,
    claims: Number(row.claim_count),
    policyYears: Number(row.policy_years),
    claimsPer1000PolicyYears: Number(row.smoothed_claims_per_1000_policy_years),
    percentile: Number(row.signal_percentile_0_100),
    level: row.historical_claim_signal,
    asOfYear: Number(row.as_of_year),
  }]));
  await writeFile(publicJsonPath, `${JSON.stringify({
    interpretation: "Area-level historical NFIP claim signal; not a property flood probability",
    source: "OpenFEMA NFIP claims and policies v3",
    areas: publicSignals,
  })}\n`);

  const report = {
    generatedAt: new Date().toISOString(),
    status: "AREA_SIGNAL_READY__PARCEL_PROBABILITY_BLOCKED",
    coverage: {
      firstExposureYear,
      latestCompleteYear,
      claims: Number(counts.claims),
      policyYears: Number(counts.policy_years),
      censusBlockGroups: Number(counts.census_block_groups),
      areaYears: Number(counts.area_years),
    },
    smoothing: {
      method: "Lagged credibility shrinkage toward the prior county claim rate",
      priorPolicyYears,
      leakageControl: "Each area-year signal uses only earlier years.",
    },
    validation,
    limitations: [
      "NFIP policyholders are not a random sample of all properties.",
      "Claims are insurance events, not all physical flooding events.",
      "Public records are redacted and cannot identify exact parcels.",
      "The rate is an area-level historical insurance signal, not a 100-year property probability.",
    ],
  };
  const validationLines = validation.map((row) =>
    `| ${row.time_split} | ${row.area_years} | ${row.policy_years} | ${row.observed_claims} | ${row.expected_claims} | ${row.observed_to_expected_ratio} |`,
  ).join("\n");
  const markdown = `# FEMA area-rate pipeline\n\nStatus: **${report.status}**\n\n` +
    `Complete Bay Area v3 records were aggregated for ${firstExposureYear}-${latestCompleteYear}.\n\n` +
    `- Claims: ${counts.claims}\n- Policy-years: ${counts.policy_years}\n` +
    `- Census block groups: ${counts.census_block_groups}\n- Area-years: ${counts.area_years}\n\n` +
    `| Split | Area-years | Policy-years | Observed claims | Expected claims | O/E |\n` +
    `|---|---:|---:|---:|---:|---:|\n${validationLines}\n\n` +
    `The lagged signal uses only prior years. It is suitable as an area-level model feature, not as a property flood probability.\n`;
  await Promise.all([
    writeFile(reportPath, `${JSON.stringify(report, null, 2)}\n`),
    writeFile(reportMarkdownPath, markdown),
  ]);
  console.log(`Wrote ${counts.area_years} area-years and ${counts.census_block_groups} area summaries.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
