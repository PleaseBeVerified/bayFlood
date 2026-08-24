import { mkdir, readFile, writeFile } from "node:fs/promises";
import { parseCsv } from "./lib/property-observations.mjs";

const featuresPath = "data/property-observation-features.csv";
const hazardPath = "data/property-hazard-table.csv";
const areaReportPath = "reports/fema-area-rate-report.json";
const reportDirectory = "reports";

const thresholds = {
  verifiedProperties: 1000,
  eventProperties: 100,
  propertyYears: 5000,
  counties: 5,
};

async function main() {
  const features = parseCsv(await readFile(featuresPath, "utf8")).rows;
  const periods = parseCsv(await readFile(hazardPath, "utf8")).rows;
  const areaReport = JSON.parse(await readFile(areaReportPath, "utf8").catch(() => "null"));
  const events = features.filter((row) => row.event_date);
  const censored = features.filter((row) => !row.event_date);
  const counties = new Set(features.map((row) => row.county).filter(Boolean));
  const checks = {
    verifiedProperties: features.length >= thresholds.verifiedProperties,
    eventProperties: events.length >= thresholds.eventProperties,
    censoredProperties: censored.length > 0,
    propertyYears: periods.length >= thresholds.propertyYears,
    counties: counties.size >= thresholds.counties,
  };
  const report = {
    generatedAt: new Date().toISOString(),
    status: Object.values(checks).every(Boolean) ? "READY_FOR_CALIBRATION" : "NOT_READY",
    counts: {
      verifiedProperties: features.length,
      eventProperties: events.length,
      censoredProperties: censored.length,
      propertyYears: periods.length,
      counties: counties.size,
    },
    thresholds,
    checks,
    areaSignal: areaReport ? {
      status: areaReport.status,
      claims: areaReport.coverage.claims,
      policyYears: areaReport.coverage.policyYears,
      censusBlockGroups: areaReport.coverage.censusBlockGroups,
      firstExposureYear: areaReport.coverage.firstExposureYear,
      latestCompleteYear: areaReport.coverage.latestCompleteYear,
      permittedUse: "Area-level historical NFIP claim-rate feature only",
      propertyProbabilityGenerated: false,
      hundredYearProbabilityGenerated: false,
    } : null,
    caveat: "Passing these minimums permits model calibration work; it does not establish accuracy or regulatory suitability.",
  };
  const markdown = `# Property hazard calibration readiness\n\n` +
    `Status: **${report.status}**\n\n` +
    `| Check | Current | Minimum | Pass |\n|---|---:|---:|:---:|\n` +
    `| Verified properties | ${features.length} | ${thresholds.verifiedProperties} | ${checks.verifiedProperties ? "Yes" : "No"} |\n` +
    `| Event properties | ${events.length} | ${thresholds.eventProperties} | ${checks.eventProperties ? "Yes" : "No"} |\n` +
    `| Censored properties | ${censored.length} | 1 | ${checks.censoredProperties ? "Yes" : "No"} |\n` +
    `| Property-years | ${periods.length} | ${thresholds.propertyYears} | ${checks.propertyYears ? "Yes" : "No"} |\n` +
    `| Counties | ${counties.size} | ${thresholds.counties} | ${checks.counties ? "Yes" : "No"} |\n\n` +
    `${report.caveat}\n\n` +
    `## FEMA area-level signal\n\n` +
    (areaReport
      ? `Status: **${areaReport.status}**\n\n` +
        `The complete public-data aggregation covers ${areaReport.coverage.claims} claims, ` +
        `${areaReport.coverage.policyYears} policy-years, and ${areaReport.coverage.censusBlockGroups} ` +
        `census block groups for ${areaReport.coverage.firstExposureYear}-${areaReport.coverage.latestCompleteYear}. ` +
        `It may be joined to an address as a historical area feature. It is not a property probability, and no ` +
        `100-year property probability was generated.\n`
      : `No FEMA area-rate report is available. Run \`npm run training:area-pipeline\`.\n`);
  await mkdir(reportDirectory, { recursive: true });
  await Promise.all([
    writeFile(`${reportDirectory}/property-hazard-readiness.json`, `${JSON.stringify(report, null, 2)}\n`),
    writeFile(`${reportDirectory}/property-hazard-readiness.md`, markdown),
  ]);
  console.log(`${report.status}: ${features.length} properties, ${periods.length} property-years.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
