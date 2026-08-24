export const bayAreaCounties = [
  { name: "Alameda", fips: "06001" },
  { name: "Contra Costa", fips: "06013" },
  { name: "Marin", fips: "06041" },
  { name: "Napa", fips: "06055" },
  { name: "San Francisco", fips: "06075" },
  { name: "San Mateo", fips: "06081" },
  { name: "Santa Clara", fips: "06085" },
  { name: "Solano", fips: "06095" },
  { name: "Sonoma", fips: "06097" },
];

const apiRoot = "https://www.fema.gov/api/open/v3";

function dateOnly(value) {
  return value ? String(value).slice(0, 10) : "";
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function recordUrl(entity, id) {
  const filter = encodeURIComponent(`id eq ${id}`);
  return `${apiRoot}/${entity}?$filter=${filter}`;
}

export function coordinatePrecisionMeters(latitude) {
  const numericLatitude = finiteNumber(latitude);
  if (numericLatitude === null) return "";
  const northSouthHalfStep = 5_566;
  const eastWestHalfStep = northSouthHalfStep * Math.cos(numericLatitude * Math.PI / 180);
  return Math.round(Math.hypot(northSouthHalfStep, eastWestHalfStep));
}

function commonCandidate(record, county, entity, recordType) {
  const latitude = finiteNumber(record.latitude);
  const longitude = finiteNumber(record.longitude);
  return {
    property_id: `openfema-${recordType}-${record.id}`,
    parcel_id: "",
    building_id: "",
    address: `REDACTED NFIP location near ZIP ${record.reportedZipCode || "unknown"}`,
    lat: latitude ?? "",
    lng: longitude ?? "",
    county: county.name,
    event_source: "",
    non_event_evidence: "",
    verification_status: "pending",
    coordinate_precision_m: coordinatePrecisionMeters(latitude),
    source_url: recordUrl(entity, record.id),
    notes: "OpenFEMA removes street addresses and rounds coordinates. This record is an automated coarse-area candidate and cannot be treated as a verified parcel label.",
    record_type: recordType,
    openfema_record_id: record.id,
    census_geoid: record.censusGeoid ?? "",
    reported_zip_code: record.reportedZipCode ?? "",
    flood_zone: record.floodZoneCurrent ?? record.ratedFloodZone ?? "",
    policy_count: record.policyCount ?? "",
    as_of_date: dateOnly(record.asOfDate),
  };
}

export function buildClaimCandidate(record, county) {
  const row = commonCandidate(record, county, "NfipClaims", "claim");
  const eventDate = dateOnly(record.dateOfLoss);
  return {
    ...row,
    observation_start: "",
    observation_end: "",
    event_date: eventDate,
    event_source: eventDate
      ? `OpenFEMA NFIP claim v3 record ${record.id}; date of loss ${eventDate}`
      : `OpenFEMA NFIP claim v3 record ${record.id}`,
    total_claim_payment: Math.round((
      (finiteNumber(record.amountPaidOnBuildingClaim) ?? 0) +
      (finiteNumber(record.amountPaidOnContentsClaim) ?? 0) +
      (finiteNumber(record.amountPaidOnIncreasedCostOfComplianceClaim) ?? 0)
    ) * 100) / 100,
    building_coverage: record.totalBuildingInsuranceCoverage ?? "",
    contents_coverage: record.totalContentsInsuranceCoverage ?? "",
  };
}

export function buildPolicyCandidate(record, county) {
  const row = commonCandidate(record, county, "NfipPolicies", "policy_exposure");
  return {
    ...row,
    observation_start: dateOnly(record.policyEffectiveDate),
    observation_end: dateOnly(record.policyTerminationDate),
    event_date: "",
    non_event_evidence: "NFIP policy-term exposure only. The public record cannot be linked reliably to a claim or parcel, so it is not evidence that the property stayed flood-free.",
    total_claim_payment: "",
    building_coverage: record.totalBuildingInsuranceCoverage ?? "",
    contents_coverage: record.totalContentsInsuranceCoverage ?? "",
  };
}

export function candidateIsMappable(row) {
  return Number.isFinite(Number(row.lat)) && Number.isFinite(Number(row.lng));
}

export const claimsSelect = [
  "id", "asOfDate", "dateOfLoss", "state", "countyCode", "censusGeoid",
  "reportedZipCode", "latitude", "longitude", "floodZoneCurrent", "ratedFloodZone",
  "policyCount", "amountPaidOnBuildingClaim", "amountPaidOnContentsClaim",
  "amountPaidOnIncreasedCostOfComplianceClaim", "totalBuildingInsuranceCoverage",
  "totalContentsInsuranceCoverage",
].join(",");

export const policiesSelect = [
  "id", "asOfDate", "policyEffectiveDate", "policyTerminationDate", "propertyState",
  "censusGeoid", "reportedZipCode", "latitude", "longitude",
  "floodZoneCurrent", "ratedFloodZone", "policyCount", "totalBuildingInsuranceCoverage",
  "totalContentsInsuranceCoverage",
].join(",");

export { apiRoot };
