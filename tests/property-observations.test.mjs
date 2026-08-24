import assert from "node:assert/strict";
import test from "node:test";
import {
  expandHazardPeriods,
  requiredHeaders,
  serializeCsv,
  validateObservationTable,
} from "../scripts/lib/property-observations.mjs";

function csv(row) {
  return serializeCsv([row], requiredHeaders);
}

const base = {
  property_id: "parcel-1", parcel_id: "APN-1", building_id: "",
  address: "1 Test Street, San Jose, CA", lat: 37.3, lng: -121.9,
  county: "Santa Clara", observation_start: "2020-07-01",
  observation_end: "2022-12-31", event_date: "2022-02-03",
  event_source: "NFIP date of loss", non_event_evidence: "",
  verification_status: "verified", coordinate_precision_m: 5,
  source_url: "https://example.test/evidence", notes: "",
};

test("accepts a verified property event observation", () => {
  const result = validateObservationTable(csv(base));
  assert.deepEqual(result.errors, []);
  assert.equal(result.verifiedRows.length, 1);
});

test("keeps pending candidates out of the verified training set", () => {
  const result = validateObservationTable(csv({ ...base,
    observation_start: "", observation_end: "", event_date: "",
    event_source: "", verification_status: "pending",
  }));
  assert.deepEqual(result.errors, []);
  assert.equal(result.verifiedRows.length, 0);
  assert.equal(result.warnings.length, 1);
});

test("rejects a verified non-event without evidence", () => {
  const result = validateObservationTable(csv({ ...base, event_date: "", event_source: "" }));
  assert.match(result.errors.join("\n"), /non_event_evidence/);
});

test("expands partial and leap-year exposure and stops at the event", () => {
  const periods = expandHazardPeriods(base);
  assert.equal(periods.length, 3);
  assert.equal(periods[0].calendar_year, 2020);
  assert.equal(periods[0].exposure_days, 184);
  assert.equal(periods[0].exposure_fraction, 0.502732);
  assert.equal(periods[1].event, 0);
  assert.equal(periods[2].event, 1);
  assert.equal(periods[2].period_end, "2022-02-03");
});
