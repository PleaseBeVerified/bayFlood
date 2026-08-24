import assert from "node:assert/strict";
import test from "node:test";
import {
  buildClaimCandidate,
  buildPolicyCandidate,
  coordinatePrecisionMeters,
} from "../scripts/lib/openfema-candidates.mjs";

const county = { name: "Santa Clara", fips: "06085" };

test("turns a redacted FEMA claim into a pending event candidate", () => {
  const row = buildClaimCandidate({
    id: 42,
    dateOfLoss: "2023-01-09T00:00:00.000Z",
    reportedZipCode: "95125",
    latitude: 37.3,
    longitude: -121.9,
    amountPaidOnBuildingClaim: 1000,
    amountPaidOnContentsClaim: 250,
  }, county);
  assert.equal(row.event_date, "2023-01-09");
  assert.equal(row.total_claim_payment, 1250);
  assert.equal(row.verification_status, "pending");
  assert.equal(row.parcel_id, "");
  assert.match(row.notes, /cannot be treated as a verified parcel label/);
});

test("turns a policy term into pending exposure, not a non-event label", () => {
  const row = buildPolicyCandidate({
    id: 84,
    policyEffectiveDate: "2022-06-01T00:00:00.000Z",
    policyTerminationDate: "2023-06-01T00:00:00.000Z",
    reportedZipCode: "95125",
    latitude: 37.3,
    longitude: -121.9,
  }, county);
  assert.equal(row.observation_start, "2022-06-01");
  assert.equal(row.observation_end, "2023-06-01");
  assert.equal(row.event_date, "");
  assert.equal(row.verification_status, "pending");
  assert.match(row.non_event_evidence, /not evidence/);
});

test("reports kilometer-scale precision for one-decimal coordinates", () => {
  assert.ok(coordinatePrecisionMeters(37.3) > 7000);
});
