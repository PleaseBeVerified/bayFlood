import assert from "node:assert/strict";
import test from "node:test";
import {
  extractCensusBlockGroup,
  getFemaAreaClaimSignal,
  lookupFemaAreaClaimSignal,
} from "../src/lib/fema-area-signal.ts";

const rimrockCensusResponse = {
  result: {
    geographies: {
      "2020 Census Blocks": [{ GEOID: "060855119103010" }],
    },
  },
};

test("extracts the Census block group from a block GEOID", () => {
  assert.equal(extractCensusBlockGroup(rimrockCensusResponse), "060855119103");
  assert.equal(extractCensusBlockGroup({}), null);
});

test("returns Rimrock's very-low area signal without changing it into a probability", () => {
  const signal = getFemaAreaClaimSignal("060855119103");
  assert.equal(signal.status, "matched");
  if (signal.status !== "matched") return;
  assert.equal(signal.claims, 0);
  assert.equal(signal.policyYears, 31.523693);
  assert.equal(signal.percentile, 14.71);
  assert.equal(signal.level, "Very low");
  assert.match(signal.interpretation, /not a property flood probability/i);
});

test("distinguishes a valid geography with no NFIP area rate", () => {
  const signal = getFemaAreaClaimSignal("999999999999");
  assert.equal(signal.status, "no_nfip_area_rate");
  assert.match(signal.interpretation, /does not mean the property is flood-free/i);
});

test("looks up coordinates and degrades independently when Census is unavailable", async () => {
  const matched = await lookupFemaAreaClaimSignal(
    -121.8690599,
    37.2059664,
    async () => Response.json(rimrockCensusResponse),
  );
  assert.equal(matched.status, "matched");

  const unavailable = await lookupFemaAreaClaimSignal(
    -121.8690599,
    37.2059664,
    async () => {
      throw new Error("network unavailable");
    },
  );
  assert.equal(unavailable.status, "unavailable");
  assert.match(unavailable.interpretation, /physical exposure score.*do not depend/i);
});
