import assert from "node:assert/strict";
import test from "node:test";
import {
  modelRiskPresentation,
  predictFloodClaimSimilarity,
} from "../src/lib/flood-model.ts";

const rimrockInput = {
  elevation_m: 134.1,
  sfha: 0,
  distance_sfha_m: 832,
  rainfall_100yr_24h_in: 7.59,
  nearest_water_distance_m: 88,
  fema_zone: "D",
  fema_subtype: "Unknown",
  nearest_water_kind: "waterbody",
};

test("keeps the high-elevation Rimrock regression case below the claim threshold", () => {
  const prediction = predictFloodClaimSimilarity(rimrockInput);

  assert.equal(prediction.classification, "comparison-like");
  assert.ok(prediction.score <= 30, `expected <= 30, received ${prediction.score}`);
  assert.equal(modelRiskPresentation(prediction.probability).level, "Low");
});

test("increasing elevation cannot increase an otherwise identical model score", () => {
  const probabilities = [0, 15, 30, 60, 120, 250, 500].map(
    (elevation_m) =>
      predictFloodClaimSimilarity({ ...rimrockInput, elevation_m }).probability,
  );

  for (let index = 1; index < probabilities.length; index += 1) {
    assert.ok(probabilities[index] <= probabilities[index - 1]);
  }
});

test("scores an address with unavailable FEMA membership and distances", () => {
  const prediction = predictFloodClaimSimilarity({
    ...rimrockInput,
    sfha: null,
    distance_sfha_m: null,
    nearest_water_distance_m: null,
    fema_zone: "Unknown",
    fema_subtype: "Unknown",
    nearest_water_kind: "Unknown",
  });

  assert.ok(Number.isFinite(prediction.probability));
  assert.ok(prediction.probability >= 0 && prediction.probability <= 1);
});
