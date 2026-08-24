import assert from "node:assert/strict";
import test from "node:test";
import parityData from "./fixtures/physical-boosted-parity.json" with {
  type: "json",
};
import { estimatePhysicalHundredYearFloodChance } from "../src/lib/physical-flood-estimate.ts";

const rimrockInput = {
  elevation_m: 134.1,
  rainfall_100yr_24h_in: 7.59,
  nearest_water_distance_m: 88,
  nearest_water_kind: "waterbody",
  slope_degrees: 7.203,
  local_elevation_250m: 13.469,
  local_elevation_500m: 5.629,
  flow_accumulation_m2: 200,
  height_above_drainage_m: 26.427,
};

test("estimates an experimental 100-year chance without FEMA address data", () => {
  const estimate = estimatePhysicalHundredYearFloodChance(rimrockInput);

  assert.equal(estimate.status, "experimental_physical_estimate");
  assert.equal(estimate.addressFemaRecordRequired, false);
  assert.ok(estimate.percent >= 0 && estimate.percent <= 100);
  assert.ok(estimate.percent < 25);
  assert.ok(estimate.range.lowerPercent <= estimate.percent);
  assert.ok(estimate.range.upperPercent >= estimate.percent);
});

test("physical constraints prevent safer terrain changes from increasing the estimate", () => {
  const elevations = [0, 15, 30, 60, 120, 250].map((elevation_m) =>
    estimatePhysicalHundredYearFloodChance({
      ...rimrockInput,
      elevation_m,
    }).percent,
  );
  const waterDistances = [0, 50, 200, 1_000, 5_000].map(
    (nearest_water_distance_m) =>
      estimatePhysicalHundredYearFloodChance({
        ...rimrockInput,
        nearest_water_distance_m,
      }).percent,
  );

  for (let index = 1; index < elevations.length; index += 1) {
    assert.ok(elevations[index] <= elevations[index - 1]);
  }
  for (let index = 1; index < waterDistances.length; index += 1) {
    assert.ok(waterDistances[index] <= waterDistances[index - 1]);
  }
});

test("increasing rainfall cannot reduce the physical estimate", () => {
  const estimates = [3, 5, 7, 9, 12].map(
    (rainfall_100yr_24h_in) =>
      estimatePhysicalHundredYearFloodChance({
        ...rimrockInput,
        rainfall_100yr_24h_in,
      }).percent,
  );

  for (let index = 1; index < estimates.length; index += 1) {
    assert.ok(estimates[index] >= estimates[index - 1]);
  }
});

test("safer local terrain and HAND cannot increase the estimate", () => {
  const localElevations = [-30, -10, 0, 10, 30].map(
    (local_elevation_250m) =>
      estimatePhysicalHundredYearFloodChance({
        ...rimrockInput,
        local_elevation_250m,
      }).percent,
  );
  const heightsAboveDrainage = [0, 2, 5, 10, 30].map(
    (height_above_drainage_m) =>
      estimatePhysicalHundredYearFloodChance({
        ...rimrockInput,
        height_above_drainage_m,
      }).percent,
  );

  for (const estimates of [localElevations, heightsAboveDrainage]) {
    for (let index = 1; index < estimates.length; index += 1) {
      assert.ok(estimates[index] <= estimates[index - 1]);
    }
  }
});

test("more modeled upslope flow cannot reduce the estimate", () => {
  const estimates = [100, 1_000, 10_000, 100_000, 1_000_000].map(
    (flow_accumulation_m2) =>
      estimatePhysicalHundredYearFloodChance({
        ...rimrockInput,
        flow_accumulation_m2,
      }).percent,
  );

  for (let index = 1; index < estimates.length; index += 1) {
    assert.ok(estimates[index] >= estimates[index - 1]);
  }
});

test("TypeScript tree evaluation matches the trained XGBoost models", () => {
  for (const fixture of parityData.fixtures) {
    const estimate = estimatePhysicalHundredYearFloodChance(fixture.input);
    for (const className of ["onePercent", "pointTwoPercent", "minimal"]) {
      assert.ok(
        Math.abs(
          estimate.classProbabilities[className] -
            fixture.expectedClassProbabilities[className],
        ) <= parityData.tolerance,
        `${className} prediction exceeded the XGBoost parity tolerance`,
      );
    }
  }
});
