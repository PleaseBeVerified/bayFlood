import assert from "node:assert/strict";
import test from "node:test";
import { projectFutureInputs } from "../src/lib/future-scenarios.ts";

test("2036 and 2046 cases interpolate input changes without changing fixed geography", () => {
  const baseline = {
    elevationMeters: 10,
    imperviousnessPercent: 95,
    rainfallMillimeters: 120,
    nearestWaterAvailable: 1,
    nearestWaterDistanceMeters: 30,
  };
  const tenYear = projectFutureInputs(baseline, 10);
  const twentyYear = projectFutureInputs(baseline, 20);
  const medium2036 = tenYear.find((scenario) => scenario.id === "medium_change");
  const medium2046 = twentyYear.find((scenario) => scenario.id === "medium_change");

  assert.equal(medium2036.year, 2036);
  assert.equal(medium2036.input.rainfallMillimeters, 126);
  assert.ok(Math.abs(medium2036.input.imperviousnessPercent - 98.33333333333333) < 1e-10);
  assert.equal(medium2046.input.rainfallMillimeters, 132);
  assert.equal(medium2046.input.imperviousnessPercent, 100);
  assert.equal(medium2046.input.elevationMeters, baseline.elevationMeters);
  assert.equal(medium2046.input.nearestWaterDistanceMeters, baseline.nearestWaterDistanceMeters);
  assert.equal("flood_risk_label" in medium2046.input, false);
});
