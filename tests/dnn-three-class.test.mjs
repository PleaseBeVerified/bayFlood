import assert from "node:assert/strict";
import test from "node:test";
import { predictMappedHazardClass } from "../src/lib/dnn-three-class.ts";

test("three-class exported model matches Python probabilities", () => {
  const result = predictMappedHazardClass({
    elevationMeters: 25,
    imperviousnessPercent: 60,
    rainfallMillimeters: 150,
    nearestWaterAvailable: 1,
    nearestWaterDistanceMeters: 80,
  });
  assert.equal(result.classLabel, "minimal");
  assert.ok(Math.abs(result.classScores.minimal - 0.522238355789103) < 1e-10);
  assert.ok(Math.abs(result.classScores.one_percent - 0.11593403313709347) < 1e-10);
  assert.ok(Math.abs(result.classScores.point_two_percent - 0.3618276110738035) < 1e-10);
  assert.ok(Math.abs(Object.values(result.classScores).reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  assert.deepEqual(result.imputedFeatures, []);
});

test("three-class model reports missing input imputation", () => {
  const result = predictMappedHazardClass({
    elevationMeters: 25,
    imperviousnessPercent: 60,
    rainfallMillimeters: 150,
    nearestWaterAvailable: null,
    nearestWaterDistanceMeters: null,
  });
  assert.deepEqual(result.imputedFeatures, [
    "nearest_water_available", "log_nearest_water_distance_m",
  ]);
  assert.ok(Math.abs(result.classScores.minimal - 0.7077744391894712) < 1e-10);
});
