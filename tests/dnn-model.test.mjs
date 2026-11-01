import assert from "node:assert/strict";
import test from "node:test";
import {
  predictDnnFloodRisk,
} from "../src/lib/dnn-model.ts";

test("evaluates the exported DNN and records imputed missing features", () => {
  const prediction = predictDnnFloodRisk({
    elevationMeters: 25,
    imperviousnessPercent: 60,
    rainfallMillimeters: 150,
    nearestWaterAvailable: 1,
    nearestWaterDistanceMeters: 80,
  });

  assert.ok(prediction.probability >= 0 && prediction.probability <= 1);
  assert.ok(prediction.scorePercent >= 0 && prediction.scorePercent <= 100);
  assert.equal(prediction.imputedFeatures.length, 0);

  const missingWater = predictDnnFloodRisk({
    elevationMeters: 25,
    imperviousnessPercent: 60,
    rainfallMillimeters: 150,
    nearestWaterAvailable: null,
    nearestWaterDistanceMeters: null,
  });
  assert.deepEqual(missingWater.imputedFeatures, [
    "nearest_water_available",
    "log_nearest_water_distance_m",
  ]);
});
