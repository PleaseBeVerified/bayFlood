import assert from "node:assert/strict";
import test from "node:test";
import report from "../reports/boosted-dnn-comparison.json" with { type: "json" };
import { predictBoostedFloodRisk } from "../src/lib/boosted-model.ts";

test("exported boosted model matches Python reference predictions", () => {
  for (const reference of report.referenceCases) {
    const [elevationMeters, imperviousnessPercent, rainfallMillimeters,
      nearestWaterAvailable, logDistance] = reference.inputs;
    const result = predictBoostedFloodRisk({
      elevationMeters,
      imperviousnessPercent,
      rainfallMillimeters,
      nearestWaterAvailable,
      nearestWaterDistanceMeters: logDistance === null ? null : Math.expm1(logDistance),
    });
    assert.ok(Math.abs(result.probability - reference.probability) < 0.00001);
  }
});

test("missing inputs are reported without pretending they were measured", () => {
  const result = predictBoostedFloodRisk({
    elevationMeters: 25,
    imperviousnessPercent: null,
    rainfallMillimeters: null,
    nearestWaterAvailable: null,
    nearestWaterDistanceMeters: null,
  });
  assert.deepEqual(result.missingFeatures, [
    "current_imperviousness_pct",
    "current_100yr_24h_rainfall_mm",
    "nearest_water_available",
    "log_nearest_water_distance_m",
  ]);
});
