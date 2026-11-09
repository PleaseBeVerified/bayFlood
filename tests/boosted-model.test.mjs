import assert from "node:assert/strict";
import test from "node:test";
import report from "../reports/terrain-fema-baselines.json" with { type: "json" };
import { predictBoostedFloodRisk } from "../src/lib/boosted-model.ts";

test("exported terrain model matches Python reference predictions", () => {
  for (const reference of report.variants.full.referenceCases) {
    const [elevationMeters, imperviousnessPercent, rainfallMillimeters,
      nearestWaterAvailable, logDistance, slopeDegrees, localElevation250Meters,
      localElevation500Meters, logFlowAccumulation, heightAboveDrainageMeters] = reference.inputs;
    const result = predictBoostedFloodRisk({
      elevationMeters,
      imperviousnessPercent,
      rainfallMillimeters,
      nearestWaterAvailable,
      nearestWaterDistanceMeters: logDistance === null ? null : Math.expm1(logDistance),
      slopeDegrees,
      localElevation250Meters,
      localElevation500Meters,
      flowAccumulationSquareMeters: Math.expm1(logFlowAccumulation),
      heightAboveDrainageMeters,
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
    slopeDegrees: 2,
    localElevation250Meters: 3,
    localElevation500Meters: 4,
    flowAccumulationSquareMeters: 500,
    heightAboveDrainageMeters: 5,
  });
  assert.deepEqual(result.missingFeatures, [
    "current_imperviousness_pct",
    "current_100yr_24h_rainfall_mm",
    "nearest_water_available",
    "log_nearest_water_distance_m",
  ]);
});
