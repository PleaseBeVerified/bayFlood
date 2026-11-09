import assert from "node:assert/strict";
import test from "node:test";
import report from "../reports/terrain-dnn-evaluation.json" with { type: "json" };
import dnnArtifact from "../models/flood-risk-terrain-dnn.json" with { type: "json" };
import boostedArtifact from "../models/flood-risk-terrain-full.json" with { type: "json" };
import { predictTerrainDnnFloodRisk } from "../src/lib/terrain-dnn-model.ts";

test("terrain DNN and boosted model use the same ordered inputs", () => {
  assert.deepEqual(dnnArtifact.features, boostedArtifact.features);
  assert.deepEqual(report.features, boostedArtifact.features);
});

test("ten-input DNN matches Python predictions", () => {
  for (const reference of report.referenceCases) {
    const [elevationMeters, imperviousnessPercent, rainfallMillimeters,
      nearestWaterAvailable, logDistance, slopeDegrees, localElevation250Meters,
      localElevation500Meters, logFlowAccumulation, heightAboveDrainageMeters] = reference.inputs;
    const prediction = predictTerrainDnnFloodRisk({
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
    assert.ok(Math.abs(prediction.probability - reference.probability) < 1e-10);
  }
});

test("ten-input DNN reports imputation for missing water values", () => {
  const prediction = predictTerrainDnnFloodRisk({
    elevationMeters: 25,
    imperviousnessPercent: 60,
    rainfallMillimeters: 150,
    nearestWaterAvailable: null,
    nearestWaterDistanceMeters: null,
    slopeDegrees: 2,
    localElevation250Meters: 0,
    localElevation500Meters: 0,
    flowAccumulationSquareMeters: 500,
    heightAboveDrainageMeters: 5,
  });
  assert.deepEqual(prediction.imputedFeatures, [
    "nearest_water_available",
    "log_nearest_water_distance_m",
  ]);
});
