import assert from "node:assert/strict";
import test from "node:test";
import report from "../reports/terrain-three-class-dnn-evaluation.json" with { type: "json" };
import dnnArtifact from "../models/flood-hazard-terrain-three-class-dnn.json" with { type: "json" };
import boostedArtifact from "../models/flood-risk-terrain-full.json" with { type: "json" };
import { predictTerrainThreeClass } from "../src/lib/terrain-three-class-model.ts";

test("three-class terrain DNN uses the same ordered inputs as the boosted model", () => {
  assert.deepEqual(dnnArtifact.features, boostedArtifact.features);
  assert.deepEqual(report.features, boostedArtifact.features);
  assert.deepEqual(dnnArtifact.classes, ["minimal", "one_percent", "point_two_percent"]);
});

test("three-class terrain DNN matches exported Python reference probabilities", () => {
  for (const reference of report.referenceCases) {
    const [elevationMeters, imperviousnessPercent, rainfallMillimeters,
      nearestWaterAvailable, logDistance, slopeDegrees, localElevation250Meters,
      localElevation500Meters, logFlowAccumulation, heightAboveDrainageMeters] = reference.inputs;
    const prediction = predictTerrainThreeClass({
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
    for (const [index, classLabel] of dnnArtifact.classes.entries()) {
      assert.ok(Math.abs(prediction.classScores[classLabel] - reference.probabilities[index]) < 1e-10);
    }
    assert.equal(prediction.classLabel,
      dnnArtifact.classes[reference.probabilities.indexOf(Math.max(...reference.probabilities))]);
    assert.ok(Math.abs(Object.values(prediction.classScores).reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
  }
});

test("three-class terrain DNN reports imputed water inputs", () => {
  const prediction = predictTerrainThreeClass({
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
    "nearest_water_available", "log_nearest_water_distance_m",
  ]);
});
