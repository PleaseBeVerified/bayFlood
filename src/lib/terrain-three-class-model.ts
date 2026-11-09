import modelArtifact from "../../models/flood-hazard-terrain-three-class-dnn.json" with { type: "json" };
import type { BoostedFloodRiskInput } from "./boosted-model";

export type TerrainHazardClass = "minimal" | "one_percent" | "point_two_percent";

type Layer = { weights: number[][]; biases: number[] };
type Artifact = {
  features: string[];
  classes: TerrainHazardClass[];
  preprocessing: {
    medianImputation: number[];
    standardizationMean: number[];
    standardizationScale: number[];
  };
  layers: Layer[];
};

const model = modelArtifact as Artifact;

export function predictTerrainThreeClass(input: BoostedFloodRiskInput) {
  const raw = [
    input.elevationMeters,
    input.imperviousnessPercent,
    input.rainfallMillimeters,
    input.nearestWaterAvailable,
    input.nearestWaterDistanceMeters === null
      ? null : Math.log1p(Math.max(0, input.nearestWaterDistanceMeters)),
    input.slopeDegrees,
    input.localElevation250Meters,
    input.localElevation500Meters,
    Math.log1p(Math.max(0, input.flowAccumulationSquareMeters)),
    input.heightAboveDrainageMeters,
  ];
  let activations = raw.map((value, index) => {
    const imputed = value ?? model.preprocessing.medianImputation[index];
    return (imputed - model.preprocessing.standardizationMean[index])
      / model.preprocessing.standardizationScale[index];
  });

  for (const [layerIndex, layer] of model.layers.entries()) {
    const values = layer.biases.map((bias, neuronIndex) =>
      activations.reduce(
        (sum, activation, featureIndex) =>
          sum + activation * layer.weights[featureIndex][neuronIndex],
        bias,
      ),
    );
    activations = layerIndex === model.layers.length - 1
      ? values : values.map((value) => Math.max(0, value));
  }

  const maximum = Math.max(...activations);
  const shifted = activations.map((value) => Math.exp(value - maximum));
  const total = shifted.reduce((sum, value) => sum + value, 0);
  const scores = shifted.map((value) => value / total);
  const bestIndex = scores.indexOf(Math.max(...scores));
  const classScores = Object.fromEntries(
    model.classes.map((name, index) => [name, scores[index]]),
  ) as Record<TerrainHazardClass, number>;

  return {
    classLabel: model.classes[bestIndex],
    classScores,
    imputedFeatures: raw
      .map((value, index) => value === null ? model.features[index] : null)
      .filter((value): value is string => value !== null),
  };
}
