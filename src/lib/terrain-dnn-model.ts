import modelArtifact from "../../models/flood-risk-terrain-dnn.json" with { type: "json" };
import type { BoostedFloodRiskInput } from "./boosted-model";

type Layer = { weights: number[][]; biases: number[] };
type Artifact = {
  features: string[];
  preprocessing: {
    medianImputation: number[];
    standardizationMean: number[];
    standardizationScale: number[];
  };
  decisionThreshold: number;
  layers: Layer[];
};

const model = modelArtifact as Artifact;

export function predictTerrainDnnFloodRisk(input: BoostedFloodRiskInput) {
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
    return (imputed - model.preprocessing.standardizationMean[index]) /
      model.preprocessing.standardizationScale[index];
  });
  for (const [index, layer] of model.layers.entries()) {
    const output = index === model.layers.length - 1;
    activations = layer.biases.map((bias, neuron) => {
      let sum = bias;
      for (let feature = 0; feature < activations.length; feature++) {
        sum += activations[feature] * layer.weights[feature][neuron];
      }
      return output ? 1 / (1 + Math.exp(-Math.max(-709, Math.min(709, sum))))
        : Math.max(0, sum);
    });
  }
  const probability = activations[0];
  return {
    probability,
    scorePercent: Math.round(probability * 1_000) / 10,
    classification: probability >= model.decisionThreshold
      ? "higher-risk pattern" : "lower-risk pattern",
    imputedFeatures: raw
      .map((value, index) => value === null ? model.features[index] : null)
      .filter((value): value is string => value !== null),
  };
}
