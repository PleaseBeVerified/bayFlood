import modelArtifact from "../../models/flood-risk-dnn.json" with {
  type: "json",
};

export type DnnFloodRiskInput = {
  elevationMeters: number;
  imperviousnessPercent: number | null;
  rainfallMillimeters: number | null;
  nearestWaterAvailable: number | null;
  nearestWaterDistanceMeters: number | null;
};

type Layer = {
  weights: number[][];
  biases: number[];
};

type DnnArtifact = {
  features: string[];
  preprocessing: {
    medianImputation: number[];
    standardizationMean: number[];
    standardizationScale: number[];
  };
  decisionThreshold: number;
  layers: Layer[];
};

const model = modelArtifact as DnnArtifact;

function sigmoid(value: number) {
  if (value < -35) return 0;
  if (value > 35) return 1;
  return 1 / (1 + Math.exp(-value));
}

function dot(values: number[], weights: number[], bias: number) {
  return values.reduce((sum, value, index) => sum + value * weights[index], bias);
}

function rawFeatureValues(input: DnnFloodRiskInput) {
  return [
    input.elevationMeters,
    input.imperviousnessPercent,
    input.rainfallMillimeters,
    input.nearestWaterAvailable,
    input.nearestWaterDistanceMeters === null
      ? null
      : Math.log1p(Math.max(0, input.nearestWaterDistanceMeters)),
  ];
}

export function predictDnnFloodRisk(input: DnnFloodRiskInput) {
  const rawValues = rawFeatureValues(input);
  let activations = rawValues.map((value, index) => {
    const imputed = value ?? model.preprocessing.medianImputation[index];
    return (
      (imputed - model.preprocessing.standardizationMean[index]) /
      model.preprocessing.standardizationScale[index]
    );
  });

  for (const [index, layer] of model.layers.entries()) {
    const isOutputLayer = index === model.layers.length - 1;
    activations = layer.biases.map((bias, neuronIndex) => {
      const value = dot(
        activations,
        layer.weights.map((featureWeights) => featureWeights[neuronIndex]),
        bias,
      );
      return isOutputLayer ? sigmoid(value) : Math.max(0, value);
    });
  }

  const probability = activations[0];
  const threshold = model.decisionThreshold;
  const level =
    probability >= 0.85
      ? "Very high"
      : probability >= threshold
        ? "High"
        : probability >= threshold * 0.65
          ? "Moderate"
          : "Low";
  const color =
    level === "Very high"
      ? "#be123c"
      : level === "High"
        ? "#dc2626"
        : level === "Moderate"
          ? "#d97706"
          : "#15803d";

  return {
    probability,
    scorePercent: Math.round(probability * 1_000) / 10,
    decisionThreshold: threshold,
    classification: probability >= threshold ? "higher-risk pattern" : "lower-risk pattern",
    level,
    color,
    imputedFeatures: rawValues
      .map((value, index) => (value === null ? model.features[index] : null))
      .filter((value): value is string => value !== null),
  };
}
