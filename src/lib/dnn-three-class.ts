import modelArtifact from "../../models/flood-hazard-three-class-dnn.json" with { type: "json" };

export type MappedHazardClass = "minimal" | "point_two_percent" | "one_percent";

export type ThreeClassDnnInput = {
  elevationMeters: number;
  imperviousnessPercent: number | null;
  rainfallMillimeters: number | null;
  nearestWaterAvailable: number | null;
  nearestWaterDistanceMeters: number | null;
};

type Layer = { weights: number[][]; biases: number[] };
type Artifact = {
  features: string[];
  classes: MappedHazardClass[];
  preprocessing: {
    medianImputation: number[];
    standardizationMean: number[];
    standardizationScale: number[];
  };
  layers: Layer[];
};

const model = modelArtifact as Artifact;

function rawFeatureValues(input: ThreeClassDnnInput) {
  return [
    input.elevationMeters,
    input.imperviousnessPercent,
    input.rainfallMillimeters,
    input.nearestWaterAvailable,
    input.nearestWaterDistanceMeters === null
      ? null : Math.log1p(Math.max(0, input.nearestWaterDistanceMeters)),
  ];
}

export function predictMappedHazardClass(input: ThreeClassDnnInput) {
  const raw = rawFeatureValues(input);
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

  const shifted = activations.map((value) => Math.exp(value - Math.max(...activations)));
  const total = shifted.reduce((sum, value) => sum + value, 0);
  const scores = shifted.map((value) => value / total);
  const bestIndex = scores.indexOf(Math.max(...scores));
  const classScores = Object.fromEntries(
    model.classes.map((name, index) => [name, scores[index]]),
  ) as Record<MappedHazardClass, number>;

  return {
    classLabel: model.classes[bestIndex],
    classScores,
    imputedFeatures: raw
      .map((value, index) => value === null ? model.features[index] : null)
      .filter((value): value is string => value !== null),
  };
}
