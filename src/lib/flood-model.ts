import modelArtifact from "../../models/flood-claim-logistic-model.json" with {
  type: "json",
};

export type FloodModelInput = {
  elevation_m: number;
  sfha: number | null;
  distance_sfha_m: number | null;
  rainfall_100yr_24h_in: number | null;
  nearest_water_distance_m: number | null;
  fema_zone: string;
  fema_subtype: string;
  nearest_water_kind: string;
};

type ModelArtifact = {
  selectedThreshold: number;
  numericFeatures: string[];
  derivedNumericFeatures: string[];
  categoricalFeatures: string[];
  preprocessor: {
    numericStats: Record<
      string,
      { median: number; mean: number; standardDeviation: number }
    >;
    categories: Record<string, string[]>;
  };
  weights: number[];
};

const model = modelArtifact as ModelArtifact;

function derivedValue(feature: string, input: FloodModelInput) {
  switch (feature) {
    case "log_elevation_m":
      return Math.log1p(Math.max(0, input.elevation_m));
    case "log_distance_sfha_m":
      return input.distance_sfha_m === null
        ? null
        : Math.log1p(input.distance_sfha_m);
    case "log_nearest_water_distance_m":
      return input.nearest_water_distance_m === null
        ? null
        : Math.log1p(input.nearest_water_distance_m);
    default:
      return null;
  }
}

function numericValue(feature: string, input: FloodModelInput) {
  if (model.derivedNumericFeatures.includes(feature)) {
    return derivedValue(feature, input);
  }

  const value = input[feature as keyof FloodModelInput];
  return typeof value === "number" ? value : null;
}

function sigmoid(value: number) {
  if (value < -35) return 0;
  if (value > 35) return 1;
  return 1 / (1 + Math.exp(-value));
}

export function predictFloodClaimSimilarity(input: FloodModelInput) {
  const vector = [
    ...model.numericFeatures,
    ...model.derivedNumericFeatures,
  ].map((feature) => {
    const stats = model.preprocessor.numericStats[feature];
    const value = numericValue(feature, input) ?? stats.median;
    return (value - stats.mean) / stats.standardDeviation;
  });

  for (const feature of model.categoricalFeatures) {
    const value = String(input[feature as keyof FloodModelInput] ?? "Unknown");
    for (const category of model.preprocessor.categories[feature]) {
      vector.push(value === category ? 1 : 0);
    }
  }

  const logit = vector.reduce(
    (sum, value, index) => sum + value * model.weights[index + 1],
    model.weights[0],
  );
  const probability = sigmoid(logit);

  return {
    probability,
    score: Math.round(probability * 100),
    threshold: model.selectedThreshold,
    classification:
      probability >= model.selectedThreshold ? "claim-like" : "comparison-like",
  };
}

export function modelRiskPresentation(probability: number) {
  if (probability >= 0.75) {
    return { level: "Very high" as const, color: "#be123c" };
  }
  if (probability >= model.selectedThreshold) {
    return { level: "High" as const, color: "#dc2626" };
  }
  if (probability >= model.selectedThreshold * 0.6) {
    return { level: "Moderate" as const, color: "#d97706" };
  }
  return { level: "Low" as const, color: "#15803d" };
}
