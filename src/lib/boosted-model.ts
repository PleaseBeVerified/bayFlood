import modelArtifact from "../../models/flood-risk-boosted.json" with { type: "json" };

export type BoostedFloodRiskInput = {
  elevationMeters: number;
  imperviousnessPercent: number | null;
  rainfallMillimeters: number | null;
  nearestWaterAvailable: number | null;
  nearestWaterDistanceMeters: number | null;
};

type Tree = { leaf: number } | {
  feature: number;
  threshold: number;
  yes: Tree;
  no: Tree;
  missing: "yes" | "no";
};

type Artifact = {
  features: string[];
  baseLogit: number;
  decisionThreshold: number;
  trees: Tree[];
};

const model = modelArtifact as Artifact;

function rawFeatures(input: BoostedFloodRiskInput): Array<number | null> {
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

export function predictBoostedFloodRisk(input: BoostedFloodRiskInput) {
  const raw = rawFeatures(input);
  // XGBoost evaluates float32 feature values, including split boundaries.
  const values = raw.map((value) => value === null ? null : Math.fround(value));
  let logit = model.baseLogit;
  for (const tree of model.trees) {
    let node = tree;
    while ("feature" in node) {
      const value = values[node.feature];
      node = value === null
        ? node[node.missing]
        : value < node.threshold ? node.yes : node.no;
    }
    logit += node.leaf;
  }
  const probability = 1 / (1 + Math.exp(-logit));
  return {
    probability,
    scorePercent: Math.round(probability * 1_000) / 10,
    classification: probability >= model.decisionThreshold
      ? "higher-risk pattern" : "lower-risk pattern",
    missingFeatures: raw
      .map((value, index) => value === null ? model.features[index] : null)
      .filter((value): value is string => value !== null),
  };
}
