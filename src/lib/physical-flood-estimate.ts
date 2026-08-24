import modelArtifact from "../../models/flood-physical-hazard-model.json" with {
  type: "json",
};

export type PhysicalFloodEstimateInput = {
  elevation_m: number;
  rainfall_100yr_24h_in: number | null;
  nearest_water_distance_m: number | null;
  nearest_water_kind: string;
  slope_degrees: number | null;
  local_elevation_250m: number | null;
  local_elevation_500m: number | null;
  flow_accumulation_m2: number | null;
  height_above_drainage_m: number | null;
};

type NumericRange = {
  minimum: number;
  maximum: number;
};

type BoostedTree = {
  leftChildren: number[];
  rightChildren: number[];
  splitIndices: number[];
  splitConditions: number[];
  defaultLeft: number[];
};

type BoostedBinaryModel = {
  baseScore: number;
  trees: BoostedTree[];
};

type PhysicalModelArtifact = {
  training: {
    datasetRows?: number;
    rows: number;
    testRows?: number;
    spatialFoldDegrees: number;
  };
  annualChanceAnchors: {
    one_percent: number;
    point_two_percent: number;
    minimal: number;
  };
  numericFeatures: string[];
  numericRanges: Record<string, NumericRange>;
  categoricalFeatures: Record<string, string[]>;
  featureNames: string[];
  terrainConfiguration?: {
    runtimeResolutionMeters: number;
  };
  models: {
    onePercent: BoostedBinaryModel;
    atLeastModerate: BoostedBinaryModel;
  };
  spatialValidation: {
    meanOnePercentAuc: number;
    meanAtLeastModerateAuc: number;
    meanClassAccuracy: number;
  };
};

const model = modelArtifact as PhysicalModelArtifact;

function probabilityOverYears(annualChance: number, years: number) {
  return 1 - (1 - annualChance) ** years;
}

function numericValue(feature: string, input: PhysicalFloodEstimateInput) {
  if (feature === "log_elevation_m") {
    return Math.log1p(Math.max(0, input.elevation_m));
  }
  if (feature === "log_nearest_water_distance_m") {
    return input.nearest_water_distance_m === null
      ? null
      : Math.log1p(input.nearest_water_distance_m);
  }
  if (feature === "log_flow_accumulation_m2") {
    return input.flow_accumulation_m2 === null
      ? null
      : Math.log1p(Math.max(0, input.flow_accumulation_m2));
  }
  if (feature === "log_height_above_drainage_m") {
    return input.height_above_drainage_m === null
      ? null
      : Math.log1p(Math.max(0, input.height_above_drainage_m));
  }
  const value = input[feature as keyof PhysicalFloodEstimateInput];
  return typeof value === "number" ? value : null;
}

function vectorize(input: PhysicalFloodEstimateInput) {
  const vector = model.numericFeatures.map(
    (feature) => numericValue(feature, input) ?? Number.NaN,
  );
  for (const [feature, categories] of Object.entries(
    model.categoricalFeatures,
  )) {
    const value = String(
      input[feature as keyof PhysicalFloodEstimateInput] ?? "Unknown",
    );
    for (const category of categories) {
      vector.push(value === category ? 1 : 0);
    }
  }
  return vector;
}

function sigmoid(value: number) {
  if (value < -35) return 0;
  if (value > 35) return 1;
  return 1 / (1 + Math.exp(-value));
}

function evaluateTree(tree: BoostedTree, vector: number[]) {
  let node = 0;
  while (tree.leftChildren[node] !== -1) {
    const value = vector[tree.splitIndices[node]];
    const goLeft = Number.isNaN(value)
      ? tree.defaultLeft[node] === 1
      : value < tree.splitConditions[node];
    node = goLeft ? tree.leftChildren[node] : tree.rightChildren[node];
  }
  return tree.splitConditions[node];
}

function predict(boostedModel: BoostedBinaryModel, vector: number[]) {
  const baseScore = Math.min(Math.max(boostedModel.baseScore, 1e-12), 1 - 1e-12);
  const baseMargin = Math.log(baseScore / (1 - baseScore));
  const margin = boostedModel.trees.reduce(
    (sum, tree) => sum + evaluateTree(tree, vector),
    baseMargin,
  );
  return sigmoid(margin);
}

function roundedPercent(probability: number) {
  return Math.round(probability * 1_000) / 10;
}

function isOutsideTrainingRange(input: PhysicalFloodEstimateInput) {
  return model.numericFeatures.some((feature) => {
    const value = numericValue(feature, input);
    const range = model.numericRanges[feature];
    return value !== null && (value < range.minimum || value > range.maximum);
  });
}

export function estimatePhysicalHundredYearFloodChance(
  input: PhysicalFloodEstimateInput,
) {
  const vector = vectorize(input);
  const onePercentProbability = predict(model.models.onePercent, vector);
  const atLeastModerateProbability = Math.max(
    onePercentProbability,
    predict(model.models.atLeastModerate, vector),
  );
  const classProbabilities = {
    onePercent: onePercentProbability,
    pointTwoPercent: atLeastModerateProbability - onePercentProbability,
    minimal: 1 - atLeastModerateProbability,
  };
  const annualChance =
    classProbabilities.onePercent * model.annualChanceAnchors.one_percent +
    classProbabilities.pointTwoPercent *
      model.annualChanceAnchors.point_two_percent +
    classProbabilities.minimal * model.annualChanceAnchors.minimal;
  const lowerAnnualChance =
    classProbabilities.onePercent * model.annualChanceAnchors.one_percent +
    classProbabilities.pointTwoPercent *
      model.annualChanceAnchors.point_two_percent;
  const upperAnnualChance =
    lowerAnnualChance +
    classProbabilities.minimal *
      model.annualChanceAnchors.point_two_percent;
  const percent = roundedPercent(probabilityOverYears(annualChance, 100));
  const lowerPercent = roundedPercent(
    probabilityOverYears(lowerAnnualChance, 100),
  );
  const upperPercent = roundedPercent(
    probabilityOverYears(upperAnnualChance, 100),
  );
  const missingInputs = [
    input.rainfall_100yr_24h_in,
    input.nearest_water_distance_m,
    input.slope_degrees,
    input.local_elevation_250m,
    input.local_elevation_500m,
    input.flow_accumulation_m2,
    input.height_above_drainage_m,
  ].filter((value) => value === null).length;
  const outsideTrainingRange = isOutsideTrainingRange(input);
  const confidence =
    missingInputs === 0 && !outsideTrainingRange ? "Medium" : "Low";

  return {
    horizonYears: 100 as const,
    status: "experimental_physical_estimate" as const,
    percent,
    display: `${percent}%`,
    annualChancePercent: Math.round(annualChance * 100_000) / 1_000,
    range: { lowerPercent, upperPercent },
    confidence,
    outsideTrainingRange,
    addressFemaRecordRequired: false as const,
    inputsUsed: [
      "elevation",
      "NOAA 100-year/24-hour rainfall",
      "nearest mapped water distance",
      "nearest mapped water type",
      "terrain slope",
      "local elevation within 250 m and 500 m",
      "upslope flow accumulation",
      "height above modeled drainage",
    ],
    classProbabilities,
    basis:
      "Gradient-boosted physical-only model estimate based on similarity to Bay Area locations in FEMA-mapped 1%, 0.2%, and minimal-hazard classes. FEMA data for this address is not required.",
    caveat:
      `Experimental screening estimate, not an official FEMA or parcel-level probability. The central estimate assumes a 0.1% annual chance for minimal-hazard terrain; its corresponding uncertainty range is ${lowerPercent}%-${upperPercent}%.`,
    validation: {
      samples: model.training.datasetRows ?? model.training.rows,
      untouchedTestSamples: model.training.testRows ?? null,
      terrainResolutionMeters:
        model.terrainConfiguration?.runtimeResolutionMeters ?? null,
      spatialHoldoutDegrees: model.training.spatialFoldDegrees,
      onePercentRocAuc:
        Math.round(model.spatialValidation.meanOnePercentAuc * 1_000) / 1_000,
      atLeastModerateRocAuc:
        Math.round(model.spatialValidation.meanAtLeastModerateAuc * 1_000) /
        1_000,
      classAccuracy:
        Math.round(model.spatialValidation.meanClassAccuracy * 1_000) / 1_000,
    },
  };
}
