import { mkdir, readFile, writeFile } from "node:fs/promises";

const inputPath = "data/training-table.csv";
const outputPath = "models/flood-claim-logistic-model.json";
const testFraction = 0.25;
const validationFoldCount = 5;
const seed = 42;

const numericFeatures = [
  "sfha",
  "rainfall_100yr_24h_in",
];

const derivedNumericFeatures = [
  {
    name: "log_elevation_m",
    calculate: (row) =>
      Math.log1p(Math.max(0, parseNumber(row.elevation_m) ?? 0)),
  },
  {
    name: "log_distance_sfha_m",
    calculate: (row) => {
      const distance = parseNumber(row.distance_sfha_m);
      return distance === null ? null : Math.log1p(distance);
    },
  },
  {
    name: "log_nearest_water_distance_m",
    calculate: (row) => {
      const distance = parseNumber(row.nearest_water_distance_m);
      return distance === null ? null : Math.log1p(distance);
    },
  },
];

const categoricalFeatures = [
  "fema_zone",
  "fema_subtype",
  "nearest_water_kind",
];

const monotonicConstraints = {
  sfha: "increasing",
  rainfall_100yr_24h_in: "increasing",
  log_elevation_m: "decreasing",
  log_distance_sfha_m: "decreasing",
  log_nearest_water_distance_m: "decreasing",
};

const hyperparameterGrid = [
  { learningRate: 0.02, l2Penalty: 0.0005, epochs: 4_000 },
  { learningRate: 0.025, l2Penalty: 0.001, epochs: 5_000 },
  { learningRate: 0.03, l2Penalty: 0.002, epochs: 5_000 },
  { learningRate: 0.035, l2Penalty: 0.003, epochs: 6_000 },
  { learningRate: 0.045, l2Penalty: 0.006, epochs: 6_000 },
  { learningRate: 0.055, l2Penalty: 0.01, epochs: 7_000 },
];

function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    const nextCharacter = line[index + 1];

    if (character === '"' && quoted && nextCharacter === '"') {
      value += '"';
      index += 1;
    } else if (character === '"') {
      quoted = !quoted;
    } else if (character === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += character;
    }
  }

  values.push(value);
  return values;
}

function parseCsv(contents) {
  const lines = contents.split(/\r?\n/).filter(Boolean);
  const headers = parseCsvLine(lines[0]);

  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function mulberry32(initialSeed) {
  let state = initialSeed;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function shuffle(rows) {
  const random = mulberry32(seed);
  const shuffled = [...rows];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

function shuffleWithSeed(rows, splitSeed) {
  const random = mulberry32(splitSeed);
  const shuffled = [...rows];

  for (let index = shuffled.length - 1; index > 0; index -= 1) {
    const swapIndex = Math.floor(random() * (index + 1));
    [shuffled[index], shuffled[swapIndex]] = [shuffled[swapIndex], shuffled[index]];
  }

  return shuffled;
}

function locationKey(row) {
  return `${Number(row.lat).toFixed(5)},${Number(row.lng).toFixed(5)}`;
}

function groupByLocation(rows) {
  const grouped = new Map();

  for (const row of rows) {
    const key = locationKey(row);
    const group = grouped.get(key) ?? [];
    group.push(row);
    grouped.set(key, group);
  }

  const groups = [...grouped.values()];
  for (const group of groups) {
    const labels = new Set(group.map((row) => row.flooded));
    if (labels.size !== 1) {
      throw new Error(
        `Location ${locationKey(group[0])} has conflicting flooded labels.`,
      );
    }
  }

  return groups;
}

function selectGroupsNearTarget(groups, targetRows, splitSeed) {
  const candidates = shuffleWithSeed(groups, splitSeed).sort(
    (a, b) => b.length - a.length,
  );
  const selected = [];
  let selectedRows = 0;

  for (const group of candidates) {
    if (selectedRows + group.length <= targetRows) {
      selected.push(group);
      selectedRows += group.length;
    }
  }

  if (selected.length === 0 && candidates.length > 0) {
    selected.push(candidates.at(-1));
  }

  return new Set(selected);
}

function stratifiedSplit(rows) {
  const groups = groupByLocation(rows);
  const positiveGroups = groups.filter((group) => group[0].flooded === "1");
  const negativeGroups = groups.filter((group) => group[0].flooded === "0");
  const positiveTarget = Math.max(
    1,
    Math.round(
      positiveGroups.reduce((sum, group) => sum + group.length, 0) * testFraction,
    ),
  );
  const negativeTarget = Math.max(
    1,
    Math.round(
      negativeGroups.reduce((sum, group) => sum + group.length, 0) * testFraction,
    ),
  );
  const testGroups = new Set([
    ...selectGroupsNearTarget(positiveGroups, positiveTarget, seed + 11),
    ...selectGroupsNearTarget(negativeGroups, negativeTarget, seed + 22),
  ]);

  return {
    trainRows: shuffle(groups.filter((group) => !testGroups.has(group)).flat()),
    testRows: shuffle([...testGroups].flat()),
  };
}

function distributeGroupsAcrossFolds(groups, folds, counts, splitSeed) {
  const candidates = shuffleWithSeed(groups, splitSeed).sort(
    (a, b) => b.length - a.length,
  );

  for (const group of candidates) {
    const smallestFoldIndex = counts
      .map((count, index) => ({ count, index }))
      .sort((a, b) => a.count - b.count || a.index - b.index)[0].index;
    folds[smallestFoldIndex].push(...group);
    counts[smallestFoldIndex] += group.length;
  }
}

function stratifiedFolds(rows, foldCount) {
  const folds = Array.from({ length: foldCount }, () => []);
  const positiveCounts = Array(foldCount).fill(0);
  const negativeCounts = Array(foldCount).fill(0);
  const groups = groupByLocation(rows);

  distributeGroupsAcrossFolds(
    groups.filter((group) => group[0].flooded === "1"),
    folds,
    positiveCounts,
    seed + 101,
  );
  distributeGroupsAcrossFolds(
    groups.filter((group) => group[0].flooded === "0"),
    folds,
    negativeCounts,
    seed + 202,
  );

  return folds.map((fold, index) =>
    shuffleWithSeed(fold, seed + index + 303),
  );
}

function parseNumber(value) {
  if (value === null || value === undefined || String(value).trim() === "") {
    return null;
  }
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function median(values) {
  const sorted = values.filter((value) => value !== null).sort((a, b) => a - b);
  if (sorted.length === 0) return 0;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle];
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function standardDeviation(values, valueMean) {
  const variance = mean(values.map((value) => (value - valueMean) ** 2));
  return Math.sqrt(variance) || 1;
}

function buildPreprocessor(trainRows) {
  const numericStats = Object.fromEntries(
    [...numericFeatures, ...derivedNumericFeatures.map((feature) => feature.name)].map((feature) => {
      const imputeValue = median(trainRows.map((row) => numericFeatureValue(row, feature)));
      const values = trainRows.map((row) => numericFeatureValue(row, feature) ?? imputeValue);
      const featureMean = mean(values);

      return [
        feature,
        {
          median: imputeValue,
          mean: featureMean,
          standardDeviation: standardDeviation(values, featureMean),
        },
      ];
    }),
  );
  const categories = Object.fromEntries(
    categoricalFeatures.map((feature) => [
      feature,
      [...new Set(trainRows.map((row) => row[feature] || "Unknown"))].sort(),
    ]),
  );
  const featureNames = [
    ...numericFeatures,
    ...derivedNumericFeatures.map((feature) => feature.name),
    ...categoricalFeatures.flatMap((feature) =>
      categories[feature].map((category) => `${feature}=${category}`),
    ),
  ];

  return { numericStats, categories, featureNames };
}

function numericFeatureValue(row, feature) {
  const derivedFeature = derivedNumericFeatures.find((candidate) => candidate.name === feature);
  return derivedFeature ? derivedFeature.calculate(row) : parseNumber(row[feature]);
}

function vectorize(row, preprocessor) {
  const vector = [...numericFeatures, ...derivedNumericFeatures.map((feature) => feature.name)].map((feature) => {
    const stats = preprocessor.numericStats[feature];
    const value = numericFeatureValue(row, feature) ?? stats.median;
    return (value - stats.mean) / stats.standardDeviation;
  });

  for (const feature of categoricalFeatures) {
    const value = row[feature] || "Unknown";
    for (const category of preprocessor.categories[feature]) {
      vector.push(value === category ? 1 : 0);
    }
  }

  return vector;
}

function sigmoid(value) {
  if (value < -35) return 0;
  if (value > 35) return 1;
  return 1 / (1 + Math.exp(-value));
}

function dot(weights, features) {
  return weights.reduce((sum, weight, index) => sum + weight * features[index], 0);
}

function train(trainRows, preprocessor, hyperparameters) {
  const examples = trainRows.map((row) => ({
    x: [1, ...vectorize(row, preprocessor)],
    y: Number(row.flooded),
  }));
  const positiveCount = examples.filter((example) => example.y === 1).length;
  const negativeCount = examples.length - positiveCount;
  const classWeights = {
    0: examples.length / (2 * negativeCount),
    1: examples.length / (2 * positiveCount),
  };
  const weights = Array(examples[0].x.length).fill(0);

  for (let epoch = 0; epoch < hyperparameters.epochs; epoch += 1) {
    const gradients = Array(weights.length).fill(0);

    for (const example of examples) {
      const prediction = sigmoid(dot(weights, example.x));
      const error = prediction - example.y;
      const exampleWeight = classWeights[example.y];

      for (let index = 0; index < weights.length; index += 1) {
        gradients[index] += exampleWeight * error * example.x[index];
      }
    }

    for (let index = 0; index < weights.length; index += 1) {
      const regularization = index === 0 ? 0 : hyperparameters.l2Penalty * weights[index];
      weights[index] -= hyperparameters.learningRate * (gradients[index] / examples.length + regularization);
    }

    for (const [feature, direction] of Object.entries(monotonicConstraints)) {
      const weightIndex = preprocessor.featureNames.indexOf(feature) + 1;
      if (weightIndex === 0) continue;
      weights[weightIndex] =
        direction === "increasing"
          ? Math.max(0, weights[weightIndex])
          : Math.min(0, weights[weightIndex]);
    }
  }

  return weights;
}

function predictProbability(row, model) {
  return sigmoid(dot(model.weights, [1, ...vectorize(row, model.preprocessor)]));
}

function calculateAuc(scoredRows) {
  const sortedRows = [...scoredRows].sort((a, b) => a.probability - b.probability);
  let rankSum = 0;
  let positiveCount = 0;
  let negativeCount = 0;

  sortedRows.forEach((row, index) => {
    if (row.actual === 1) {
      positiveCount += 1;
      rankSum += index + 1;
    } else {
      negativeCount += 1;
    }
  });

  if (positiveCount === 0 || negativeCount === 0) return null;
  return (rankSum - (positiveCount * (positiveCount + 1)) / 2) / (positiveCount * negativeCount);
}

function evaluate(rows, model, threshold = 0.5) {
  const scoredRows = rows.map((row) => ({
    actual: Number(row.flooded),
    probability: predictProbability(row, model),
  }));
  const confusion = { truePositive: 0, trueNegative: 0, falsePositive: 0, falseNegative: 0 };

  for (const row of scoredRows) {
    const predicted = row.probability >= threshold ? 1 : 0;

    if (predicted === 1 && row.actual === 1) confusion.truePositive += 1;
    if (predicted === 0 && row.actual === 0) confusion.trueNegative += 1;
    if (predicted === 1 && row.actual === 0) confusion.falsePositive += 1;
    if (predicted === 0 && row.actual === 1) confusion.falseNegative += 1;
  }

  const precision =
    confusion.truePositive + confusion.falsePositive === 0
      ? 0
      : confusion.truePositive / (confusion.truePositive + confusion.falsePositive);
  const recall =
    confusion.truePositive + confusion.falseNegative === 0
      ? 0
      : confusion.truePositive / (confusion.truePositive + confusion.falseNegative);
  const specificity =
    confusion.trueNegative + confusion.falsePositive === 0
      ? 0
      : confusion.trueNegative / (confusion.trueNegative + confusion.falsePositive);
  const accuracy =
    (confusion.truePositive + confusion.trueNegative) /
    (confusion.truePositive +
      confusion.trueNegative +
      confusion.falsePositive +
      confusion.falseNegative);
  const brierScore = mean(
    scoredRows.map((row) => (row.probability - row.actual) ** 2),
  );

  return {
    threshold,
    accuracy,
    precision,
    recall,
    specificity,
    rocAuc: calculateAuc(scoredRows),
    brierScore,
    confusion,
  };
}

function thresholdCandidates() {
  const thresholds = [];
  for (let threshold = 0.2; threshold <= 0.8; threshold += 0.025) {
    thresholds.push(Number(threshold.toFixed(3)));
  }
  return thresholds;
}

function selectBestThreshold(rows, model) {
  return thresholdCandidates()
    .map((threshold) => evaluate(rows, model, threshold))
    .sort((a, b) => {
      if (b.accuracy !== a.accuracy) return b.accuracy - a.accuracy;
      if (b.rocAuc !== a.rocAuc) return (b.rocAuc ?? 0) - (a.rocAuc ?? 0);
      return b.recall - a.recall;
    })[0];
}

function nearestThreshold(value) {
  return thresholdCandidates()
    .map((threshold) => ({
      threshold,
      distance: Math.abs(threshold - value),
    }))
    .sort((a, b) => a.distance - b.distance)[0].threshold;
}

function averageMetrics(metrics) {
  const metricNames = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "rocAuc",
    "brierScore",
  ];
  return Object.fromEntries(
    metricNames.map((metricName) => [
      metricName,
      mean(metrics.map((metric) => metric[metricName] ?? 0)),
    ]),
  );
}

function crossValidate(rows) {
  const folds = stratifiedFolds(rows, validationFoldCount);

  return hyperparameterGrid
    .map((hyperparameters) => {
      const foldMetrics = folds.map((validationRows, foldIndex) => {
        const foldTrainRows = folds.flatMap((fold, index) =>
          index === foldIndex ? [] : fold,
        );
        const preprocessor = buildPreprocessor(foldTrainRows);
        const weights = train(foldTrainRows, preprocessor, hyperparameters);
        const model = { preprocessor, weights };

        return selectBestThreshold(validationRows, model);
      });

      return {
        ...hyperparameters,
        folds: foldMetrics.map((metrics) => ({
          threshold: metrics.threshold,
          accuracy: metrics.accuracy,
          precision: metrics.precision,
          recall: metrics.recall,
          specificity: metrics.specificity,
          rocAuc: metrics.rocAuc,
          brierScore: metrics.brierScore,
        })),
        meanMetrics: averageMetrics(foldMetrics),
        meanThreshold: mean(foldMetrics.map((metrics) => metrics.threshold)),
      };
    })
    .sort((a, b) => {
      if (b.meanMetrics.rocAuc !== a.meanMetrics.rocAuc) {
        return b.meanMetrics.rocAuc - a.meanMetrics.rocAuc;
      }
      return b.meanMetrics.accuracy - a.meanMetrics.accuracy;
    });
}

function topWeights(model) {
  return model.preprocessor.featureNames
    .map((feature, index) => ({
      feature,
      weight: model.weights[index + 1],
    }))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 15);
}

async function main() {
  const rows = parseCsv(await readFile(inputPath, "utf8"));
  const validRows = rows.filter((row) => row.flooded === "0" || row.flooded === "1");
  const { trainRows, testRows } = stratifiedSplit(validRows);
  const crossValidationResults = crossValidate(trainRows);
  const bestRun = crossValidationResults[0];
  const hyperparameters = {
    learningRate: bestRun.learningRate,
    l2Penalty: bestRun.l2Penalty,
    epochs: bestRun.epochs,
  };
  const preprocessor = buildPreprocessor(trainRows);
  const weights = train(trainRows, preprocessor, hyperparameters);
  const model = {
    modelType: "logistic_regression",
    target: "flooded",
    meaning: "Probability that a location resembles known NFIP flood-claim points in this dataset.",
    createdAt: new Date().toISOString(),
    sourceTable: inputPath,
    training: {
      rows: trainRows.length,
      uniqueLocations: groupByLocation(trainRows).length,
      positives: trainRows.filter((row) => row.flooded === "1").length,
      negatives: trainRows.filter((row) => row.flooded === "0").length,
      ...hyperparameters,
    },
    crossValidation: {
      folds: validationFoldCount,
      selectedBy: "mean ROC AUC, then mean accuracy",
      bestMeanMetrics: bestRun.meanMetrics,
      bestMeanThreshold: bestRun.meanThreshold,
      runs: crossValidationResults,
    },
    validation: {
      rows: testRows.length,
      uniqueLocations: groupByLocation(testRows).length,
      positives: testRows.filter((row) => row.flooded === "1").length,
      negatives: testRows.filter((row) => row.flooded === "0").length,
    },
    numericFeatures,
    derivedNumericFeatures: derivedNumericFeatures.map((feature) => feature.name),
    categoricalFeatures,
    monotonicConstraints,
    preprocessor,
    weights,
  };
  const selectedThreshold = nearestThreshold(bestRun.meanThreshold);
  const metrics = evaluate(testRows, model, selectedThreshold);

  const thresholds = [0.3, 0.35, 0.4, selectedThreshold, 0.45, 0.5, 0.55, 0.6]
    .filter((threshold, index, allThresholds) => allThresholds.indexOf(threshold) === index)
    .sort((a, b) => a - b);
  console.log("Cross-validation leaderboard:");
  for (const run of crossValidationResults.slice(0, 5)) {
    console.log({
      learningRate: run.learningRate,
      l2Penalty: run.l2Penalty,
      epochs: run.epochs,
      meanThreshold: Number(run.meanThreshold.toFixed(3)),
      meanAccuracy: Number(run.meanMetrics.accuracy.toFixed(3)),
      meanRecall: Number(run.meanMetrics.recall.toFixed(3)),
      meanRocAuc: Number(run.meanMetrics.rocAuc.toFixed(3)),
    });
  }

  console.log("Threshold report:");
  for (const threshold of thresholds) {
    const thresholdMetrics = evaluate(testRows, model, threshold);

    console.log({
      threshold,
      accuracy: Number(thresholdMetrics.accuracy.toFixed(3)),
      precision: Number(thresholdMetrics.precision.toFixed(3)),
      recall: Number(thresholdMetrics.recall.toFixed(3)),
      truePositive: thresholdMetrics.confusion.truePositive,
      trueNegative: thresholdMetrics.confusion.trueNegative,
      falsePositive: thresholdMetrics.confusion.falsePositive,
      falseNegative: thresholdMetrics.confusion.falseNegative,
      brierScore: Number(thresholdMetrics.brierScore.toFixed(3)),
    });
  }

  const rankedWeights = topWeights(model);
  const output = { ...model, selectedThreshold: metrics.threshold, metrics, topWeights: rankedWeights };

  await mkdir("models", { recursive: true });
  await writeFile(outputPath, JSON.stringify(output, null, 2) + "\n");

  console.log(`Trained on ${model.training.rows} rows; validated on ${model.validation.rows} rows.`);
  console.log(
    `Validation: accuracy ${metrics.accuracy.toFixed(3)}, precision ${metrics.precision.toFixed(3)}, recall ${metrics.recall.toFixed(3)}, ROC AUC ${metrics.rocAuc?.toFixed(3) ?? "n/a"}, Brier ${metrics.brierScore.toFixed(3)}.`,
  );
  console.log(`Selected threshold: ${metrics.threshold.toFixed(3)}`);
  console.log("Confusion matrix:", metrics.confusion);
  console.log("Top learned signals:");
  for (const { feature, weight } of rankedWeights.slice(0, 8)) {
    console.log(`  ${feature}: ${weight.toFixed(3)}`);
  }
  console.log(`Wrote ${outputPath}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
