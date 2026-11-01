import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { parseCsv, serializeCsv } from "./lib/csv.mjs";
import {
  buildDnnScenarioTables,
  buildImperviousnessTemplate,
  imperviousnessHeaders,
  scenarioHeaders,
  threeClassTrainingHeaders,
  trainingHeaders,
} from "./lib/dnn-scenarios.mjs";

const mode = process.argv[2] ?? "build";
const includeResearch = process.argv.includes("--include-research");
const option = (name, fallback) =>
  process.argv.find((argument) => argument.startsWith(`--${name}=`))?.split("=").slice(1).join("=") ??
  fallback;

const inputPath = option("input", "data/physical-hazard-calibration.csv");
const templatePath = option(
  "template-output",
  "data/dnn-imperviousness-template.csv",
);
const imperviousnessPath = option(
  "imperviousness",
  "data/dnn-imperviousness.csv",
);
const configPath = option("config", "data/dnn-scenarios.json");
const trainingPath = option("training-output", "data/dnn-training-table.csv");
const threeClassTrainingPath = option("three-class-output", "data/dnn-three-class-training-table.csv");
const scenarioPath = option("scenario-output", "data/dnn-2056-scenarios.csv");
const reportPath = option("report-output", "reports/dnn-data-readiness.json");

if (process.argv.includes("--help") || !new Set(["template", "build"]).has(mode)) {
  console.log(`Usage:
  node scripts/build-dnn-data.mjs template [options]
  node scripts/build-dnn-data.mjs build [options]

Options:
  --input=CSV                 Physical calibration input
  --template-output=CSV       Blank keyed imperviousness template
  --imperviousness=CSV        Completed sourced imperviousness table
  --config=JSON               Projection scenario configuration
  --training-output=CSV       Current-condition binary training table
  --three-class-output=CSV    Current-condition three-class training table
  --scenario-output=CSV       Unlabeled future scenario table
  --report-output=JSON        Data-readiness report
  --include-research          Also write three-class and future-scenario CSVs`);
  process.exit(process.argv.includes("--help") ? 0 : 1);
}

async function writeOutput(filePath, contents) {
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, contents);
}

const calibration = parseCsv(await readFile(inputPath, "utf8")).rows;

if (mode === "template") {
  const rows = buildImperviousnessTemplate(calibration);
  await writeOutput(templatePath, serializeCsv(rows, imperviousnessHeaders));
  console.log(`Wrote ${rows.length} imperviousness template rows to ${templatePath}.`);
  process.exit(0);
}

let imperviousnessContents;
try {
  imperviousnessContents = await readFile(imperviousnessPath, "utf8");
} catch (error) {
  if (error?.code === "ENOENT") {
    throw new Error(
      `${imperviousnessPath} does not exist. Run npm run dnn:data:sample-imperviousness ` +
        "after placing the Annual NLCD raster in data/nlcd/.",
    );
  }
  throw error;
}

const scenarioConfig = JSON.parse(await readFile(configPath, "utf8"));
const imperviousness = parseCsv(imperviousnessContents).rows;
const result = buildDnnScenarioTables({
  calibrationRows: calibration,
  imperviousnessRows: imperviousness,
  scenarioConfig,
});

const outputs = [
  writeOutput(trainingPath, serializeCsv(result.trainingRows, trainingHeaders)),
  writeOutput(reportPath, `${JSON.stringify(result.report, null, 2)}\n`),
];
if (includeResearch) {
  outputs.push(
    writeOutput(threeClassTrainingPath, serializeCsv(result.threeClassTrainingRows, threeClassTrainingHeaders)),
    writeOutput(scenarioPath, serializeCsv(result.scenarioRows, scenarioHeaders)),
  );
}
await Promise.all(outputs);

console.log(
  `Wrote ${result.trainingRows.length} binary rows to ${trainingPath} and ${reportPath}.` +
    (includeResearch
      ? ` Also wrote ${result.threeClassTrainingRows.length} three-class rows to ${threeClassTrainingPath} and ${result.scenarioRows.length} scenario rows to ${scenarioPath}.`
      : ""),
);
