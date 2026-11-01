import assert from "node:assert/strict";
import test from "node:test";
import {
  buildDnnScenarioTables,
  buildImperviousnessTemplate,
  scenarioHeaders,
  threeClassTrainingHeaders,
} from "../scripts/lib/dnn-scenarios.mjs";

const calibrationRows = [
  {
    sample_id: "high-1",
    county: "Alameda",
    lat: "37.585671",
    lng: "-121.943316",
    mapped_hazard_class: "one_percent",
    elevation_m: "12",
    rainfall_100yr_24h_in: "5",
    nearest_water_distance_m: "120",
    label_source: "FEMA NFHL",
  },
  {
    sample_id: "middle-1",
    county: "Alameda",
    lat: "37.61",
    lng: "-121.91",
    mapped_hazard_class: "point_two_percent",
    elevation_m: "20",
    rainfall_100yr_24h_in: "4",
    nearest_water_distance_m: "",
    label_source: "FEMA NFHL",
  },
  {
    sample_id: "low-1",
    county: "Santa Clara",
    lat: "37.3",
    lng: "-121.8",
    mapped_hazard_class: "minimal",
    elevation_m: "45",
    rainfall_100yr_24h_in: "3",
    nearest_water_distance_m: "900",
    label_source: "FEMA NFHL",
  },
];

const imperviousnessRows = [
  { location_id: "high-1", current_imperviousness_pct: "85", source: "test" },
  { location_id: "middle-1", current_imperviousness_pct: "50", source: "test" },
  { location_id: "low-1", current_imperviousness_pct: "40", source: "test" },
];

const scenarioConfig = {
  scenarioYear: 2056,
  scenarioStatus: "illustrative_only",
  scenarios: [
    {
      id: "medium_change",
      imperviousnessIncreasePercentagePoints: 20,
      rainfallIncreasePercent: 15,
    },
  ],
};

test("creates a keyed blank imperviousness template", () => {
  const template = buildImperviousnessTemplate(calibrationRows);
  assert.equal(template.length, 3);
  assert.equal(template[0].location_id, "high-1");
  assert.equal(template[0].current_imperviousness_pct, "");
});

test("builds current binary training data and excludes the middle class", () => {
  const result = buildDnnScenarioTables({
    calibrationRows,
    imperviousnessRows,
    scenarioConfig,
  });
  assert.equal(result.trainingRows.length, 2);
  assert.deepEqual(
    result.trainingRows.map((row) => row.flood_risk_label),
    [1, 0],
  );
  assert.equal(result.trainingRows[0].current_100yr_24h_rainfall_mm, 127);
  assert.equal(result.report.excludedPointTwoPercentRows, 1);
  assert.equal(result.threeClassTrainingRows.length, 3);
  assert.deepEqual(
    result.threeClassTrainingRows.map((row) => row.mapped_hazard_class),
    ["one_percent", "point_two_percent", "minimal"],
  );
  assert.equal(result.report.threeClassTrainingLabelCounts.point_two_percent, 1);
  assert.equal(threeClassTrainingHeaders.includes("mapped_hazard_class"), true);
});

test("builds unlabeled scenarios and caps projected imperviousness", () => {
  const result = buildDnnScenarioTables({
    calibrationRows,
    imperviousnessRows,
    scenarioConfig,
  });
  assert.equal(result.scenarioRows.length, 3);
  assert.equal(result.scenarioRows[0].projected_imperviousness_pct, 100);
  assert.equal(result.scenarioRows[0].projected_100yr_24h_rainfall_mm, 146.05);
  assert.equal("flood_risk_label" in result.scenarioRows[0], false);
  assert.equal(scenarioHeaders.includes("flood_risk_label"), false);
  assert.equal(result.scenarioRows[1].nearest_water_available, 0);
  assert.equal(result.scenarioRows[1].nearest_water_distance_m, "");
});

test("refuses incomplete or unsourced imperviousness data", () => {
  assert.throws(
    () =>
      buildDnnScenarioTables({
        calibrationRows,
        imperviousnessRows: imperviousnessRows.slice(0, 2),
        scenarioConfig,
      }),
    /missing sourced imperviousness values/,
  );
  assert.throws(
    () =>
      buildDnnScenarioTables({
        calibrationRows,
        imperviousnessRows: imperviousnessRows.map((row) =>
          row.location_id === "low-1" ? { ...row, source: "" } : row,
        ),
        scenarioConfig,
      }),
    /imperviousness source is required/,
  );
});
