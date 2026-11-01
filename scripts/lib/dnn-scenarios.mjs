export const imperviousnessHeaders = [
  "location_id",
  "lat",
  "lng",
  "county",
  "current_imperviousness_pct",
  "source",
];

export const trainingHeaders = [
  "location_id",
  "county",
  "lat",
  "lng",
  "spatial_partition",
  "elevation_m",
  "current_imperviousness_pct",
  "current_100yr_24h_rainfall_mm",
  "nearest_water_available",
  "nearest_water_distance_m",
  "flood_risk_label",
  "label_source",
];

export const threeClassTrainingHeaders = trainingHeaders.map((header) =>
  header === "flood_risk_label" ? "mapped_hazard_class" : header,
);

export const scenarioHeaders = [
  "location_id",
  "county",
  "lat",
  "lng",
  "source_spatial_partition",
  "scenario_id",
  "scenario_year",
  "elevation_m",
  "projected_imperviousness_pct",
  "projected_100yr_24h_rainfall_mm",
  "nearest_water_available",
  "nearest_water_distance_m",
  "imperviousness_increase_percentage_points",
  "rainfall_increase_percent",
  "scenario_status",
];

const hazardClasses = new Set(["one_percent", "point_two_percent", "minimal"]);

function finiteNumber(value, field, locationId) {
  if (String(value ?? "").trim() === "") {
    throw new Error(`${locationId}: ${field} is required.`);
  }
  const number = Number(value);
  if (!Number.isFinite(number)) {
    throw new Error(`${locationId}: ${field} must be a finite number.`);
  }
  return number;
}

function rounded(value, digits = 3) {
  const scale = 10 ** digits;
  return Math.round(value * scale) / scale;
}

function stringHash(value) {
  let hash = 2_166_136_261;
  for (const character of value) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16_777_619);
  }
  return hash >>> 0;
}

export function spatialPartition(longitude, latitude) {
  const latitudeTile = Math.floor(latitude / 0.05);
  const longitudeTile = Math.floor(longitude / 0.05);
  const bucket = stringHash(`${latitudeTile},${longitudeTile}`) % 20;
  if (bucket < 14) return "training";
  if (bucket < 17) return "validation";
  return "test";
}

function normalizedCalibrationRows(rows) {
  const ids = new Set();
  return rows.map((row, index) => {
    const locationId = String(row.sample_id ?? "").trim();
    if (!locationId) throw new Error(`Calibration row ${index + 2}: sample_id is required.`);
    if (ids.has(locationId)) throw new Error(`Duplicate calibration sample_id: ${locationId}.`);
    ids.add(locationId);

    const latitude = finiteNumber(row.lat, "lat", locationId);
    const longitude = finiteNumber(row.lng, "lng", locationId);
    const elevation = finiteNumber(row.elevation_m, "elevation_m", locationId);
    const rainfallInches = finiteNumber(
      row.rainfall_100yr_24h_in,
      "rainfall_100yr_24h_in",
      locationId,
    );
    const rawWaterDistance = String(row.nearest_water_distance_m ?? "").trim();
    const waterDistance = rawWaterDistance
      ? finiteNumber(rawWaterDistance, "nearest_water_distance_m", locationId)
      : null;
    const hazardClass = String(row.mapped_hazard_class ?? "").trim();

    if (latitude < -90 || latitude > 90) {
      throw new Error(`${locationId}: lat is outside -90 to 90.`);
    }
    if (longitude < -180 || longitude > 180) {
      throw new Error(`${locationId}: lng is outside -180 to 180.`);
    }
    if (rainfallInches < 0) {
      throw new Error(`${locationId}: rainfall cannot be negative.`);
    }
    if (waterDistance !== null && waterDistance < 0) {
      throw new Error(`${locationId}: nearest-water distance cannot be negative.`);
    }
    if (!hazardClasses.has(hazardClass)) {
      throw new Error(`${locationId}: unsupported mapped_hazard_class ${hazardClass}.`);
    }

    return {
      locationId,
      county: String(row.county ?? "").trim(),
      latitude,
      longitude,
      elevation,
      rainfallMillimeters: rainfallInches * 25.4,
      waterDistance,
      hazardClass,
      labelSource: String(row.label_source ?? "").trim(),
      partition: spatialPartition(longitude, latitude),
    };
  });
}

export function buildImperviousnessTemplate(calibrationRows) {
  const ids = new Set();
  return calibrationRows.map((row, index) => {
    const locationId = String(row.sample_id ?? "").trim();
    if (!locationId) throw new Error(`Calibration row ${index + 2}: sample_id is required.`);
    if (ids.has(locationId)) throw new Error(`Duplicate calibration sample_id: ${locationId}.`);
    ids.add(locationId);
    const latitude = finiteNumber(row.lat, "lat", locationId);
    const longitude = finiteNumber(row.lng, "lng", locationId);
    return {
      location_id: locationId,
      lat: latitude,
      lng: longitude,
      county: String(row.county ?? "").trim(),
      current_imperviousness_pct: "",
      source: "",
    };
  });
}

export function validateScenarioConfig(config) {
  if (!config || typeof config !== "object") {
    throw new Error("Scenario configuration must be a JSON object.");
  }
  const scenarioYear = Number(config.scenarioYear);
  const scenarioStatus = String(config.scenarioStatus ?? "").trim();
  if (!Number.isInteger(scenarioYear) || scenarioYear < 2026) {
    throw new Error("scenarioYear must be an integer of 2026 or later.");
  }
  if (!scenarioStatus) {
    throw new Error("scenarioStatus is required.");
  }
  if (!Array.isArray(config.scenarios) || config.scenarios.length === 0) {
    throw new Error("At least one projection scenario is required.");
  }

  const ids = new Set();
  const scenarios = config.scenarios.map((scenario, index) => {
    const id = String(scenario.id ?? "").trim();
    const imperviousnessIncrease = Number(
      scenario.imperviousnessIncreasePercentagePoints,
    );
    const rainfallIncrease = Number(scenario.rainfallIncreasePercent);
    if (!id) throw new Error(`Scenario ${index + 1}: id is required.`);
    if (ids.has(id)) throw new Error(`Duplicate scenario id: ${id}.`);
    ids.add(id);
    if (!Number.isFinite(imperviousnessIncrease)) {
      throw new Error(`${id}: imperviousness increase must be finite.`);
    }
    if (!Number.isFinite(rainfallIncrease) || rainfallIncrease <= -100) {
      throw new Error(`${id}: rainfall increase must be finite and greater than -100%.`);
    }
    return {
      id,
      imperviousnessIncrease,
      rainfallIncrease,
    };
  });

  return { scenarioYear, scenarioStatus, scenarios };
}

function imperviousnessByLocation(rows, knownLocations) {
  const result = new Map();
  const errors = [];
  for (const row of rows) {
    const locationId = String(row.location_id ?? "").trim();
    if (!locationId) {
      errors.push(`Imperviousness row ${row.rowNumber ?? "?"}: location_id is required.`);
      continue;
    }
    if (!knownLocations.has(locationId)) {
      errors.push(`${locationId}: no matching calibration location.`);
      continue;
    }
    if (result.has(locationId)) {
      errors.push(`${locationId}: duplicate imperviousness row.`);
      continue;
    }
    const rawValue = String(row.current_imperviousness_pct ?? "").trim();
    const source = String(row.source ?? "").trim();
    const value = Number(rawValue);
    if (!rawValue || !Number.isFinite(value) || value < 0 || value > 100) {
      errors.push(`${locationId}: current_imperviousness_pct must be from 0 to 100.`);
      continue;
    }
    if (!source) {
      errors.push(`${locationId}: an imperviousness source is required.`);
      continue;
    }
    result.set(locationId, value);
  }

  const missing = [...knownLocations].filter((locationId) => !result.has(locationId));
  if (missing.length > 0) {
    errors.push(
      `${missing.length} calibration locations are missing sourced imperviousness values` +
        ` (first: ${missing.slice(0, 5).join(", ")}).`,
    );
  }
  if (errors.length > 0) {
    throw new Error(errors.slice(0, 20).join("\n"));
  }
  return result;
}

function incrementCount(record, key) {
  record[key] = (record[key] ?? 0) + 1;
}

export function buildDnnScenarioTables({
  calibrationRows,
  imperviousnessRows,
  scenarioConfig,
}) {
  const calibration = normalizedCalibrationRows(calibrationRows);
  const config = validateScenarioConfig(scenarioConfig);
  const knownLocations = new Set(calibration.map((row) => row.locationId));
  const imperviousness = imperviousnessByLocation(
    imperviousnessRows,
    knownLocations,
  );
  const trainingRows = [];
  const threeClassTrainingRows = [];
  const scenarioRows = [];
  const labels = {};
  const threeClassLabels = {};
  const partitions = {};
  const threeClassPartitions = {};

  for (const row of calibration) {
    const currentImperviousness = imperviousness.get(row.locationId);
    const currentInputs = {
      location_id: row.locationId,
      county: row.county,
      lat: row.latitude,
      lng: row.longitude,
      spatial_partition: row.partition,
      elevation_m: row.elevation,
      current_imperviousness_pct: currentImperviousness,
      current_100yr_24h_rainfall_mm: rounded(row.rainfallMillimeters),
      nearest_water_available: row.waterDistance === null ? 0 : 1,
      nearest_water_distance_m: row.waterDistance ?? "",
      label_source: row.labelSource,
    };
    threeClassTrainingRows.push({
      ...currentInputs,
      mapped_hazard_class: row.hazardClass,
    });
    incrementCount(threeClassLabels, row.hazardClass);
    incrementCount(threeClassPartitions, row.partition);

    if (row.hazardClass !== "point_two_percent") {
      const label = row.hazardClass === "one_percent" ? 1 : 0;
      trainingRows.push({
        ...currentInputs,
        flood_risk_label: label,
      });
      incrementCount(labels, String(label));
      incrementCount(partitions, row.partition);
    }

    for (const scenario of config.scenarios) {
      scenarioRows.push({
        location_id: row.locationId,
        county: row.county,
        lat: row.latitude,
        lng: row.longitude,
        source_spatial_partition: row.partition,
        scenario_id: scenario.id,
        scenario_year: config.scenarioYear,
        elevation_m: row.elevation,
        projected_imperviousness_pct: rounded(
          Math.min(100, Math.max(0, currentImperviousness + scenario.imperviousnessIncrease)),
        ),
        projected_100yr_24h_rainfall_mm: rounded(
          row.rainfallMillimeters * (1 + scenario.rainfallIncrease / 100),
        ),
        nearest_water_available: row.waterDistance === null ? 0 : 1,
        nearest_water_distance_m: row.waterDistance ?? "",
        imperviousness_increase_percentage_points:
          scenario.imperviousnessIncrease,
        rainfall_increase_percent: scenario.rainfallIncrease,
        scenario_status: config.scenarioStatus,
      });
    }
  }

  return {
    trainingRows,
    threeClassTrainingRows,
    scenarioRows,
    report: {
      schemaVersion: 1,
      trainingTarget:
        "Binary current FEMA-equivalent class: one_percent=1, minimal=0; point_two_percent excluded.",
      projectionInterpretation:
        "Scenario sensitivity inputs only; not validated 2056 forecasts or future labels.",
      inputLocations: calibration.length,
      trainingRows: trainingRows.length,
      threeClassTrainingRows: threeClassTrainingRows.length,
      excludedPointTwoPercentRows: calibration.length - trainingRows.length,
      scenarioRows: scenarioRows.length,
      scenarioYear: config.scenarioYear,
      scenarioIds: config.scenarios.map((scenario) => scenario.id),
      trainingLabelCounts: labels,
      threeClassTrainingLabelCounts: threeClassLabels,
      trainingPartitionCounts: partitions,
      threeClassTrainingPartitionCounts: threeClassPartitions,
    },
  };
}
