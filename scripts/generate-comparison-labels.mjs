import { readFile, writeFile } from "node:fs/promises";

const comparisonsPath = "data/firststreet-comparisons.csv";
const labelsPath = "data/training-labels.csv";
const comparisonPrefix = "firststreet-";
const requiredCount = 100;
const positiveScoreMinimum = 4;

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
  const lines = contents.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length === 0) return [];

  const headers = parseCsvLine(lines[0]);
  return lines.slice(1).map((line, index) => {
    const values = parseCsvLine(line);
    return {
      ...Object.fromEntries(
        headers.map((header, valueIndex) => [header, values[valueIndex] ?? ""]),
      ),
      rowNumber: index + 2,
    };
  });
}

function serializeCsvValue(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text)
    ? `"${text.replaceAll('"', '""')}"`
    : text;
}

function serializeCsv(rows, headers) {
  return [
    headers.join(","),
    ...rows.map((row) =>
      headers.map((header) => serializeCsvValue(row[header])).join(","),
    ),
  ].join("\n") + "\n";
}

function normalizedAddress(address) {
  return address.trim().toLowerCase().replace(/\s+/g, " ");
}

function validateComparisons(rows) {
  if (rows.length !== requiredCount) {
    throw new Error(
      `${comparisonsPath} must contain exactly ${requiredCount} properties; found ${rows.length}.`,
    );
  }

  const addresses = new Set();
  const coordinates = new Set();
  const scores = new Set();

  for (const row of rows) {
    const address = normalizedAddress(row.address ?? "");
    const latitude = Number(row.lat);
    const longitude = Number(row.lng);
    const score = Number(row.flood_factor);

    if (!address) {
      throw new Error(`Row ${row.rowNumber}: address is required.`);
    }
    if (addresses.has(address)) {
      throw new Error(`Row ${row.rowNumber}: duplicate address "${row.address}".`);
    }
    addresses.add(address);

    if (!Number.isFinite(latitude) || latitude < -90 || latitude > 90) {
      throw new Error(`Row ${row.rowNumber}: lat must be a valid latitude.`);
    }
    if (!Number.isFinite(longitude) || longitude < -180 || longitude > 180) {
      throw new Error(`Row ${row.rowNumber}: lng must be a valid longitude.`);
    }

    const coordinateKey = `${latitude.toFixed(5)},${longitude.toFixed(5)}`;
    if (coordinates.has(coordinateKey)) {
      throw new Error(`Row ${row.rowNumber}: duplicate property coordinates.`);
    }
    coordinates.add(coordinateKey);

    if (!Number.isInteger(score) || score < 1 || score > 10) {
      throw new Error(`Row ${row.rowNumber}: flood_factor must be an integer from 1 to 10.`);
    }
    scores.add(score);

    if (!/^https:\/\/(www\.)?firststreet\.org\//i.test(row.source_url ?? "")) {
      throw new Error(`Row ${row.rowNumber}: source_url must be a First Street URL.`);
    }
  }

  const missingScores = Array.from(
    { length: 10 },
    (_, index) => index + 1,
  ).filter((score) => !scores.has(score));

  if (missingScores.length > 0) {
    throw new Error(
      `Include at least one property for every Flood Factor score. Missing: ${missingScores.join(", ")}.`,
    );
  }
}

async function main() {
  const comparisons = parseCsv(await readFile(comparisonsPath, "utf8"));
  validateComparisons(comparisons);

  const existingRows = parseCsv(await readFile(labelsPath, "utf8"));
  const preservedRows = existingRows.filter(
    (row) => !row.id?.startsWith(comparisonPrefix),
  );
  const importedRows = comparisons.map((row, index) => {
    const score = Number(row.flood_factor);
    return {
      id: `${comparisonPrefix}${String(index + 1).padStart(3, "0")}`,
      lat: Number(row.lat),
      lng: Number(row.lng),
      flooded: score >= positiveScoreMinimum ? 1 : 0,
      source: "firststreet_flood_factor",
      address: row.address.trim(),
      reference_score: score,
      reference_label: row.risk_label?.trim() ?? "",
      source_url: row.source_url.trim(),
      notes:
        `First Street Flood Factor ${score}; binary comparison label uses ` +
        `${positiveScoreMinimum}-10 as positive`,
    };
  });
  const headers = [
    "id",
    "lat",
    "lng",
    "flooded",
    "source",
    "address",
    "reference_score",
    "reference_label",
    "source_url",
    "notes",
  ];

  await writeFile(
    labelsPath,
    serializeCsv([...preservedRows, ...importedRows], headers),
  );

  const scoreCounts = Object.fromEntries(
    Array.from({ length: 10 }, (_, index) => {
      const score = index + 1;
      return [
        score,
        comparisons.filter((row) => Number(row.flood_factor) === score).length,
      ];
    }),
  );

  console.log(`Imported ${importedRows.length} First Street comparisons.`);
  console.log("Flood Factor distribution:", scoreCounts);
  console.log(`Updated ${labelsPath}. Next run: npm run training:table`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
