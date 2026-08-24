const requiredHeaders = [
  "property_id",
  "parcel_id",
  "building_id",
  "address",
  "lat",
  "lng",
  "county",
  "observation_start",
  "observation_end",
  "event_date",
  "event_source",
  "non_event_evidence",
  "verification_status",
  "coordinate_precision_m",
  "source_url",
  "notes",
];

const verificationStatuses = new Set(["pending", "verified", "excluded"]);

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

export function parseCsv(contents) {
  const lines = contents.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length === 0) return { headers: [], rows: [] };

  const headers = parseCsvLine(lines[0]);
  const rows = lines.slice(1).map((line, index) => {
    const values = parseCsvLine(line);
    return {
      ...Object.fromEntries(
        headers.map((header, valueIndex) => [header, values[valueIndex] ?? ""]),
      ),
      rowNumber: index + 2,
    };
  });

  return { headers, rows };
}

function escapeCsv(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text)
    ? `"${text.replaceAll('"', '""')}"`
    : text;
}

export function serializeCsv(rows, headers) {
  return `${[
    headers.join(","),
    ...rows.map((row) =>
      headers.map((header) => escapeCsv(row[header])).join(","),
    ),
  ].join("\n")}\n`;
}

function parseIsoDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value)) return null;
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isNaN(date.valueOf()) || date.toISOString().slice(0, 10) !== value
    ? null
    : date;
}

function parseFiniteNumber(value) {
  if (String(value).trim() === "") return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function validateObservationTable(contents) {
  const { headers, rows } = parseCsv(contents);
  const errors = [];
  const warnings = [];
  const missingHeaders = requiredHeaders.filter(
    (header) => !headers.includes(header),
  );

  if (missingHeaders.length > 0) {
    errors.push(`Missing required columns: ${missingHeaders.join(", ")}.`);
    return { headers, rows: [], verifiedRows: [], errors, warnings };
  }

  const propertyIds = new Set();
  const normalizedRows = [];

  for (const row of rows) {
    const prefix = `Row ${row.rowNumber}`;
    const status = row.verification_status.trim().toLowerCase();
    const latitude = parseFiniteNumber(row.lat);
    const longitude = parseFiniteNumber(row.lng);
    const coordinatePrecision = parseFiniteNumber(row.coordinate_precision_m);

    if (!row.property_id.trim()) errors.push(`${prefix}: property_id is required.`);
    if (propertyIds.has(row.property_id.trim())) {
      errors.push(`${prefix}: duplicate property_id ${row.property_id}.`);
    }
    propertyIds.add(row.property_id.trim());

    if (!row.address.trim()) errors.push(`${prefix}: address is required.`);
    if (!row.county.trim()) errors.push(`${prefix}: county is required.`);
    if (latitude === null || latitude < -90 || latitude > 90) {
      errors.push(`${prefix}: lat must be a valid latitude.`);
    }
    if (longitude === null || longitude < -180 || longitude > 180) {
      errors.push(`${prefix}: lng must be a valid longitude.`);
    }
    if (!verificationStatuses.has(status)) {
      errors.push(
        `${prefix}: verification_status must be pending, verified, or excluded.`,
      );
    }

    if (status === "verified") {
      const observationStart = parseIsoDate(row.observation_start);
      const observationEnd = parseIsoDate(row.observation_end);
      const eventDate = row.event_date ? parseIsoDate(row.event_date) : null;

      if (!observationStart) {
        errors.push(`${prefix}: verified rows require observation_start (YYYY-MM-DD).`);
      }
      if (!observationEnd) {
        errors.push(`${prefix}: verified rows require observation_end (YYYY-MM-DD).`);
      }
      if (observationStart && observationEnd && observationEnd < observationStart) {
        errors.push(`${prefix}: observation_end is before observation_start.`);
      }
      if (row.event_date && !eventDate) {
        errors.push(`${prefix}: event_date must use YYYY-MM-DD.`);
      }
      if (
        eventDate &&
        observationStart &&
        observationEnd &&
        (eventDate < observationStart || eventDate > observationEnd)
      ) {
        errors.push(`${prefix}: event_date is outside the observation period.`);
      }
      if (eventDate && !row.event_source.trim()) {
        errors.push(`${prefix}: event_source is required when event_date is set.`);
      }
      if (!eventDate && !row.non_event_evidence.trim()) {
        errors.push(
          `${prefix}: verified censored rows require non_event_evidence.`,
        );
      }
      if (!row.source_url.trim()) {
        errors.push(`${prefix}: verified rows require source_url.`);
      }
      if (
        coordinatePrecision === null ||
        coordinatePrecision <= 0 ||
        coordinatePrecision > 30
      ) {
        errors.push(
          `${prefix}: verified parcel rows require coordinate_precision_m from 0 to 30.`,
        );
      }
      if (!row.parcel_id.trim() && !row.building_id.trim()) {
        errors.push(
          `${prefix}: verified rows require parcel_id or building_id.`,
        );
      }
    } else if (status === "pending") {
      warnings.push(
        `${prefix}: pending row excluded from hazard training until its observation evidence is verified.`,
      );
    }

    normalizedRows.push({
      ...row,
      verification_status: status,
      lat: latitude,
      lng: longitude,
      coordinate_precision_m: coordinatePrecision,
    });
  }

  return {
    headers,
    rows: normalizedRows,
    verifiedRows: normalizedRows.filter(
      (row) => row.verification_status === "verified",
    ),
    errors,
    warnings,
  };
}

export function expandHazardPeriods(row) {
  const observationStart = new Date(`${row.observation_start}T00:00:00Z`);
  const observationEnd = new Date(`${row.observation_end}T00:00:00Z`);
  const eventDate = row.event_date
    ? new Date(`${row.event_date}T00:00:00Z`)
    : null;
  const finalDate = eventDate ?? observationEnd;
  const periods = [];

  for (
    let year = observationStart.getUTCFullYear(), timeIndex = 1;
    year <= finalDate.getUTCFullYear();
    year += 1, timeIndex += 1
  ) {
    const yearStart = new Date(Date.UTC(year, 0, 1));
    const yearEnd = new Date(Date.UTC(year, 11, 31));
    const periodStart = observationStart > yearStart ? observationStart : yearStart;
    const periodEnd = finalDate < yearEnd ? finalDate : yearEnd;
    const exposureDays =
      Math.floor((periodEnd.valueOf() - periodStart.valueOf()) / 86_400_000) + 1;
    const daysInYear =
      (Date.UTC(year + 1, 0, 1) - Date.UTC(year, 0, 1)) / 86_400_000;

    periods.push({
      property_id: row.property_id,
      calendar_year: year,
      time_index: timeIndex,
      period_start: periodStart.toISOString().slice(0, 10),
      period_end: periodEnd.toISOString().slice(0, 10),
      exposure_days: exposureDays,
      exposure_fraction: Math.round((exposureDays / daysInYear) * 1_000_000) / 1_000_000,
      event: eventDate?.getUTCFullYear() === year ? 1 : 0,
    });
  }

  return periods;
}

export { requiredHeaders };
