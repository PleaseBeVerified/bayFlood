import { writeFile } from "node:fs/promises";

const outputPath = "data/firststreet-property-candidates.csv";
const overpassUrls = [
  "https://overpass-api.de/api/interpreter",
  "https://overpass.kumi.systems/api/interpreter",
];
const samplesPerArea = 5;
const minimumSpacingMeters = 750;

const areas = [
  ["Alameda", "Oakland", 37.8044, -122.2712],
  ["Alameda", "Fremont", 37.5485, -121.9886],
  ["Contra Costa", "Richmond", 37.9358, -122.3477],
  ["Contra Costa", "Walnut Creek", 37.9101, -122.0652],
  ["Marin", "San Rafael", 37.9735, -122.5311],
  ["Marin", "Novato", 38.1074, -122.5697],
  ["Napa", "Napa", 38.2975, -122.2869],
  ["Napa", "American Canyon", 38.1749, -122.2608],
  ["San Francisco", "San Francisco West", 37.763, -122.476],
  ["San Francisco", "San Francisco East", 37.756, -122.414],
  ["San Mateo", "Daly City", 37.6879, -122.4702],
  ["San Mateo", "Redwood City", 37.4852, -122.2364],
  ["Santa Clara", "San Jose", 37.3382, -121.8863],
  ["Santa Clara", "Palo Alto", 37.4419, -122.143],
  ["Solano", "Vallejo", 38.1041, -122.2566],
  ["Solano", "Fairfield", 38.2494, -122.04],
  ["Sonoma", "Santa Rosa", 38.4404, -122.7141],
  ["Sonoma", "Petaluma", 38.2324, -122.6367],
  ["Coastal", "Half Moon Bay", 37.4636, -122.4286],
  ["Bay Shore", "East Palo Alto", 37.4688, -122.1411],
];

function escapeCsv(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text)
    ? `"${text.replaceAll('"', '""')}"`
    : text;
}

function distanceMeters(a, b) {
  const latitudeScale = 111_320;
  const latitude = ((a.lat + b.lat) / 2) * Math.PI / 180;
  const longitudeScale = latitudeScale * Math.cos(latitude);
  return Math.hypot(
    (a.lng - b.lng) * longitudeScale,
    (a.lat - b.lat) * latitudeScale,
  );
}

function addressFromTags(tags) {
  const street = tags["addr:street"];
  const number = tags["addr:housenumber"];
  const city = tags["addr:city"];
  const postcode = tags["addr:postcode"];
  if (!street || !number) return null;

  return [
    `${number} ${street}`,
    city,
    "CA",
    postcode,
  ].filter(Boolean).join(", ");
}

async function fetchAreaCandidates(area) {
  const [county, areaName, latitude, longitude] = area;
  const query = `
    [out:json][timeout:60];
    (
      nwr(around:9000,${latitude},${longitude})
        ["addr:housenumber"]["addr:street"]
        ["building"~"^(house|residential|apartments|detached|semidetached_house|terrace)$"];
    );
    out center tags;
  `;
  let lastStatus = "unknown";
  for (let attempt = 0; attempt < 6; attempt += 1) {
    const endpoint = overpassUrls[attempt % overpassUrls.length];
    const response = await fetch(endpoint, {
      method: "POST",
      signal: AbortSignal.timeout(45_000),
      headers: {
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": "BayFlood educational training-data sampler",
      },
      body: new URLSearchParams({ data: query }),
    });

    if (response.ok) {
      const data = await response.json();
      return data.elements.flatMap((element) => {
        const lat = element.lat ?? element.center?.lat;
        const lng = element.lon ?? element.center?.lon;
        const address = addressFromTags(element.tags ?? {});
        if (!address || !Number.isFinite(lat) || !Number.isFinite(lng)) return [];
        return [{ county, area: areaName, address, lat, lng }];
      });
    }

    lastStatus = response.status;
    if (response.status !== 429 && response.status < 500) break;
    await new Promise((resolve) => setTimeout(resolve, 5_000 * (attempt + 1)));
  }
  throw new Error(`${areaName}: Overpass returned HTTP ${lastStatus}.`);
}

function selectSpaced(
  candidates,
  count,
  minimumSpacing = minimumSpacingMeters,
  excludedAddresses = new Set(),
  excludedCoordinates = new Set(),
) {
  const sorted = [...candidates].sort((a, b) =>
    a.address.localeCompare(b.address),
  );
  const selected = [];

  for (const candidate of sorted) {
    const addressKey = candidate.address.toLowerCase();
    const coordinateKey = `${candidate.lat.toFixed(5)},${candidate.lng.toFixed(5)}`;
    if (
      excludedAddresses.has(addressKey) ||
      excludedCoordinates.has(coordinateKey)
    ) {
      continue;
    }
    if (
      selected.every(
        (existing) => distanceMeters(candidate, existing) >= minimumSpacing,
      )
    ) {
      selected.push(candidate);
      if (selected.length === count) break;
    }
  }

  return selected;
}

async function main() {
  const rows = [];
  const selectedAddresses = new Set();
  const selectedCoordinates = new Set();

  for (const area of areas) {
    const candidates = await fetchAreaCandidates(area);
    let selected = [];
    let spacing = minimumSpacingMeters;
    for (const candidateSpacing of [750, 600, 450, 300]) {
      const attempt = selectSpaced(
        candidates,
        samplesPerArea,
        candidateSpacing,
        selectedAddresses,
        selectedCoordinates,
      );
      if (attempt.length > selected.length) {
        selected = attempt;
        spacing = candidateSpacing;
      }
      if (selected.length === samplesPerArea) break;
    }
    if (selected.length < samplesPerArea) {
      throw new Error(
        `${area[1]} only produced ${selected.length}/${samplesPerArea} spaced residential addresses.`,
      );
    }
    rows.push(...selected);
    for (const row of selected) {
      selectedAddresses.add(row.address.toLowerCase());
      selectedCoordinates.add(`${row.lat.toFixed(5)},${row.lng.toFixed(5)}`);
    }
    console.log(
      `Selected ${selected.length} addresses around ${area[1]} (${spacing} m spacing).`,
    );
    await new Promise((resolve) => setTimeout(resolve, 2_000));
  }

  const headers = [
    "address",
    "lat",
    "lng",
    "county",
    "sample_area",
    "flood_factor",
    "risk_label",
    "source_url",
  ];
  const csv = [
    headers.join(","),
    ...rows.map((row) =>
      [
        row.address,
        row.lat.toFixed(6),
        row.lng.toFixed(6),
        row.county,
        row.area,
        "",
        "",
        "",
      ].map(escapeCsv).join(","),
    ),
  ].join("\n") + "\n";

  await writeFile(outputPath, csv);
  console.log(`Wrote ${rows.length} candidates to ${outputPath}.`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
