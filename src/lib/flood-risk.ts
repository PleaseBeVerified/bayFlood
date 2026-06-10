export type FloodZoneInput = {
  zone?: string | null;
  subtype?: string | null;
  sfha?: string | null;
};

export type FloodRisk = {
  score: number;
  level: "Low" | "Moderate" | "High" | "Very high";
  color: string;
  elevationContribution: number;
  femaContribution: number;
  nearbyFloodplainContribution: number;
  rainfallContribution: number;
  explanation: string;
};

function elevationRisk(elevationMeters: number) {
  if (elevationMeters <= 0) return 42;
  if (elevationMeters <= 3) return 38;
  if (elevationMeters <= 7) return 32;
  if (elevationMeters <= 15) return 23;
  if (elevationMeters <= 30) return 14;
  if (elevationMeters <= 60) return 7;
  if (elevationMeters <= 120) return 3;
  return 0;
}

function femaRisk({ zone, subtype, sfha }: FloodZoneInput) {
  const normalizedZone = zone?.toUpperCase().trim() ?? "";
  const normalizedSubtype = subtype?.toUpperCase() ?? "";

  if (["V", "VE"].includes(normalizedZone)) return 58;
  if (
    ["A", "AE", "A1-30", "AO", "AH", "AR", "A99"].includes(normalizedZone) ||
    sfha === "T"
  ) {
    return 52;
  }
  if (
    normalizedSubtype.includes("0.2 PCT") ||
    normalizedSubtype.includes("MODERATE FLOOD HAZARD")
  ) {
    return 31;
  }
  if (normalizedZone === "X") return 10;
  if (normalizedZone === "D") return 8;
  return 15;
}

function rainfallRisk(
  rainfallInches: number | null,
  elevationMeters: number,
) {
  if (rainfallInches === null) return 0;

  const rainfallSeverity =
    rainfallInches <= 4
      ? 2
      : rainfallInches <= 5
        ? 4
        : rainfallInches <= 6.5
          ? 6
          : rainfallInches <= 8
            ? 8
            : rainfallInches <= 10
              ? 10
              : 12;
  const terrainMultiplier =
    elevationMeters <= 15
      ? 1
      : elevationMeters <= 30
        ? 0.85
        : elevationMeters <= 60
          ? 0.65
          : elevationMeters <= 120
            ? 0.45
            : 0.25;

  return Math.round(rainfallSeverity * terrainMultiplier);
}

export function calculateFloodRisk(
  elevationMeters: number,
  floodZone: FloodZoneInput,
  distanceToSfhaMeters: number | null,
  rainfallInches: number | null,
): FloodRisk {
  const elevationContribution = elevationRisk(elevationMeters);
  const femaContribution = femaRisk(floodZone);
  const rainfallContribution = rainfallRisk(rainfallInches, elevationMeters);
  let nearbyFloodplainContribution = 0;

  if (floodZone.sfha !== "T" && distanceToSfhaMeters !== null) {
    const lowTerrainMultiplier =
      elevationMeters <= 15 ? 1 : elevationMeters <= 30 ? 0.75 : 0.4;
    const proximityRisk =
      distanceToSfhaMeters <= 75
        ? 30
        : distanceToSfhaMeters <= 200
          ? 23
          : distanceToSfhaMeters <= 500
            ? 14
            : distanceToSfhaMeters <= 1_000
              ? 6
              : 0;
    nearbyFloodplainContribution = Math.round(proximityRisk * lowTerrainMultiplier);
  }

  const score = Math.min(
    100,
    elevationContribution +
      femaContribution +
      nearbyFloodplainContribution +
      rainfallContribution,
  );

  let level: FloodRisk["level"] = "Low";
  let color = "#15803d";

  if (score >= 75) {
    level = "Very high";
    color = "#be123c";
  } else if (score >= 50) {
    level = "High";
    color = "#dc2626";
  } else if (score >= 28) {
    level = "Moderate";
    color = "#d97706";
  }

  const elevationDescription =
    elevationMeters <= 7
      ? "Very low terrain increases exposure to coastal, tidal, and drainage flooding."
      : elevationMeters <= 30
        ? "Low terrain adds some flood exposure."
        : elevationMeters >= 120
          ? "High terrain substantially reduces broad floodplain exposure."
          : "Elevation provides some protection from broad floodplain flooding.";
  const proximityDescription =
    nearbyFloodplainContribution > 0
      ? ` The property is about ${distanceToSfhaMeters?.toLocaleString()} meters from a FEMA Special Flood Hazard Area, which adds creek or floodplain overflow exposure.`
      : "";
  const rainfallDescription =
    rainfallInches !== null
      ? ` NOAA Atlas 14 estimates ${rainfallInches.toFixed(2)} inches for a 100-year, 24-hour storm at this location.`
      : "";
  const zoneDescription =
    floodZone.zone?.toUpperCase().trim() === "D"
      ? " FEMA Zone D indicates that flood hazards are possible but have not been determined; it is not a mapped high-risk floodplain."
      : "";

  return {
    score,
    level,
    color,
    elevationContribution,
    femaContribution,
    nearbyFloodplainContribution,
    rainfallContribution,
    explanation:
      elevationDescription +
      zoneDescription +
      proximityDescription +
      rainfallDescription,
  };
}
