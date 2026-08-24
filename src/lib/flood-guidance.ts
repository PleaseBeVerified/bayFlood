export type FloodZoneInput = {
  zone?: string | null;
  subtype?: string | null;
  sfha?: string | null;
};

export type FloodGuidance = {
  confidence: "Low" | "Medium" | "High";
  explanation: string;
  feedback: string[];
};

export function describeFloodConditions(
  elevationMeters: number,
  floodZone: FloodZoneInput,
  distanceToSfhaMeters: number | null,
  rainfallInches: number | null,
): FloodGuidance {
  const feedback: string[] = [];
  const normalizedZone = floodZone.zone?.toUpperCase().trim() ?? "";

  if (["V", "VE"].includes(normalizedZone)) {
    feedback.push(
      "Coastal flood exposure is significant. Review evacuation routes, storm-surge alerts, flood insurance, and options for raising utilities and valuable belongings.",
    );
  } else if (
    ["A", "AE", "A1-30", "AO", "AH", "AR", "A99"].includes(
      normalizedZone,
    ) || floodZone.sfha === "T"
  ) {
    feedback.push(
      "This location is within a FEMA Special Flood Hazard Area. Review flood insurance requirements and have a professional assess floodproofing options.",
    );
  }

  if (elevationMeters <= 15) {
    feedback.push(
      "Low elevation increases flood exposure. Consider raising electrical systems, appliances, and important belongings above floor level.",
    );
  }

  if (rainfallInches !== null && rainfallInches >= 8) {
    feedback.push(
      "The mapped 100-year storm rainfall is substantial. Keep gutters, storm drains, and drainage paths clear, and consider backflow protection.",
    );
  }

  if (normalizedZone === "D") {
    feedback.push(
      "Flood hazards have not been fully determined in this area. Consider requesting a professional flood-risk assessment.",
    );
  }

  if (feedback.length === 0) {
    feedback.push(
      "Mapped flood exposure appears relatively low. Continue monitoring severe-weather alerts and keep gutters and drains clear.",
    );
  }

  const elevationDescription =
    elevationMeters <= 7
      ? "The property is on very low terrain, which can increase exposure to coastal, tidal, and drainage flooding."
      : elevationMeters <= 30
        ? "The property is on relatively low terrain."
        : elevationMeters >= 120
          ? "The property is on high terrain, which generally reduces broad floodplain exposure."
          : "The property's elevation generally reduces broad floodplain exposure.";
  const proximityDescription =
    floodZone.sfha !== "T" &&
    distanceToSfhaMeters !== null &&
    distanceToSfhaMeters <= 1_000
      ? ` The property is about ${distanceToSfhaMeters.toLocaleString()} meters from a FEMA Special Flood Hazard Area.`
      : "";
  const rainfallDescription =
    rainfallInches !== null
      ? ` NOAA Atlas 14 estimates ${rainfallInches.toFixed(2)} inches for a 100-year, 24-hour storm at this location.`
      : "";
  const zoneDescription =
    normalizedZone === "D"
      ? " FEMA Zone D means flood hazards are possible but have not been determined; it is not a mapped high-risk floodplain."
      : "";

  let availableDataSources = 1;
  if (floodZone.zone || floodZone.sfha) availableDataSources += 1;
  if (rainfallInches !== null) availableDataSources += 1;

  const confidence =
    availableDataSources === 3
      ? "High"
      : availableDataSources === 2
        ? "Medium"
        : "Low";

  return {
    confidence,
    feedback,
    explanation:
      elevationDescription +
      zoneDescription +
      proximityDescription +
      rainfallDescription,
  };
}
