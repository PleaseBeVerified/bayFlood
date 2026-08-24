import type { FloodZoneInput } from "@/lib/flood-guidance";

const ONE_PERCENT_ANNUAL_CHANCE = 0.01;
const POINT_TWO_PERCENT_ANNUAL_CHANCE = 0.002;

export type HundredYearFloodBenchmark = {
  horizonYears: 100;
  status: "available" | "not_determined";
  percent: number | null;
  display: string;
  annualChancePercent: number | null;
  event: "one_percent_annual_chance" | "point_two_percent_annual_chance" | null;
  basis: string;
  caveat: string;
};

export function probabilityOverYears(
  annualChance: number,
  years: number,
): number {
  if (annualChance < 0 || annualChance > 1) {
    throw new RangeError("annualChance must be between 0 and 1.");
  }
  if (!Number.isInteger(years) || years < 1) {
    throw new RangeError("years must be a positive integer.");
  }

  return 1 - (1 - annualChance) ** years;
}

function roundedPercent(probability: number) {
  return Math.round(probability * 1_000) / 10;
}

function isSpecialFloodHazardArea({ zone, sfha }: FloodZoneInput) {
  const normalizedZone = zone?.toUpperCase().trim() ?? "";
  return (
    sfha === "T" ||
    ["A", "AE", "A1-30", "AO", "AH", "AR", "A99", "V", "VE", "V1-30"].includes(
      normalizedZone,
    )
  );
}

function isPointTwoPercentArea({ subtype }: FloodZoneInput) {
  const normalizedSubtype = subtype?.toUpperCase().trim() ?? "";
  return (
    normalizedSubtype.includes("0.2 PCT") ||
    normalizedSubtype.includes("MODERATE FLOOD HAZARD")
  );
}

export function femaHundredYearBenchmark(
  floodZone: FloodZoneInput,
): HundredYearFloodBenchmark {
  if (isSpecialFloodHazardArea(floodZone)) {
    const percent = roundedPercent(
      probabilityOverYears(ONE_PERCENT_ANNUAL_CHANCE, 100),
    );
    return {
      horizonYears: 100,
      status: "available",
      percent,
      display: `${percent}%`,
      annualChancePercent: 1,
      event: "one_percent_annual_chance",
      basis:
        "FEMA maps this location in a Special Flood Hazard Area associated with the 1%-annual-chance flood.",
      caveat:
        "This is the chance that the mapped 1%-annual-chance event occurs at least once in 100 years, assuming the annual chance stays constant and years are independent. It is not a parcel-specific forecast.",
    };
  }

  if (isPointTwoPercentArea(floodZone)) {
    const percent = roundedPercent(
      probabilityOverYears(POINT_TWO_PERCENT_ANNUAL_CHANCE, 100),
    );
    return {
      horizonYears: 100,
      status: "available",
      percent,
      display: `${percent}%`,
      annualChancePercent: 0.2,
      event: "point_two_percent_annual_chance",
      basis:
        "FEMA maps this location in a moderate-hazard area associated with the 0.2%-annual-chance flood.",
      caveat:
        "This is the chance that the mapped 0.2%-annual-chance event occurs at least once in 100 years, assuming the annual chance stays constant and years are independent. It is not a parcel-specific forecast.",
    };
  }

  return {
    horizonYears: 100,
    status: "not_determined",
    percent: null,
    display: "Not determined",
    annualChancePercent: null,
    event: null,
    basis:
      "This FEMA classification does not provide a numeric annual flood probability for this location.",
    caveat:
      "The model exposure score is not converted into a 100-year probability because the current claim/comparison data has no verified observation period for never-flooded properties.",
  };
}
