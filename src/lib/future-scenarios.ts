import scenarioConfig from "../../data/dnn-scenarios.json" with { type: "json" };
import type { DnnFloodRiskInput } from "./dnn-model";

// The existing 2056 cases are illustrative sensitivity assumptions. Interpolating
// their input changes creates comparable 10- and 20-year cases, not a forecast.
const BASELINE_YEAR = 2026;
export const FUTURE_HORIZONS = [10, 20] as const;

export function projectFutureInputs(input: DnnFloodRiskInput, yearsAhead: 10 | 20) {
  const fraction = yearsAhead / (scenarioConfig.scenarioYear - BASELINE_YEAR);
  return scenarioConfig.scenarios.map((scenario) => {
    const imperviousnessIncreasePercentagePoints =
      scenario.imperviousnessIncreasePercentagePoints * fraction;
    const rainfallIncreasePercent = scenario.rainfallIncreasePercent * fraction;
    return {
      id: scenario.id,
      year: BASELINE_YEAR + yearsAhead,
      imperviousnessIncreasePercentagePoints,
      rainfallIncreasePercent,
      input: {
        ...input,
        imperviousnessPercent: input.imperviousnessPercent === null
          ? null
          : Math.min(100, input.imperviousnessPercent + imperviousnessIncreasePercentagePoints),
        rainfallMillimeters: input.rainfallMillimeters === null
          ? null
          : input.rainfallMillimeters * (1 + rainfallIncreasePercent / 100),
      },
    };
  });
}
