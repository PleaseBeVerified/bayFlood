import assert from "node:assert/strict";
import test from "node:test";
import {
  femaHundredYearBenchmark,
  probabilityOverYears,
} from "../src/lib/flood-probability.ts";

test("converts an annual chance to a cumulative 100-year chance", () => {
  assert.equal(
    Math.round(probabilityOverYears(0.01, 100) * 1_000) / 10,
    63.4,
  );
  assert.equal(
    Math.round(probabilityOverYears(0.002, 100) * 1_000) / 10,
    18.1,
  );
});

test("uses FEMA event frequencies only for classifications that define one", () => {
  assert.deepEqual(
    femaHundredYearBenchmark({ zone: "AE", subtype: null, sfha: "T" }),
    {
      horizonYears: 100,
      status: "available",
      percent: 63.4,
      display: "63.4%",
      annualChancePercent: 1,
      event: "one_percent_annual_chance",
      basis:
        "FEMA maps this location in a Special Flood Hazard Area associated with the 1%-annual-chance flood.",
      caveat:
        "This is the chance that the mapped 1%-annual-chance event occurs at least once in 100 years, assuming the annual chance stays constant and years are independent. It is not a parcel-specific forecast.",
    },
  );

  assert.equal(
    femaHundredYearBenchmark({
      zone: "X",
      subtype: "0.2 PCT ANNUAL CHANCE FLOOD HAZARD",
      sfha: "F",
    }).percent,
    18.1,
  );

  const unmapped = femaHundredYearBenchmark({
    zone: "X",
    subtype: "AREA OF MINIMAL FLOOD HAZARD",
    sfha: "F",
  });
  assert.equal(unmapped.status, "not_determined");
  assert.equal(unmapped.percent, null);
});

test("rejects invalid probability inputs", () => {
  assert.throws(() => probabilityOverYears(-0.1, 100), RangeError);
  assert.throws(() => probabilityOverYears(0.01, 0), RangeError);
});
