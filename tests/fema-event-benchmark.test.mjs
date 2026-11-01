import assert from "node:assert/strict";
import test from "node:test";
import { cumulativeEventChance, cumulativeEventUpperBound, isMinimalHazardZone, mappedAnnualChance } from "../src/lib/fema-event-benchmark.ts";

test("10- and 20-year mapped-event benchmarks use annual exceedance chance", () => {
  assert.equal(cumulativeEventChance(0.01, 10), 9.6);
  assert.equal(cumulativeEventChance(0.01, 20), 18.2);
  assert.equal(cumulativeEventChance(0.002, 10), 2.0);
  assert.equal(cumulativeEventChance(0.002, 20), 3.9);
  assert.equal(cumulativeEventUpperBound(0.002, 10), 2.0);
  assert.equal(cumulativeEventUpperBound(0.002, 20), 4.0);
});

test("only mapped annual-chance zones produce a numeric benchmark", () => {
  assert.equal(mappedAnnualChance({ zone: "AE", subtype: null, sfha: "T" }), 0.01);
  assert.equal(mappedAnnualChance({ zone: "X", subtype: "0.2 PCT ANNUAL CHANCE FLOOD HAZARD", sfha: "F" }), 0.002);
  assert.equal(mappedAnnualChance({ zone: "X", subtype: "1 PCT DEPTH LESS THAN 1 FOOT", sfha: "F" }), 0.01);
  assert.equal(mappedAnnualChance({ zone: "X", subtype: "1 PCT DRAINAGE AREA LESS THAN 1 SQUARE MILE", sfha: "F" }), 0.01);
  assert.equal(mappedAnnualChance({ zone: "X", subtype: "1 PCT FUTURE CONDITIONS", sfha: "F" }), null);
  assert.equal(mappedAnnualChance({ zone: "X", subtype: "AREA OF MINIMAL FLOOD HAZARD", sfha: "F" }), null);
  assert.equal(mappedAnnualChance({ zone: "D", subtype: null, sfha: "F" }), null);
  assert.equal(isMinimalHazardZone({ zone: "X", subtype: "AREA OF MINIMAL FLOOD HAZARD", sfha: "F" }), true);
  assert.equal(isMinimalHazardZone({ zone: "X", subtype: "0.2 PCT ANNUAL CHANCE FLOOD HAZARD", sfha: "F" }), false);
});
