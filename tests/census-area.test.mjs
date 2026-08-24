import assert from "node:assert/strict";
import test from "node:test";
import { coordinateKey, extractCensusGeography } from "../scripts/lib/census-area.mjs";

test("extracts a block group from a Census block response", () => {
  const result = extractCensusGeography({
    result: { geographies: { "2020 Census Blocks": [{ GEOID: "060855119103010" }] } },
  });
  assert.deepEqual(result, {
    census_block_geoid: "060855119103010",
    census_geoid: "060855119103",
    census_tract_geoid: "06085511910",
    county_fips: "06085",
  });
});

test("uses stable six-decimal coordinate cache keys", () => {
  assert.equal(coordinateKey(37.2059664, -121.8690599), "37.205966,-121.869060");
});
