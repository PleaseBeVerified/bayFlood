import assert from "node:assert/strict";
import test from "node:test";
import { sampleTerrainFeatures } from "../src/lib/terrain.ts";

test("samples all projected terrain rasters at a Bay Area coordinate", async () => {
  const terrain = await sampleTerrainFeatures(-121.873, 37.285);

  for (const value of Object.values(terrain)) {
    assert.equal(typeof value, "number");
    assert.ok(Number.isFinite(value));
  }
  assert.ok(Math.abs(terrain.slope_degrees - 12.4718) < 0.01);
  assert.equal(terrain.flow_accumulation_m2, 23_800);
  assert.ok(Math.abs(terrain.height_above_drainage_m - 7.5778) < 0.01);
});

test("returns null terrain values outside the raster coverage", async () => {
  const terrain = await sampleTerrainFeatures(-118.2437, 34.0522);
  assert.deepEqual(new Set(Object.values(terrain)), new Set([null]));
});
