import assert from "node:assert/strict";
import test from "node:test";
import { lookupFema } from "../src/lib/fema.ts";

test("local FEMA lookup covers known Bay Area zones and county edges", async () => {
  const cases = [
    {
      name: "downtown San Francisco",
      longitude: -122.4194,
      latitude: 37.7749,
      expected: { zone: "X", subtype: "AREA OF MINIMAL FLOOD HAZARD", sfha: "F" },
    },
    {
      name: "SFO",
      longitude: -122.388186,
      latitude: 37.614532,
      expected: { zone: "AE", subtype: null, sfha: "T" },
    },
    {
      name: "Palo Alto",
      longitude: -122.1601873,
      latitude: 37.4512686,
      expected: { zone: "AH", subtype: null, sfha: "T" },
    },
    {
      name: "Carquinez boundary",
      longitude: -122.4,
      latitude: 38,
      expected: { zone: "OPEN WATER", subtype: null, sfha: "F" },
    },
    {
      name: "eastern Contra Costa boundary",
      longitude: -121.579105,
      latitude: 37.859116,
      expected: { zone: "AE", subtype: null, sfha: "T" },
    },
  ];

  for (const sample of cases) {
    const result = await lookupFema(sample.longitude, sample.latitude);
    assert.deepEqual(result.floodZone, sample.expected, sample.name);
  }
});
