import assert from "node:assert/strict";
import test from "node:test";
import { FEMA_NFHL_QUERY_URL, lookupFemaZoneWithLocalFallback, lookupLiveFemaFloodZone } from "../src/lib/fema-live-lookup.ts";

const minimalZone = { zone: "X", subtype: "AREA OF MINIMAL FLOOD HAZARD", sfha: "F" };
const highZone = { zone: "AE", subtype: null, sfha: "T" };

test("live lookup queries FEMA's point layer and selects the higher mapped hazard", async () => {
  let requestUrl;
  let requestOptions;
  const result = await lookupLiveFemaFloodZone(-122.03, 37.54, async (url, options) => {
    requestUrl = url;
    requestOptions = options;
    return Response.json({ features: [
      { attributes: { FLD_ZONE: "X", ZONE_SUBTY: minimalZone.subtype, SFHA_TF: "F" } },
      { attributes: { FLD_ZONE: "AE", ZONE_SUBTY: null, SFHA_TF: "T" } },
    ] });
  });

  assert.equal(requestUrl.origin + requestUrl.pathname, FEMA_NFHL_QUERY_URL);
  assert.equal(requestUrl.searchParams.get("geometry"), "-122.03,37.54");
  assert.equal(requestUrl.searchParams.get("inSR"), "4326");
  assert.equal(requestUrl.searchParams.get("outFields"), "FLD_ZONE,ZONE_SUBTY,SFHA_TF");
  assert.equal(requestOptions.headers["User-Agent"], "Mozilla/5.0");
  assert.deepEqual(result, { status: "matched", zone: highZone });
});

test("live lookup distinguishes no polygon from service failure", async () => {
  assert.deepEqual(
    await lookupLiveFemaFloodZone(-122, 38, async () => Response.json({ features: [] })),
    { status: "no_match" },
  );
  assert.deepEqual(
    await lookupLiveFemaFloodZone(-122, 38, async () => Response.json({ error: { message: "Unavailable" } })),
    { status: "unavailable" },
  );
  assert.deepEqual(
    await lookupLiveFemaFloodZone(-122, 38, async () => { throw new Error("network down"); }),
    { status: "unavailable" },
  );
});

test("live FEMA takes priority and local polygons provide fallback", async () => {
  const localMatch = await lookupFemaZoneWithLocalFallback(
    -122, 38,
    async () => { throw new Error("local lookup should not run after a live match"); },
    async () => ({ status: "matched", zone: highZone }),
  );
  assert.deepEqual(localMatch, { zone: highZone, source: "live_nfhl", liveStatus: "matched" });

  const localFallback = await lookupFemaZoneWithLocalFallback(
    -122, 38,
    async () => minimalZone,
    async () => ({ status: "unavailable" }),
  );
  assert.deepEqual(localFallback, { zone: minimalZone, source: "local_snapshot", liveStatus: "unavailable" });

  const noLiveMatch = await lookupFemaZoneWithLocalFallback(
    -122, 38,
    async () => minimalZone,
    async () => ({ status: "no_match" }),
  );
  assert.deepEqual(noLiveMatch, { zone: minimalZone, source: "local_snapshot", liveStatus: "no_match" });

  const unavailable = await lookupFemaZoneWithLocalFallback(
    -122, 38,
    async () => null,
    async () => ({ status: "unavailable" }),
  );
  assert.deepEqual(unavailable, { zone: null, source: "none", liveStatus: "unavailable" });
});
