import { cumulativeEventChance, cumulativeEventUpperBound, isMinimalHazardZone, mappedAnnualChance } from "@/lib/fema-event-benchmark";
import { lookupFemaZoneWithLocalFallback } from "@/lib/fema-live-lookup";
import { predictBoostedFloodRisk } from "@/lib/boosted-model";
import { sampleElevation } from "@/lib/elevation";
import { sampleImperviousness } from "@/lib/nlcd";
import { sampleRainfall } from "@/lib/rainfall";
import { lookupNearbyWater } from "@/lib/water";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const parameters = new URL(request.url).searchParams;
  const latitude = Number(parameters.get("lat"));
  const longitude = Number(parameters.get("lng"));

  if (
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude) ||
    latitude < -90 ||
    latitude > 90 ||
    longitude < -180 ||
    longitude > 180
  ) {
    return Response.json(
      { error: "Valid lat and lng parameters are required." },
      { status: 400 },
    );
  }

  try {
    const [zoneLookup, samples] = await Promise.all([
      lookupFemaZoneWithLocalFallback(longitude, latitude),
      Promise.allSettled([
        sampleElevation(longitude, latitude),
        sampleImperviousness(longitude, latitude),
        sampleRainfall(longitude, latitude),
        lookupNearbyWater(longitude, latitude),
      ]),
    ]);
    const floodZone = zoneLookup.zone;
    const [elevationResult, imperviousnessResult, rainfallResult, waterResult] = samples;
    const elevationMeters = elevationResult.status === "fulfilled" ? elevationResult.value : null;
    const imperviousnessPercent = imperviousnessResult.status === "fulfilled" ? imperviousnessResult.value : null;
    const rainfallInches = rainfallResult.status === "fulfilled" ? rainfallResult.value : null;
    const nearbyWater = waterResult.status === "fulfilled" ? waterResult.value : null;
    const modelPrediction = elevationMeters === null ? null : predictBoostedFloodRisk({
      elevationMeters,
      imperviousnessPercent,
      rainfallMillimeters: rainfallInches === null ? null : rainfallInches * 25.4,
      nearestWaterAvailable: nearbyWater === null || nearbyWater.status === "unavailable"
        ? null : nearbyWater.status === "found" ? 1 : 0,
      nearestWaterDistanceMeters: nearbyWater?.distanceMeters ?? null,
    });
    const model = modelPrediction === null ? null : {
      scoreOutOf100: modelPrediction.scorePercent,
      classification: modelPrediction.classification,
      missingFeatures: modelPrediction.missingFeatures,
    };
    const annualChance = floodZone === null ? null : mappedAnnualChance(floodZone);
    const upperBound = floodZone !== null && isMinimalHazardZone(floodZone);
    const modelMapDisagreement = model === null ? null
      : upperBound && model.classification === "higher-risk pattern"
        ? "The model flags this point, but FEMA maps it as minimal flood hazard. This high model score should not be read as a high flood probability."
        : annualChance === 0.01 && model.classification === "lower-risk pattern"
          ? "The model gives this point a low score, but FEMA maps it in a 1%-annual-chance flood hazard area. Use the mapped classification for the event benchmark."
          : null;

    return Response.json({
      latitude,
      longitude,
      model,
      modelMapDisagreement,
      floodEventChance: {
        source: zoneLookup.source,
        liveStatus: zoneLookup.liveStatus,
        kind: annualChance !== null ? "mapped_benchmark" : upperBound ? "upper_bound" : "unavailable",
        tenYearPercent: annualChance !== null
          ? cumulativeEventChance(annualChance, 10)
          : upperBound ? cumulativeEventUpperBound(0.002, 10) : null,
        twentyYearPercent: annualChance !== null
          ? cumulativeEventChance(annualChance, 20)
          : upperBound ? cumulativeEventUpperBound(0.002, 20) : null,
        annualChancePercent: annualChance !== null ? annualChance * 100 : upperBound ? 0.2 : null,
        zone: floodZone?.zone ?? null,
        basis: floodZone === null
          ? zoneLookup.liveStatus === "unavailable"
            ? "No local FEMA polygon matched, and the live FEMA service could not be reached. The mapped flood-event percentage is unavailable until the current map can be checked."
            : "Neither the local FEMA files nor the live FEMA flood-hazard layer matched this point. No mapped annual-chance benchmark is available here."
          : upperBound
            ? `${zoneLookup.source === "live_nfhl" ? "The live FEMA map" : "The local FEMA map"} places this point in unshaded Zone X, outside the 0.2%-annual-chance floodplain. These are upper bounds for the mapped flood hazard, not exact flood probabilities.`
          : annualChance === null
            ? floodZone.zone?.toUpperCase().trim() === "D"
              ? "FEMA maps this point in Zone D, where flood hazard is possible but the annual chance is undetermined."
              : floodZone.zone?.toUpperCase().trim() === "X"
                ? `FEMA maps this point in Zone X (${floodZone.subtype ?? "special designation"}). This classification does not provide a usable annual-chance benchmark in the app.`
                : "This FEMA flood-zone classification does not supply a numeric annual-chance benchmark."
          : floodZone.zone?.toUpperCase().trim() === "X" && annualChance === 0.01
            ? `FEMA maps this point in Zone X (${floodZone.subtype}), an area of 1%-annual-chance shallow flooding. This is outside the Special Flood Hazard Area designation.`
            : `FEMA maps this point in a ${annualChance * 100}%-annual-chance flood hazard area.`,
      },
      caveat:
        "This map-based calculation assumes unchanged annual flood hazard and independent years. It is not the chance a particular building floods. Local drainage, other flood sources, and future climate change can change the actual risk. Verify the current FEMA map before making decisions.",
    });
  } catch (error) {
    console.error("Flood risk calculation failed", error);
    return Response.json(
      { error: "Flood risk calculation is temporarily unavailable." },
      { status: 500 },
    );
  }
}
