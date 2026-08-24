import { sampleElevation } from "@/lib/elevation";
import { lookupFemaAreaClaimSignal } from "@/lib/fema-area-signal";
import { lookupFema } from "@/lib/fema";
import {
  modelRiskPresentation,
  predictFloodClaimSimilarity,
} from "@/lib/flood-model";
import { describeFloodConditions } from "@/lib/flood-guidance";
import { femaHundredYearBenchmark } from "@/lib/flood-probability";
import { sampleRainfall } from "@/lib/rainfall";
import { estimatePhysicalHundredYearFloodChance } from "@/lib/physical-flood-estimate";
import { sampleTerrainFeatures } from "@/lib/terrain";
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
    const [
      elevationMeters,
      fema,
      rainfallInches,
      nearbyWater,
      areaClaimSignal,
      terrain,
    ] = await Promise.all([
      sampleElevation(longitude, latitude),
      lookupFema(longitude, latitude),
      sampleRainfall(longitude, latitude),
      lookupNearbyWater(longitude, latitude),
      lookupFemaAreaClaimSignal(longitude, latitude),
      sampleTerrainFeatures(longitude, latitude),
    ]);

    if (elevationMeters === null) {
      return Response.json(
        { error: "No elevation tile covers this location." },
        { status: 404 },
      );
    }

    const guidance = describeFloodConditions(
      elevationMeters,
      fema.floodZone,
      fema.distanceToSfhaMeters,
      rainfallInches,
    );
    const modelPrediction = predictFloodClaimSimilarity({
      elevation_m: Math.round(elevationMeters * 10) / 10,
      sfha:
        fema.floodZone.sfha === null || fema.floodZone.sfha === undefined
          ? null
          : fema.floodZone.sfha === "T"
            ? 1
            : 0,
      distance_sfha_m: fema.distanceToSfhaMeters,
      rainfall_100yr_24h_in:
        rainfallInches === null ? null : Math.round(rainfallInches * 100) / 100,
      nearest_water_distance_m: nearbyWater.distanceMeters,
      fema_zone: fema.floodZone.zone ?? "Unknown",
      fema_subtype: fema.floodZone.subtype ?? "Unknown",
      nearest_water_kind: nearbyWater.kind ?? "Unknown",
    });
    const presentation = modelRiskPresentation(modelPrediction.probability);
    const hundredYearFloodBenchmark = femaHundredYearBenchmark(fema.floodZone);
    const physicalHundredYearEstimate = estimatePhysicalHundredYearFloodChance({
      elevation_m: Math.round(elevationMeters * 10) / 10,
      rainfall_100yr_24h_in:
        rainfallInches === null ? null : Math.round(rainfallInches * 100) / 100,
      nearest_water_distance_m: nearbyWater.distanceMeters,
      nearest_water_kind: nearbyWater.kind ?? "Unknown",
      ...terrain,
    });
    const hundredYearFloodEstimate =
      hundredYearFloodBenchmark.status === "available"
        ? {
            ...hundredYearFloodBenchmark,
            status: "fema_mapped_benchmark" as const,
            sourceLabel: "FEMA mapped benchmark",
            range: null,
            confidence: null,
            addressFemaRecordRequired: true as const,
          }
        : {
            ...physicalHundredYearEstimate,
            sourceLabel: "Experimental physical estimate",
          };

    return Response.json({
      latitude,
      longitude,
      elevationMeters: Math.round(elevationMeters * 10) / 10,
      elevationFeet: Math.round(elevationMeters * 3.28084),
      floodZone: fema.floodZone,
      distanceToSfhaMeters: fema.distanceToSfhaMeters,
      rainfallInches:
        rainfallInches === null ? null : Math.round(rainfallInches * 100) / 100,
      nearestWater: nearbyWater,
      terrain,
      ...guidance,
      score: modelPrediction.score,
      level: presentation.level,
      color: presentation.color,
      scoreMethod: "trained_model",
      modelProbability: modelPrediction.probability,
      modelExposureLikelihoodPercent: modelPrediction.score,
      modelExposureMeaning:
        "Similarity to known NFIP claim locations versus comparison locations in the training data. It can screen an address with no known flood history, but it is not a time-based flood probability.",
      addressFloodHistoryUsed: false,
      modelThreshold: modelPrediction.threshold,
      modelClassification: modelPrediction.classification,
      hundredYearFloodBenchmark,
      physicalHundredYearEstimate,
      hundredYearFloodEstimate,
      areaClaimSignal,
    });
  } catch (error) {
    console.error("Flood risk calculation failed", error);
    return Response.json(
      { error: "Flood risk calculation is temporarily unavailable." },
      { status: 500 },
    );
  }
}
