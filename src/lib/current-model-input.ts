import { sampleElevation } from "./elevation";
import { sampleImperviousness } from "./nlcd";
import { sampleRainfall } from "./rainfall";
import { sampleTerrain } from "./terrain";
import { lookupNearbyWater } from "./water";
import type { BoostedFloodRiskInput } from "./boosted-model";

export async function sampleCurrentInputData(
  longitude: number,
  latitude: number,
) {
  const [elevationResult, imperviousnessResult, rainfallResult, waterResult, terrainResult] =
    await Promise.allSettled([
      sampleElevation(longitude, latitude),
      sampleImperviousness(longitude, latitude),
      sampleRainfall(longitude, latitude),
      lookupNearbyWater(longitude, latitude),
      sampleTerrain(longitude, latitude),
    ]);
  const elevationMeters = elevationResult.status === "fulfilled" ? elevationResult.value : null;
  const imperviousnessPercent = imperviousnessResult.status === "fulfilled" ? imperviousnessResult.value : null;
  const rainfallInches = rainfallResult.status === "fulfilled" ? rainfallResult.value : null;
  const nearbyWater = waterResult.status === "fulfilled" ? waterResult.value : null;
  const terrain = terrainResult.status === "fulfilled" ? terrainResult.value : null;
  const inputs = {
    elevationMeters,
    imperviousnessPercent,
    rainfallMillimeters: rainfallInches === null ? null : rainfallInches * 25.4,
    nearestWaterAvailable: nearbyWater === null || nearbyWater.status === "unavailable"
      ? null : nearbyWater.status === "found" ? 1 : 0,
    nearestWaterDistanceMeters: nearbyWater?.distanceMeters ?? null,
    slopeDegrees: terrain?.slopeDegrees ?? null,
    localElevation250Meters: terrain?.localElevation250Meters ?? null,
    localElevation500Meters: terrain?.localElevation500Meters ?? null,
    flowAccumulationSquareMeters: terrain?.flowAccumulationSquareMeters ?? null,
    heightAboveDrainageMeters: terrain?.heightAboveDrainageMeters ?? null,
  };
  const modelInput: BoostedFloodRiskInput | null = elevationMeters === null || terrain === null
    ? null : { ...inputs, elevationMeters, ...terrain };
  return { inputs, modelInput };
}
