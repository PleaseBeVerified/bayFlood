import { sampleElevation } from "@/lib/elevation";
import { lookupFema } from "@/lib/fema";
import { calculateFloodRisk } from "@/lib/flood-risk";
import { sampleRainfall } from "@/lib/rainfall";

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
    const [elevationMeters, fema, rainfallInches] = await Promise.all([
      sampleElevation(longitude, latitude),
      lookupFema(longitude, latitude),
      sampleRainfall(longitude, latitude),
    ]);

    if (elevationMeters === null) {
      return Response.json(
        { error: "No elevation tile covers this location." },
        { status: 404 },
      );
    }

    const risk = calculateFloodRisk(
      elevationMeters,
      fema.floodZone,
      fema.distanceToSfhaMeters,
      rainfallInches,
    );

    return Response.json({
      latitude,
      longitude,
      elevationMeters: Math.round(elevationMeters * 10) / 10,
      elevationFeet: Math.round(elevationMeters * 3.28084),
      floodZone: fema.floodZone,
      distanceToSfhaMeters: fema.distanceToSfhaMeters,
      rainfallInches:
        rainfallInches === null ? null : Math.round(rainfallInches * 100) / 100,
      ...risk,
    });
  } catch (error) {
    console.error("Flood risk calculation failed", error);
    return Response.json(
      { error: "Flood risk calculation is temporarily unavailable." },
      { status: 500 },
    );
  }
}
