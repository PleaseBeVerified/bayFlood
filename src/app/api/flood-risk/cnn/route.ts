import { sampleCurrentInputData } from "@/lib/current-model-input";
import { predictTerrainCnn } from "@/lib/terrain-cnn";

export const runtime = "nodejs";

export async function GET(request: Request) {
  const parameters = new URL(request.url).searchParams;
  const latitudeText = parameters.get("lat");
  const longitudeText = parameters.get("lng");
  const latitude = Number(latitudeText);
  const longitude = Number(longitudeText);
  if (latitudeText === null || longitudeText === null ||
      !Number.isFinite(latitude) || !Number.isFinite(longitude) ||
      latitude < -90 || latitude > 90 || longitude < -180 || longitude > 180) {
    return Response.json({ error: "Valid lat and lng parameters are required." }, { status: 400 });
  }

  try {
    const { modelInput: input } = await sampleCurrentInputData(longitude, latitude);
    if (input === null) {
      return Response.json({ cnnModel: null, reason: "Local elevation or terrain data is unavailable." });
    }
    const cnnModel = await predictTerrainCnn(longitude, latitude, input);
    return Response.json({
      cnnModel,
      reason: cnnModel === null ? "The CNN raster has no terrain data at this location." : null,
    }, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    console.error("Terrain CNN calculation failed", error);
    return Response.json(
      { error: "The research CNN is unavailable. Check its local raster cache and Python model environment." },
      { status: 503 },
    );
  }
}
