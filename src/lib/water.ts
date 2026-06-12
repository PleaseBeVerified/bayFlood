const OVERPASS_URL = "https://overpass-api.de/api/interpreter";
const SEARCH_RADIUS = 5_000;

type WaterElement = {
  tags?: Record<string, string>;
  geometry?: { lat: number; lon: number }[];
};

export type NearbyWater = {
  distanceMeters: number | null;
  name: string | null;
  kind: string | null;
};

function distanceToSegment(
  longitude: number,
  latitude: number,
  start: { lat: number; lon: number },
  end: { lat: number; lon: number },
) {
  const latitudeScale = 111_320;
  const longitudeScale =
    latitudeScale * Math.cos((latitude * Math.PI) / 180);

  const px = longitude * longitudeScale;
  const py = latitude * latitudeScale;
  const x1 = start.lon * longitudeScale;
  const y1 = start.lat * latitudeScale;
  const x2 = end.lon * longitudeScale;
  const y2 = end.lat * latitudeScale;

  const dx = x2 - x1;
  const dy = y2 - y1;
  const lengthSquared = dx * dx + dy * dy;

  const amount =
    lengthSquared === 0
      ? 0
      : Math.max(
          0,
          Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lengthSquared),
        );

  return Math.hypot(px - (x1 + amount * dx), py - (y1 + amount * dy));
}

export async function lookupNearbyWater(
  longitude: number,
  latitude: number,
): Promise<NearbyWater> {
  const query = `
    [out:json][timeout:12];
    (
      way(around:${SEARCH_RADIUS},${latitude},${longitude})
        ["waterway"~"^(river|stream|canal|drain)$"];
      way(around:${SEARCH_RADIUS},${latitude},${longitude})["natural"="water"];
      way(around:${SEARCH_RADIUS},${latitude},${longitude})["natural"="coastline"];
    );
    out tags geom;
  `;

  try {
    const response = await fetch(OVERPASS_URL, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({ data: query }),
      cache: "no-store",
    });

    if (!response.ok) throw new Error(`Overpass returned ${response.status}`);

    const data = (await response.json()) as { elements?: WaterElement[] };
    let nearest: NearbyWater = {
      distanceMeters: null,
      name: null,
      kind: null,
    };

    for (const element of data.elements ?? []) {
      const geometry = element.geometry ?? [];

      for (let index = 1; index < geometry.length; index += 1) {
        const distance = distanceToSegment(
          longitude,
          latitude,
          geometry[index - 1],
          geometry[index],
        );

        if (nearest.distanceMeters === null || distance < nearest.distanceMeters) {
          nearest = {
            distanceMeters: Math.round(distance),
            name: element.tags?.name ?? null,
            kind:
              element.tags?.waterway ??
              (element.tags?.natural === "coastline"
                ? "coastline"
                : element.tags?.water ?? "water body"),
          };
        }
      }
    }

    return nearest;
  } catch (error) {
    console.error("Nearby water lookup failed", error);
    return { distanceMeters: null, name: null, kind: null };
  }
}