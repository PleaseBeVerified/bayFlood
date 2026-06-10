const BAY_AREA_BBOX = "-123.6,36.85,-121.15,38.9";
const PHOTON_ENDPOINT =
  process.env.PHOTON_GEOCODER_URL ?? "https://photon.komoot.io/api";

type PhotonFeature = {
  geometry?: {
    coordinates?: [number, number];
  };
  properties?: {
    name?: string;
    housenumber?: string;
    street?: string;
    city?: string;
    county?: string;
    state?: string;
    postcode?: string;
    country?: string;
    osm_id?: number;
    osm_type?: string;
  };
};

function formatAddress(properties: NonNullable<PhotonFeature["properties"]>) {
  const streetAddress = [properties.housenumber, properties.street]
    .filter(Boolean)
    .join(" ");

  return [
    streetAddress || properties.name,
    properties.city || properties.county,
    properties.state,
    properties.postcode,
  ]
    .filter(Boolean)
    .join(", ");
}

export async function GET(request: Request) {
  const query = new URL(request.url).searchParams.get("q")?.trim();

  if (!query || query.length < 3) {
    return Response.json({ results: [] });
  }

  const url = new URL(PHOTON_ENDPOINT);
  url.searchParams.set("q", query);
  url.searchParams.set("limit", "5");
  url.searchParams.set("lang", "en");
  url.searchParams.set("bbox", BAY_AREA_BBOX);

  try {
    const response = await fetch(url, {
      headers: {
        Accept: "application/geo+json, application/json",
      },
      next: { revalidate: 86400 },
    });

    if (!response.ok) {
      throw new Error(`Geocoder returned ${response.status}`);
    }

    const data = (await response.json()) as { features?: PhotonFeature[] };
    const results = (data.features ?? []).flatMap((feature) => {
      const coordinates = feature.geometry?.coordinates;
      const properties = feature.properties;

      if (!coordinates || !properties) return [];

      return [
        {
          id: `${properties.osm_type ?? "place"}-${properties.osm_id ?? coordinates.join("-")}`,
          label: formatAddress(properties),
          longitude: coordinates[0],
          latitude: coordinates[1],
        },
      ];
    });

    return Response.json(
      { results },
      { headers: { "Cache-Control": "public, max-age=3600" } },
    );
  } catch (error) {
    console.error("Address search failed", error);
    return Response.json(
      { error: "Address search is temporarily unavailable.", results: [] },
      { status: 502 },
    );
  }
}
