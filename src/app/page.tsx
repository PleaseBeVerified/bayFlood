"use client";

import { useEffect, useRef, useState } from "react";
import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { Protocol } from "pmtiles";
import "maplibre-gl/dist/maplibre-gl.css";

type SearchResult = {
  id: string;
  label: string;
  longitude: number;
  latitude: number;
};

type RiskResult = {
  score: number;
  modelExposureLikelihoodPercent: number;
  modelExposureMeaning: string;
  addressFloodHistoryUsed: false;
  level: string;
  color: string;
  confidence: "Low" | "Medium" | "High";
  elevationMeters: number;
  elevationFeet: number;
  rainfallInches: number | null;
  distanceToSfhaMeters: number | null;
  explanation: string;
  feedback: string[];
  nearestWater: {
    status: "found" | "not_found" | "unavailable";
    distanceMeters: number | null;
    name: string | null;
    kind: string | null;
    searchRadiusMeters: number;
  };
  floodZone: {
    zone: string | null;
    subtype: string | null;
    sfha: string | null;
  };
  hundredYearFloodBenchmark: {
    status: "available" | "not_determined";
    display: string;
    percent: number | null;
    annualChancePercent: number | null;
    basis: string;
    caveat: string;
  };
  hundredYearFloodEstimate: {
    horizonYears: 100;
    status: "fema_mapped_benchmark" | "experimental_physical_estimate";
    sourceLabel: string;
    display: string;
    percent: number;
    annualChancePercent: number;
    basis: string;
    caveat: string;
    range: { lowerPercent: number; upperPercent: number } | null;
    confidence: "Low" | "Medium" | null;
    addressFemaRecordRequired: boolean;
  };
  areaClaimSignal:
    | {
        status: "matched";
        censusBlockGroup: string;
        county: string;
        claims: number;
        policyYears: number;
        claimsPer1000PolicyYears: number;
        percentile: number;
        level: "Very low" | "Low" | "Moderate" | "High" | "Very high";
        asOfYear: number;
        source: string;
        interpretation: string;
      }
    | {
        status: "no_nfip_area_rate";
        censusBlockGroup: string;
        source: string;
        interpretation: string;
      }
    | {
        status: "unavailable";
        censusBlockGroup: null;
        source: string;
        interpretation: string;
      };
};

const floodZoneColors = [
  { color: "#9d174d", label: "Coastal high hazard (V / VE)" },
  { color: "#dc2626", label: "High risk (A / AE / AO / AH)" },
  { color: "#f59e0b", label: "Moderate risk (X, shaded)" },
  { color: "#2563eb", label: "Other mapped flood zones" },
];
function formatWaterKind(kind: string | null) {
  if (!kind) return "Water source";

  const labels: Record<string, string> = {
    "stream/river": "Stream or river",
    "canal/ditch": "Canal or ditch",
    "dam/weir": "Dam or weir",
    "sea/ocean": "Sea or ocean",
    "swamp/marsh": "Swamp or marsh",
    foreshore: "Foreshore",
    coastline: "Coastline",
    reservoir: "Reservoir",
    waterbody: "Water body",
  };

  return labels[kind] ?? kind.charAt(0).toUpperCase() + kind.slice(1);
}

function formatDistance(distanceMeters: number | null) {
  if (distanceMeters === null) return "Not available";
  if (distanceMeters < 1_000) return `${distanceMeters.toLocaleString()} m`;
  return `${(distanceMeters / 1_000).toFixed(1)} km`;
}
export default function Home() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const marker = useRef<maplibregl.Marker | null>(null);
  const skipNextSearch = useRef(false);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [risk, setRisk] = useState<RiskResult | null>(null);
  const [riskError, setRiskError] = useState("");
  const [loadingRisk, setLoadingRisk] = useState(false);

  async function assessLocation(longitude: number, latitude: number) {
    const nextMap = map.current;
    if (!nextMap) return;

    setLoadingRisk(true);
    setRiskError("");
    setRisk(null);

    const point = nextMap.project([longitude, latitude]);
    const features = nextMap.queryRenderedFeatures(point, {
      layers: ["flood-zones-fill"],
    });
    const properties = features[0]?.properties;
    const parameters = new URLSearchParams({
      lat: latitude.toString(),
      lng: longitude.toString(),
    });

    if (properties?.FLD_ZONE) parameters.set("zone", properties.FLD_ZONE);
    if (properties?.ZONE_SUBTY) parameters.set("subtype", properties.ZONE_SUBTY);
    if (properties?.SFHA_TF) parameters.set("sfha", properties.SFHA_TF);

    try {
      const response = await fetch(`/api/flood-risk?${parameters}`);
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setRisk(data);
    } catch (error) {
      setRiskError(
        error instanceof Error ? error.message : "Unable to calculate risk.",
      );
    } finally {
      setLoadingRisk(false);
    }
  }

  useEffect(() => {
    if (query.trim().length < 3) return;
    if (skipNextSearch.current) {
      skipNextSearch.current = false;
      return;
    }

    const controller = new AbortController();
    const timeout = window.setTimeout(async () => {
      setSearching(true);
      try {
        const response = await fetch(
          `/api/geocode?q=${encodeURIComponent(query)}`,
          { signal: controller.signal },
        );
        const data = await response.json();
        setResults(data.results ?? []);
      } catch (error) {
        if ((error as Error).name !== "AbortError") setResults([]);
      } finally {
        setSearching(false);
      }
    }, 350);

    return () => {
      controller.abort();
      window.clearTimeout(timeout);
    };
  }, [query]);

  useEffect(() => {
    if (!mapContainer.current || map.current) return;

    const protocol = new Protocol();
    maplibregl.addProtocol("pmtiles", protocol.tile);
    const nextMap = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          basemap: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution:
              '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          },
          floodZones: {
            type: "vector",
            url: "pmtiles:///data/bay-area-flood-zones.pmtiles",
            attribution: "FEMA National Flood Hazard Layer",
          },
        },
        layers: [
          { id: "basemap", type: "raster", source: "basemap" },
          {
            id: "flood-zones-fill",
            type: "fill",
            source: "floodZones",
            "source-layer": "flood_zones",
            paint: {
              "fill-color": [
                "match",
                ["get", "FLD_ZONE"],
                ["V", "VE"],
                "#9d174d",
                ["A", "AE", "A1-30", "AO", "AH", "AR", "A99"],
                "#dc2626",
                ["X", "0.2 PCT ANNUAL CHANCE FLOOD HAZARD"],
                "#f59e0b",
                "#2563eb",
              ],
              "fill-opacity": 0.48,
            },
          },
          {
            id: "flood-zones-outline",
            type: "line",
            source: "floodZones",
            "source-layer": "flood_zones",
            paint: {
              "line-color": "#172554",
              "line-opacity": 0.55,
              "line-width": ["interpolate", ["linear"], ["zoom"], 7, 0.25, 13, 1],
            },
          },
        ],
      },
      bounds: [
        [-123.6, 36.85],
        [-121.15, 38.9],
      ],
      fitBoundsOptions: { padding: 36 },
      maxBounds: [ 
        [-125, 35.5],
        [-119.5, 40],
      ],
    }); 

    nextMap.addControl(new maplibregl.NavigationControl(), "top-right");
    nextMap.addControl(
      new maplibregl.ScaleControl({ unit: "imperial" }),
      "bottom-right",  
    );
    nextMap.on("click", (event) => {
      marker.current?.remove();
      marker.current = new maplibregl.Marker({ color: "#0369a1" })
        .setLngLat(event.lngLat)
        .addTo(nextMap);
      void assessLocation(event.lngLat.lng, event.lngLat.lat);
    });

    map.current = nextMap; 
    return () => {
      marker.current?.remove();
      nextMap.remove(); 
      map.current = null;
      maplibregl.removeProtocol("pmtiles");
    };
  }, []);

  function selectResult(result: SearchResult) {
    skipNextSearch.current = true;
    setQuery(result.label);
    setResults([]);
    map.current?.flyTo({
      center: [result.longitude, result.latitude],
      zoom: 14,
      essential: true,
    });
    marker.current?.remove();
    marker.current = new maplibregl.Marker({ color: "#0369a1" })
      .setLngLat([result.longitude, result.latitude])
      .addTo(map.current!);
    map.current?.once("idle", () => {
      void assessLocation(result.longitude, result.latitude);
    });
  }

  return (
    <main className="map-shell">
      <div ref={mapContainer} className="map" aria-label="Bay Area flood risk map" />
      <section className="map-panel">
        <p className="eyebrow">FEMA zones + USGS elevation</p>
        <h1>Bay Area Flood Risk</h1>
        <p className="panel-description">
          Search for an address or click the map to estimate flood exposure.
        </p>
        
        <div className="address-search">
          <label htmlFor="address">Address</label>
          <div className="search-input-wrap">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <circle cx="11" cy="11" r="7" />
              <path d="m20 20-4-4" />
            </svg>
            <input 
              id="address"
              value={query}
              onChange={(event) => {
                const nextQuery = event.target.value;
                setQuery(nextQuery);
                if (nextQuery.trim().length < 3)  setResults([]);
              }}
              placeholder="Search a Bay Area address"
              autoComplete="off"
            />
            {searching && <span className="search-spinner" />}
          </div>
          {results.length > 0 && (
            <div className="search-results">
              {results.map((result, index) => (
                <button
                  key={`${result.id}-${result.longitude}-${result.latitude}-${index}`}
                  onClick={() => selectResult(result)}
                >
                  <span className="result-pin">●</span>
                  {result.label}
                </button>
              ))}
              <small>Search results from Photon / OpenStreetMap</small>
            </div>
          )}
        </div>

        {(loadingRisk || riskError || risk) && (
          <section className="risk-card" aria-live="polite">
            {loadingRisk && (
              <div className="risk-loading">
                <span className="search-spinner" />
                Reading elevation and FEMA data...
              </div>
            )}
            {riskError && <p className="risk-description">{riskError}</p>}
            {risk && (
              <>
                <div className="risk-heading">
                  <div>
                    <span>Estimated chance of flooding within 100 years</span>
                    <strong
                      style={{
                        color: "#be123c",
                      }}
                    >
                      {risk.hundredYearFloodEstimate.sourceLabel}
                    </strong>
                  </div>
                  <div
                    className="risk-score"
                    style={{ borderColor: "#be123c", color: "#be123c" }}
                  >
                    {risk.hundredYearFloodEstimate.percent}
                    <small>%</small>
                  </div>
                </div>
                <div className="risk-meter">
                  <span
                    style={{
                      width: `${risk.hundredYearFloodEstimate.percent}%`,
                      backgroundColor: "#be123c",
                    }}
                  />
                </div>
                <div
                  className={`probability-benchmark probability-${risk.hundredYearFloodEstimate.status}`}
                >
                  <span>What this result means</span>
                  <p>{risk.hundredYearFloodEstimate.basis}</p>
                  <small>{risk.hundredYearFloodEstimate.caveat}</small>
                </div>
                <div
                  className={`area-claim-signal area-claim-${risk.areaClaimSignal.status}`}
                >
                  <div className="area-claim-heading">
                    <span>FEMA neighborhood claim history</span>
                    <strong>
                      {risk.areaClaimSignal.status === "matched"
                        ? risk.areaClaimSignal.level
                        : risk.areaClaimSignal.status === "no_nfip_area_rate"
                          ? "No area rate"
                          : "Unavailable"}
                    </strong>
                  </div>
                  {risk.areaClaimSignal.status === "matched" && (
                    <dl>
                      <div>
                        <dt>Historical claims</dt>
                        <dd>{risk.areaClaimSignal.claims.toLocaleString()}</dd>
                      </div>
                      <div>
                        <dt>NFIP policy-years</dt>
                        <dd>{risk.areaClaimSignal.policyYears.toFixed(1)}</dd>
                      </div>
                      <div>
                        <dt>Bay Area percentile</dt>
                        <dd>{risk.areaClaimSignal.percentile.toFixed(1)}</dd>
                      </div>
                    </dl>
                  )}
                  <p>{risk.areaClaimSignal.interpretation}</p>
                  {risk.areaClaimSignal.censusBlockGroup && (
                    <small>
                      Census block group {risk.areaClaimSignal.censusBlockGroup}
                      {risk.areaClaimSignal.status === "matched" &&
                        ` · Data through ${risk.areaClaimSignal.asOfYear}`}
                    </small>
                  )}
                </div>
                <div className="model-similarity-context">
                  <div>
                    <span>Claim-location similarity</span>
                    <strong>{risk.modelExposureLikelihoodPercent}/100</strong>
                  </div>
                  <p>
                    {risk.modelExposureMeaning} This secondary screening score
                    is not used as the 100-year probability.
                  </p>
                </div>
                <p className="risk-zone">
                  {risk.elevationFeet.toLocaleString()} ft elevation
                  {" · "}
                  FEMA {risk.floodZone.zone ? `Zone ${risk.floodZone.zone}` : "zone not mapped"}
                </p>
                <p className="confidence-rating">
                  Data confidence: {" "}
                  <strong className={`confidence-${risk.confidence.toLowerCase()}`}>
                    {risk.confidence}
                  </strong>
                  {risk.rainfallInches !== null &&
                    ` · 100-year/24-hour rainfall ${risk.rainfallInches.toFixed(2)} in`}
                </p>
                <div
                  className={`nearby-water ${risk.nearestWater.status === "found" ? "" : "nearby-water-muted"}`}
                >
                  <strong>Physical inputs</strong>
                  <dl>
                    <div>
                      <dt>Nearest water</dt>
                      <dd>
                        {risk.nearestWater.status === "found"
                          ? risk.nearestWater.name || formatWaterKind(risk.nearestWater.kind)
                          : "None found within 5 km"}
                      </dd>
                    </div>
                    <div>
                      <dt>Water distance</dt>
                      <dd>{formatDistance(risk.nearestWater.distanceMeters)}</dd>
                    </div>
                    <div>
                      <dt>SFHA distance</dt>
                      <dd>{formatDistance(risk.distanceToSfhaMeters)}</dd>
                    </div>
                  </dl>
                </div>
                <p className="risk-description">
                  {risk.explanation} The trained model uses these conditions and
                  nearby-water context to calculate the claim-location
                  similarity shown above.
                </p>
                <div className="risk-feedback">
                  <strong>What you can do</strong>
                  <ul>
                    {risk.feedback.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              </>
            )}
          </section>
        )}

        <div className="legend" aria-label="Flood zone legend">
          {floodZoneColors.map((item) => (
            <div className="legend-row" key={item.label}>
              <span
                className="legend-swatch"
                style={{ backgroundColor: item.color }}
              />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
        <p className="disclaimer">
          Screening estimate only, not a parcel-level flood determination. The
          primary percentage is a FEMA mapped-event benchmark where available;
          otherwise it is an experimental physical-only estimate. Claim-location
          similarity is shown separately as a score out of 100. Neither model
          captures parcel drainage, building elevation, or future climate change.
        </p>
      </section>
    </main>
  );
}
