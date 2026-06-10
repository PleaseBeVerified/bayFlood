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
  level: string;
  color: string;
  elevationMeters: number;
  elevationFeet: number;
  elevationContribution: number;
  femaContribution: number;
  nearbyFloodplainContribution: number;
  rainfallContribution: number;
  rainfallInches: number | null;
  distanceToSfhaMeters: number | null;
  explanation: string;
  floodZone: {
    zone: string | null;
    subtype: string | null;
    sfha: string | null;
  };
};

const floodZoneColors = [
  { color: "#9d174d", label: "Coastal high hazard (V / VE)" },
  { color: "#dc2626", label: "High risk (A / AE / AO / AH)" },
  { color: "#f59e0b", label: "Moderate risk (X, shaded)" },
  { color: "#2563eb", label: "Other mapped flood zones" },
];

export default function Home() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const marker = useRef<maplibregl.Marker | null>(null);
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

    const parameters = new URLSearchParams({
      lat: latitude.toString(),
      lng: longitude.toString(),
    });

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
        <div className="app-brand">
          <span>BayFlood</span>
        </div>
        <p className="eyebrow">FEMA zones + USGS elevation + NOAA rainfall</p>
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
                if (nextQuery.trim().length < 3) setResults([]);
              }}
              placeholder="Search a Bay Area address"
              autoComplete="off"
            />
            {searching && <span className="search-spinner" />}
          </div>
          {results.length > 0 && (
            <div className="search-results">
              {results.map((result) => (
                <button key={result.id} onClick={() => selectResult(result)}>
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
                Reading elevation, FEMA, and NOAA rainfall data...
              </div>
            )}
            {riskError && <p className="risk-description">{riskError}</p>}
            {risk && (
              <>
                <div className="risk-heading">
                  <div>
                    <span>Estimated flood risk</span>
                    <strong style={{ color: risk.color }}>{risk.level}</strong>
                  </div>
                  <div
                    className="risk-score"
                    style={{ borderColor: risk.color, color: risk.color }}
                  >
                    {risk.score}<small>/100</small>
                  </div>
                </div>
                <div className="risk-meter">
                  <span
                    style={{
                      width: `${risk.score}%`,
                      backgroundColor: risk.color,
                    }}
                  />
                </div>
                <p className="risk-zone">
                  {risk.elevationFeet.toLocaleString()} ft elevation
                  {" · "}
                  FEMA {risk.floodZone.zone ? `Zone ${risk.floodZone.zone}` : "zone not mapped"}
                  {risk.distanceToSfhaMeters !== null &&
                    risk.distanceToSfhaMeters > 0 &&
                    ` · ${risk.distanceToSfhaMeters.toLocaleString()} m from high-risk floodplain`}
                  {risk.rainfallInches !== null &&
                    ` · ${risk.rainfallInches.toFixed(2)} in 100-year/24-hour rainfall`}
                </p>
                <p className="risk-description">
                  {risk.explanation} Score components: elevation{" "}
                  {risk.elevationContribution}, FEMA {risk.femaContribution},
                  nearby floodplain {risk.nearbyFloodplainContribution}, rainfall{" "}
                  {risk.rainfallContribution}.
                </p>
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
          Screening estimate only, not a parcel-level flood determination.
          This estimate does not model drainage capacity, levees, impervious
          surfaces, or future sea level rise.
        </p>
      </section>
    </main>
  );
}
