"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";

type SearchResult = { id: string; label: string; longitude: number; latitude: number };

type ModelInputs = {
  elevationMeters: number | null;
  imperviousnessPercent: number | null;
  rainfallMillimeters: number | null;
  nearestWaterAvailable: number | null;
  nearestWaterDistanceMeters: number | null;
  slopeDegrees: number | null;
  localElevation250Meters: number | null;
  localElevation500Meters: number | null;
  flowAccumulationSquareMeters: number | null;
  heightAboveDrainageMeters: number | null;
};

type RiskResult = {
  latitude: number;
  longitude: number;
  modelInputs: ModelInputs;
  model: {
    scoreOutOf100: number;
    classification: string;
    missingFeatures: string[];
  } | null;
  terrainDnnModel: {
    scoreOutOf100: number;
    classification: string;
    imputedFeatures: string[];
  } | null;
  modelMapDisagreement: string | null;
  floodEventChance: {
    source: "local_snapshot" | "live_nfhl" | "none";
    liveStatus: "matched" | "no_match" | "unavailable";
    kind: "mapped_benchmark" | "upper_bound" | "unavailable";
    tenYearPercent: number | null;
    twentyYearPercent: number | null;
    annualChancePercent: number | null;
    zone: string | null;
    basis: string;
  };
  caveat: string;
};

type CnnResult = {
  onePercentPatternScore: number;
  classification: string;
  imputedFeatures: string[];
  topographyCoverage: number;
  imperviousnessCoverage: number;
};

type SelectedLocation = {
  latitude: number;
  longitude: number;
  label: string;
};

function formatInput(value: number | null, unit: string) {
  return value === null ? "Unavailable" : `${value.toLocaleString(undefined, { maximumFractionDigits: 3 })}${unit}`;
}

export default function Home() {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const marker = useRef<maplibregl.Marker | null>(null);
  const skipNextSearch = useRef(false);
  const riskRequest = useRef<AbortController | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [risk, setRisk] = useState<RiskResult | null>(null);
  const [riskError, setRiskError] = useState("");
  const [loadingRisk, setLoadingRisk] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<SelectedLocation | null>(null);
  const [selectedModel, setSelectedModel] = useState<"boosted" | "terrainDnn" | "cnn">("boosted");
  const [cnnState, setCnnState] = useState<{ key: string; model: CnnResult | null; error: string } | null>(null);

  const assessLocation = useCallback(async (longitude: number, latitude: number) => {
    riskRequest.current?.abort();
    const controller = new AbortController();
    riskRequest.current = controller;
    setLoadingRisk(true);
    setRiskError("");
    setRisk(null);
    setCnnState(null);

    try {
      const parameters = new URLSearchParams({ lat: latitude.toString(), lng: longitude.toString() });
      const response = await fetch(`/api/flood-risk?${parameters}`, { signal: controller.signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setRisk(data);
    } catch (error) {
      if (controller.signal.aborted) return;
      setRiskError(error instanceof Error ? error.message : "Unable to calculate the mapped flood-event benchmark.");
    } finally {
      if (!controller.signal.aborted) setLoadingRisk(false);
    }
  }, []);

  const selectMapLocation = useCallback((longitude: number, latitude: number, label: string) => {
    setSelectedLocation({ longitude, latitude, label });
    void assessLocation(longitude, latitude);
  }, [assessLocation]);

  function clearAssessment() {
    riskRequest.current?.abort();
    setLoadingRisk(false);
    marker.current?.remove();
    setSelectedLocation(null);
    setRisk(null);
    setRiskError("");
    setCnnState(null);
    setQuery("");
    setResults([]);
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
        const response = await fetch(`/api/geocode?q=${encodeURIComponent(query)}`, {
          signal: controller.signal,
        });
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
    const nextMap = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          basemap: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
          },
        },
        layers: [
          { id: "basemap", type: "raster", source: "basemap" },
        ],
      },
      bounds: [[-123.1, 36.95], [-121.0, 38.9]],
      fitBoundsOptions: { padding: 36 },
      maxBounds: [[-123.2, 36.85], [-120.9, 39.0]],
    });

    nextMap.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    nextMap.addControl(new maplibregl.ScaleControl({ unit: "imperial" }), "bottom-right");
    nextMap.on("click", (event) => {
      marker.current?.remove();
      marker.current = new maplibregl.Marker({ color: "#0369a1" }).setLngLat(event.lngLat).addTo(nextMap);
      selectMapLocation(event.lngLat.lng, event.lngLat.lat, "Pinned map location");
    });

    map.current = nextMap;
    const resizeObserver = new ResizeObserver(() => nextMap.resize());
    resizeObserver.observe(mapContainer.current);
    return () => {
      resizeObserver.disconnect();
      riskRequest.current?.abort();
      marker.current?.remove();
      nextMap.remove();
      map.current = null;
    };
  }, [selectMapLocation]);

  function selectResult(result: SearchResult) {
    skipNextSearch.current = true;
    setQuery(result.label);
    setResults([]);
    map.current?.flyTo({ center: [result.longitude, result.latitude], zoom: 14 });
    marker.current?.remove();
    if (map.current) {
      marker.current = new maplibregl.Marker({ color: "#0369a1" })
        .setLngLat([result.longitude, result.latitude])
        .addTo(map.current);
    }
    selectMapLocation(result.longitude, result.latitude, result.label);
  }

  const selectedDnnModel = risk?.terrainDnnModel;
  const cnnKey = risk !== null && selectedLocation !== null &&
    risk.latitude === selectedLocation.latitude && risk.longitude === selectedLocation.longitude
    ? `${risk.latitude},${risk.longitude}` : null;
  const activeCnnState = cnnState?.key === cnnKey ? cnnState : null;
  const cnnLoading = selectedModel === "cnn" && cnnKey !== null && activeCnnState === null;
  const cnnModel = activeCnnState?.model ?? null;
  const cnnError = activeCnnState?.error ?? "";

  useEffect(() => {
    if (selectedModel !== "cnn" || cnnKey === null || selectedLocation === null || activeCnnState !== null) {
      return;
    }
    const controller = new AbortController();
    const parameters = new URLSearchParams({
      lat: selectedLocation.latitude.toString(),
      lng: selectedLocation.longitude.toString(),
    });
    void (async () => {
      try {
        const response = await fetch(`/api/flood-risk/cnn?${parameters}`, { signal: controller.signal });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error ?? "CNN calculation failed.");
        setCnnState({ key: cnnKey, model: data.cnnModel ?? null, error: data.reason ?? "" });
      } catch (error) {
        if (!controller.signal.aborted) {
          setCnnState({ key: cnnKey, model: null, error: error instanceof Error ? error.message : "CNN calculation failed." });
        }
      }
    })();
    return () => controller.abort();
  }, [selectedModel, selectedLocation, cnnKey, activeCnnState]);

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">
            <svg viewBox="0 0 32 32"><path d="M6 20c2.7 0 2.7-2 5.4-2s2.7 2 5.3 2 2.7-2 5.4-2 2.7 2 5.4 2" /><path d="M6 25c2.7 0 2.7-2 5.4-2s2.7 2 5.3 2 2.7-2 5.4-2 2.7 2 5.4 2" /><path d="M16 5v8" /></svg>
          </div>
          <div>
            <h1>Bay Area <span>Flood Risk Analyzer</span></h1>
            <p>Local insight. A clearer perspective.</p>
          </div>
        </div>
        <div className="header-context"><span className="coverage-dot" /> California <span className="header-divider">/</span> Research project</div>
      </header>
      <div className="workspace">
      <section className="map-panel" aria-label="Location analysis">
        <div className="panel-intro">
          <p className="eyebrow">Location explorer</p>
          <h2>A clearer view <br />of flood risk.</h2>
        <p className="panel-description">
          Explore FEMA map-based flood-event chances for the current year and the next 10 or 20 years.
        </p>
        </div>

        <div className="address-search">
          <label htmlFor="address">Search a location</label>
          <div className="search-input-wrap">
            <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></svg>
            <input
              id="address"
              value={query}
              onChange={(event) => {
                const nextQuery = event.target.value;
                setQuery(nextQuery);
                if (nextQuery.trim().length < 3) setResults([]);
              }}
              onKeyDown={(event) => {
                if (event.key === "Enter" && results[0]) {
                  event.preventDefault();
                  selectResult(results[0]);
                }
                if (event.key === "Escape") setResults([]);
              }}
              placeholder="Search an address or place…"
              autoComplete="off"
              role="combobox"
              aria-expanded={results.length > 0}
              aria-controls="address-results"
              aria-autocomplete="list"
            />
            {searching && <span className="search-spinner" />}
          </div>
          {results.length > 0 && (
            <div id="address-results" className="search-results" role="listbox">
              {results.map((result) => (
                <button key={`${result.id}-${result.longitude}-${result.latitude}`} role="option" aria-selected={false} onClick={() => selectResult(result)}>
                  <span className="result-pin">●</span>{result.label}
                </button>
              ))}
              <small>Search results from Photon / OpenStreetMap</small>
            </div>
          )}
        </div>

        {!selectedLocation && (
          <div className="empty-state">
            <div className="contour-art" aria-hidden="true">
              <svg viewBox="0 0 320 150" fill="none">
                <path d="M-20 95C20 10 80 130 132 53S252 0 340 30M-20 108C22 24 82 144 142 67S252 14 340 44M-20 121C22 38 90 158 152 81S256 28 340 58M-20 134C26 52 96 172 162 95S256 42 340 72M-20 147C26 66 100 186 172 109S260 56 340 86M-20 160C26 80 106 200 182 123S260 70 340 100" />
                <circle cx="160" cy="68" r="29" className="pin-halo" />
                <path d="M160 39a18 18 0 0 0-18 18c0 14 18 29 18 29s18-15 18-29a18 18 0 0 0-18-18Z" className="art-pin" />
                <circle cx="160" cy="57" r="6" className="pin-center" />
              </svg>
              <span>YOUR NEXT PERSPECTIVE STARTS HERE</span>
            </div>
            <h3>Every location tells a story.</h3>
            <p>Search above or select a point on the map to see its current, 10-year, and 20-year FEMA mapped benchmark.</p>
          </div>
        )}

        {selectedLocation && (
          <div className="selected-location">
            <div>
              <span>Selected location</span>
              <strong>{selectedLocation.label}</strong>
              <small>{selectedLocation.latitude.toFixed(4)}, {selectedLocation.longitude.toFixed(4)}</small>
            </div>
            <button className="text-button" onClick={clearAssessment}>Clear</button>
          </div>
        )}

        {(loadingRisk || riskError || risk) && (
          <section className="risk-card" aria-live="polite">
            {loadingRisk && (
              <div className="risk-loading">
                <span className="search-spinner" />
                <div><strong>Analyzing location</strong><span>Checking the FEMA flood zone and current physical conditions…</span></div>
              </div>
            )}
            {riskError && (
              <div className="risk-error">
                <strong>We couldn’t assess this location</strong>
                <p>{riskError}</p>
                {selectedLocation && <button className="retry-button" onClick={() => assessLocation(selectedLocation.longitude, selectedLocation.latitude)}>Try again</button>}
              </div>
            )}
            {risk && (
              <>
                <div className="event-panel">
                  <span className="event-eyebrow">FEMA mapped flood-event chance</span>
                  <p className="event-source">Source: {risk.floodEventChance.source === "live_nfhl" ? "live FEMA National Flood Hazard Layer" : risk.floodEventChance.source === "local_snapshot" ? `local FEMA download (live FEMA ${risk.floodEventChance.liveStatus === "unavailable" ? "unavailable" : "had no match"})` : "no matching FEMA data"}</p>
                  <div className="event-percentages">
                    <div><small>Current annual benchmark</small><strong className={risk.floodEventChance.annualChancePercent === null ? "event-value-unavailable" : undefined}>{risk.floodEventChance.annualChancePercent === null ? "Unavailable" : `${risk.floodEventChance.kind === "upper_bound" ? "<" : ""}${risk.floodEventChance.annualChancePercent.toFixed(1)}%`}</strong></div>
                    <div><small>Next 10 years</small><strong className={risk.floodEventChance.tenYearPercent === null ? "event-value-unavailable" : undefined}>{risk.floodEventChance.tenYearPercent === null ? "Unavailable" : `${risk.floodEventChance.kind === "upper_bound" ? "<" : ""}${risk.floodEventChance.tenYearPercent.toFixed(1)}%`}</strong></div>
                    <div><small>Next 20 years</small><strong className={risk.floodEventChance.twentyYearPercent === null ? "event-value-unavailable" : undefined}>{risk.floodEventChance.twentyYearPercent === null ? "Unavailable" : `${risk.floodEventChance.kind === "upper_bound" ? "<" : ""}${risk.floodEventChance.twentyYearPercent.toFixed(1)}%`}</strong></div>
                  </div>
                  <p>{risk.floodEventChance.basis}</p>
                  {risk.floodEventChance.tenYearPercent === null && <p>A blank map benchmark does not mean zero flood risk. Check the current FEMA map for this location.</p>}
                </div>
                <div className="current-score-panel">
                  <span className="event-eyebrow">Terrain model pattern score</span>
                  <label className="model-select-label" htmlFor="terrain-model">Choose model</label>
                  <select
                    id="terrain-model"
                    className="model-select"
                    value={selectedModel}
                    onChange={(event) => setSelectedModel(event.target.value === "terrainDnn" ? "terrainDnn" : event.target.value === "cnn" ? "cnn" : "boosted")}
                  >
                    <option value="boosted">Terrain boosted · 2 classes</option>
                    <option value="terrainDnn">Terrain DNN · 2 classes</option>
                    <option value="cnn">Terrain CNN · 2 classes</option>
                  </select>
                  {selectedModel === "boosted" ? (
                    <>
                      <p>{risk.model ? `${risk.model.scoreOutOf100} / 100 — similarity to the model's higher-risk training pattern, not a flood percentage.` : "Unavailable because local elevation or terrain data could not be sampled."}</p>
                      {risk.model && <p>Predicted class: <strong>{risk.model.classification}</strong></p>}
                      {risk.modelMapDisagreement && <p className="model-map-disagreement">{risk.modelMapDisagreement}</p>}
                      {risk.model && risk.model.missingFeatures.length > 0 && <p>Some current inputs were unavailable, so this score may be less reliable.</p>}
                    </>
                  ) : selectedModel === "cnn" ? (
                    cnnLoading ? <p>Reading the surrounding terrain grid and running the CNN…</p>
                    : cnnError ? <p>{cnnError}</p>
                    : cnnModel ? (
                      <>
                        <p>{(cnnModel.onePercentPatternScore * 100).toFixed(1)} / 100 — similarity to the FEMA 1% training pattern, not a flood percentage.</p>
                        <p>Predicted class: <strong>{cnnModel.classification}</strong></p>
                        <p>This research CNN compares FEMA Minimal and 1% classes. The 0.2% class was excluded from its training.</p>
                        {cnnModel.imputedFeatures.length > 0 && <p>Some point inputs were unavailable and imputed, so this score may be less reliable.</p>}
                        {(cnnModel.topographyCoverage < 0.8 || cnnModel.imperviousnessCoverage < 0.8) && <p>Part of the surrounding grid lacked data, so this score may be less reliable.</p>}
                      </>
                    ) : <p>The CNN score is unavailable at this location.</p>
                  ) : selectedDnnModel ? (
                    <>
                      <p>{selectedDnnModel.scoreOutOf100} / 100 — similarity to the FEMA 1% training pattern, not a flood percentage.</p>
                      <p>Predicted class: <strong>{selectedDnnModel.classification}</strong></p>
                      <p>This DNN compares FEMA minimal and 1% classes. The 0.2% class was excluded from its training.</p>
                      {selectedDnnModel.imputedFeatures.length > 0 && <p>Some current inputs were unavailable and imputed, so this score may be less reliable.</p>}
                    </>
                  ) : <p>Unavailable because local elevation or terrain data could not be sampled.</p>}
                </div>
                <details className="result-details input-details">
                  <summary>Show model input data <span aria-hidden="true">+</span></summary>
                  <p>These ten point values are sampled for this location and shared by the terrain models. The CNN also reads a 64 × 64 grid with terrain and imperviousness pixels and valid-data masks.</p>
                  <dl className="input-list">
                    <div><dt>Elevation</dt><dd>{formatInput(risk.modelInputs.elevationMeters, " m")}</dd></div>
                    <div><dt>Impervious surface</dt><dd>{formatInput(risk.modelInputs.imperviousnessPercent, "%")}</dd></div>
                    <div><dt>100-year, 24-hour rainfall</dt><dd>{formatInput(risk.modelInputs.rainfallMillimeters, " mm")}</dd></div>
                    <div><dt>Mapped water nearby</dt><dd>{risk.modelInputs.nearestWaterAvailable === null ? "Unavailable" : risk.modelInputs.nearestWaterAvailable === 1 ? "Yes" : "No"}</dd></div>
                    <div><dt>Distance to nearest mapped water</dt><dd>{formatInput(risk.modelInputs.nearestWaterDistanceMeters, " m")}</dd></div>
                    <div><dt>Terrain slope</dt><dd>{formatInput(risk.modelInputs.slopeDegrees, "°")}</dd></div>
                    <div><dt>Local elevation (250 m)</dt><dd>{formatInput(risk.modelInputs.localElevation250Meters, " m")}</dd></div>
                    <div><dt>Local elevation (500 m)</dt><dd>{formatInput(risk.modelInputs.localElevation500Meters, " m")}</dd></div>
                    <div><dt>Flow accumulation</dt><dd>{formatInput(risk.modelInputs.flowAccumulationSquareMeters, " m²")}</dd></div>
                    <div><dt>Height above drainage</dt><dd>{formatInput(risk.modelInputs.heightAboveDrainageMeters, " m")}</dd></div>
                  </dl>
                  <p>Rainfall is a NOAA design-storm value, not observed rainfall for a specific event. Models transform and standardize some values internally.</p>
                </details>
                <p className="disclaimer">{risk.caveat}</p>
              </>
            )}
          </section>
        )}

        <details className="result-details">
          <summary>About this analysis <span aria-hidden="true">+</span></summary>
          <p>The current annual, 10-year, and 20-year percentages use FEMA flood-zone thresholds. In unshaded Zone X, the less-than values are upper bounds for the mapped hazard. They do not predict whether a building floods. The boosted terrain model, two-class terrain DNN, and two-class terrain CNN score similarity to the FEMA 1% training pattern. The 0.2% class was excluded from their training. These model outputs are not flood probabilities.</p>
        </details>
        <footer className="panel-footer">Built for exploration. Informed by local data.</footer>
      </section>
      <section className="map-workspace" aria-label="Interactive Bay Area map">
        <div className="map-toolbar">
          <div><span className="coverage-dot" /><strong>Bay Area</strong><span className="map-subtitle">California, United States</span></div>
          <button onClick={() => map.current?.fitBounds([[-123.1, 36.95], [-121.0, 38.9]], { padding: 36 })} className="reset-view" aria-label="Reset map to Bay Area"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 9V4h5M20 15v5h-5M4 4l6 6m10 10-6-6" /></svg>Reset view</button>
        </div>
        <div className="map-canvas-wrap">
          <div ref={mapContainer} className="map" aria-label="Bay Area flood risk map" />
          <div className="map-hint"><span className="hint-icon" aria-hidden="true">+</span><span><strong>Select a point to explore</strong><small>Click the map for a mapped flood-event benchmark</small></span></div>
        </div>
        <div className="map-bottom-bar"><span>Geographic context</span><span>FEMA mapped benchmark</span></div>
      </section>
      </div>
    </main>
  );
}
