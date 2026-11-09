"""Fetch FEMA NFHL polygons and USGS 3DHP flowlines for a San Joaquin holdout.

The source files are mapped-hazard inputs, not observed flood-event outcomes.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
STUDY_BOUNDS = (-121.55, 37.68, -121.15, 37.98)
HYDRO_BOUNDS = (-121.62, 37.61, -121.08, 38.05)
DFIRM_ID = "06077C"
FEMA_LAYER = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28"
HYDRO_LAYER = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/50"
FEMA_OUTPUT = ROOT / "FEMAdata/tracy_lathrop_stockton_nfhl_flood_zones.geojson"
HYDRO_OUTPUT = ROOT / "hydrography/tracy_lathrop_stockton_3dhp_flowlines.geojson"
FEMA_FIELDS = "OBJECTID,DFIRM_ID,VERSION_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,FLD_AR_ID,SOURCE_CIT"
HYDRO_FIELDS = "OBJECTID,gnisidlabel,featuretypelabel"
PAGE_SIZE = 1000


def request_json(url, params):
    request = Request(
        f"{url}?{urlencode(params)}",
        headers={"User-Agent": "BayFlood external geographic test"},
    )
    with urlopen(request, timeout=90) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(f"ArcGIS error: {result['error']}")
    return result


def save_source(path, features, manifest):
    features.sort(key=lambda item: item["properties"]["OBJECTID"])
    ids = [item["properties"]["OBJECTID"] for item in features]
    if len(ids) != len(set(ids)) or any(not item.get("geometry") for item in features):
        raise RuntimeError(f"Duplicate ID or missing geometry in {path.name}")
    payload = (
        json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"))
        + "\n"
    ).encode()
    manifest.update({
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": len(features),
        "output_crs": "EPSG:4326",
        "geometry_precision_decimal_places": 7,
        "sha256": hashlib.sha256(payload).hexdigest(),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".geojson.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    path.with_suffix(".source.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved {len(features)} features to {path.relative_to(ROOT)}", flush=True)


def fetch_fema():
    metadata = request_json(FEMA_LAYER, {"f": "json"})
    if metadata.get("name") != "Flood Hazard Zones":
        raise RuntimeError("Unexpected FEMA NFHL layer")
    query = {
        "where": f"DFIRM_ID='{DFIRM_ID}'",
        "geometry": ",".join(map(str, STUDY_BOUNDS)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": FEMA_FIELDS,
        "outSR": 4326,
        "geometryPrecision": 7,
        "orderByFields": "OBJECTID ASC",
        "f": "geojson",
    }
    count = request_json(f"{FEMA_LAYER}/query", {
        **query, "f": "json", "returnCountOnly": "true",
    })["count"]
    if count > metadata.get("maxRecordCount", 0):
        raise RuntimeError(f"FEMA has {count} polygons; add pagination before downloading")
    features = request_json(f"{FEMA_LAYER}/query", query).get("features", [])
    if len(features) != count or any(f["properties"].get("DFIRM_ID") != DFIRM_ID
                                     for f in features):
        raise RuntimeError(f"Incomplete or mixed-county FEMA response: {len(features)}/{count}")
    save_source(FEMA_OUTPUT, features, {
        "dataset_kind": "FEMA NFHL mapped flood-zone polygons",
        "source_url": FEMA_LAYER,
        "source_layer": metadata["name"],
        "study_bounds_wgs84": STUDY_BOUNDS,
        "dfirm_id": DFIRM_ID,
        "source_fields": FEMA_FIELDS.split(","),
        "limitations": "Mapped flood hazard, not observed event inundation. Features intersect the window but are not clipped to it.",
    })


def fetch_hydrography():
    metadata = request_json(HYDRO_LAYER, {"f": "json"})
    if metadata.get("name") != "Flowline":
        raise RuntimeError("Unexpected USGS 3DHP layer")
    query = {
        "where": "1=1",
        "geometry": ",".join(map(str, HYDRO_BOUNDS)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": HYDRO_FIELDS,
        "outSR": 4326,
        "geometryPrecision": 7,
        "orderByFields": "OBJECTID ASC",
        "f": "geojson",
    }
    count = request_json(f"{HYDRO_LAYER}/query", {
        **query, "f": "json", "returnCountOnly": "true",
    })["count"]
    features = []
    for offset in range(0, count, PAGE_SIZE):
        page = request_json(f"{HYDRO_LAYER}/query", {
            **query, "resultOffset": offset, "resultRecordCount": PAGE_SIZE,
        })
        batch = page.get("features", [])
        if len(batch) != min(PAGE_SIZE, count - offset):
            raise RuntimeError(f"Incomplete 3DHP page at offset {offset}")
        features.extend(batch)
    save_source(HYDRO_OUTPUT, features, {
        "dataset_kind": "USGS 3D Hydrography Program flowlines",
        "source_url": HYDRO_LAYER,
        "query_bounds_wgs84": HYDRO_BOUNDS,
        "source_fields": HYDRO_FIELDS.split(","),
        "limitations": "Local water proximity source; not a flood-event extent. Whole flowlines intersecting the query window are included.",
    })


def main():
    fetch_fema()
    fetch_hydrography()


if __name__ == "__main__":
    main()
