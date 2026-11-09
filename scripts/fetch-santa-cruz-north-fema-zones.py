"""Fetch FEMA NFHL flood-zone polygons for the northern Santa Cruz pilot.

Run: python3 scripts/fetch-santa-cruz-north-fema-zones.py
The GeoJSON is a local source cache, not observed flood-event data.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LAYER = "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28"
BOUNDS = (-122.16, 37.005, -121.99, 37.155)
DFIRM_ID = "06087C"  # Santa Cruz County
OUTPUT = ROOT / "FEMAdata/santa_cruz_north_nfhl_flood_zones.geojson"
MANIFEST = ROOT / "FEMAdata/santa_cruz_north_nfhl_flood_zones.source.json"
FIELDS = "OBJECTID,DFIRM_ID,VERSION_ID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,FLD_AR_ID,SOURCE_CIT"


def request_json(url, params):
    request = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "BayFlood research dataset builder"})
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(f"ArcGIS error: {result['error']}")
    return result


def main():
    metadata = request_json(LAYER, {"f": "json"})
    if metadata.get("name") != "Flood Hazard Zones":
        raise RuntimeError("Unexpected FEMA NFHL layer")
    params = {
        "where": f"DFIRM_ID='{DFIRM_ID}'",
        "geometry": ",".join(str(value) for value in BOUNDS),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": FIELDS,
        "outSR": 4326,
        "geometryPrecision": 7,
        "orderByFields": "OBJECTID ASC",
        "f": "geojson",
    }
    count = request_json(f"{LAYER}/query", {**params, "returnCountOnly": "true", "f": "json"})["count"]
    if count > metadata.get("maxRecordCount", 0):
        raise RuntimeError(f"Source has {count} features; implement pagination before downloading")
    source = request_json(f"{LAYER}/query", params)
    features = source.get("features", [])
    if len(features) != count or any(not item.get("geometry") for item in features):
        raise RuntimeError(f"Incomplete FEMA response: {len(features)}/{count} polygons")
    if any(item["properties"].get("DFIRM_ID") != DFIRM_ID for item in features):
        raise RuntimeError("Response includes another county map")
    features.sort(key=lambda item: item["properties"]["OBJECTID"])
    ids = [item["properties"]["OBJECTID"] for item in features]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate FEMA OBJECTID")

    payload = (json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "dataset_kind": "source FEMA NFHL mapped flood-zone polygons",
        "source_url": LAYER,
        "source_layer": metadata["name"],
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "study_bounds_wgs84": BOUNDS,
        "dfirm_id": DFIRM_ID,
        "feature_count": count,
        "output_crs": "EPSG:4326",
        "geometry_precision_decimal_places": 7,
        "source_fields": FIELDS.split(","),
        "sha256": digest,
        "limitations": "FEMA mapped hazard zones, not observed inundation or verified flood-event outcomes. Features intersect the study window but geometry is not clipped to it.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".geojson.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, OUTPUT)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {count} FEMA polygons ({len(payload):,} bytes; sha256 {digest})")


if __name__ == "__main__":
    main()
