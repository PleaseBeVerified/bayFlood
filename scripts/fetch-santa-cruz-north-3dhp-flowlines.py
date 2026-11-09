"""Fetch official USGS 3DHP flowlines around the Santa Cruz pilot area."""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
LAYER = "https://3dhp.nationalmap.gov/arcgis/rest/services/usgs_3dhp_all/FeatureServer/50"
BOUNDS = (-122.23, 36.94, -121.92, 37.22)  # Study area plus more than 5 km.
OUTPUT = ROOT / "hydrography/santa_cruz_north_3dhp_flowlines.geojson"
MANIFEST = ROOT / "hydrography/santa_cruz_north_3dhp_flowlines.source.json"
FIELDS = "OBJECTID,gnisidlabel,featuretypelabel"
PAGE_SIZE = 1000


def request_json(url, params):
    request = Request(f"{url}?{urlencode(params)}", headers={"User-Agent": "BayFlood hydrography audit"})
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    if "error" in result:
        raise RuntimeError(f"USGS 3DHP ArcGIS error: {result['error']}")
    return result


def main():
    metadata = request_json(LAYER, {"f": "json"})
    if metadata.get("name") != "Flowline":
        raise RuntimeError("Unexpected USGS 3DHP layer")
    query = {
        "where": "1=1",
        "geometry": ",".join(map(str, BOUNDS)),
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": FIELDS,
        "outSR": 4326,
        "geometryPrecision": 7,
        "orderByFields": "OBJECTID ASC",
        "f": "geojson",
    }
    count = request_json(f"{LAYER}/query", {**query, "f": "json", "returnCountOnly": "true"})["count"]
    features = []
    for offset in range(0, count, PAGE_SIZE):
        page = request_json(f"{LAYER}/query", {
            **query, "resultOffset": offset, "resultRecordCount": PAGE_SIZE,
        })
        batch = page.get("features", [])
        if len(batch) != min(PAGE_SIZE, count - offset):
            raise RuntimeError(f"Incomplete 3DHP page {offset}: {len(batch)} features")
        features.extend(batch)
    features.sort(key=lambda item: item["properties"]["OBJECTID"])
    ids = [item["properties"]["OBJECTID"] for item in features]
    if len(ids) != len(set(ids)) or any(not item.get("geometry") for item in features):
        raise RuntimeError("Duplicate or missing 3DHP flowline geometry")
    payload = (json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "dataset_kind": "USGS 3D Hydrography Program flowlines",
        "source_url": LAYER,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "query_bounds_wgs84": BOUNDS,
        "feature_count": count,
        "output_crs": "EPSG:4326",
        "geometry_precision_decimal_places": 7,
        "source_fields": FIELDS.split(","),
        "sha256": digest,
        "limitations": "Flowlines intersecting the query window are included with whole geometries; this is local hydrography coverage for the pilot, not event-specific water extent.",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".geojson.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, OUTPUT)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {count} USGS 3DHP flowlines ({len(payload):,} bytes; sha256 {digest})")


if __name__ == "__main__":
    main()
