"""Fetch Sacramento County's public FEMA 100/500-year floodplain layer.

The output is source map geometry, not observed flood-event outcomes or a
training table. Run: python3 scripts/fetch-sacramento-flood-zones.py
"""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from datetime import datetime, timezone
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
LAYER = "https://mapservices.gis.saccounty.net/arcgis/rest/services/DWR_HYDROLOGY/MapServer/7"
OUTPUT = ROOT / "FEMAdata/sacramento_county_100_500_year_floodplain.geojson"
MANIFEST = ROOT / "FEMAdata/sacramento_county_100_500_year_floodplain.source.json"
FIELDS = "OBJECTID,FLD_ZONE,ZONE_SUBTY,SFHA_TF,FLD_AR_ID,DFIRM_ID,VERSION_ID,SOURCE_CIT"
PAGE_SIZE = 50


def request_json(url, params):
    endpoint = f"{url}?{urlencode(params)}"
    for attempt in range(4):
        try:
            with urlopen(endpoint, timeout=45) as response:
                result = json.load(response)
            if "error" in result:
                raise RuntimeError(f"ArcGIS error: {result['error']}")
            return result
        except (OSError, ValueError) as error:
            if attempt == 3:
                raise RuntimeError(f"Unable to fetch {endpoint}") from error
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def main():
    metadata = request_json(LAYER, {"f": "json"})
    if metadata.get("name") != "FEMA 100 and 500 Year Floodplain":
        raise RuntimeError("Unexpected ArcGIS layer; refusing to write dataset")

    count = request_json(
        f"{LAYER}/query", {"where": "1=1", "returnCountOnly": "true", "f": "json"}
    )["count"]
    def fetch_page(offset):
        page = request_json(
            f"{LAYER}/query",
            {
                "where": "1=1",
                "outFields": FIELDS,
                "returnGeometry": "true",
                "outSR": 4326,
                "geometryPrecision": 6,
                "maxAllowableOffset": 0.00001,
                "orderByFields": "OBJECTID ASC",
                "resultOffset": offset,
                "resultRecordCount": PAGE_SIZE,
                "f": "geojson",
            },
        )
        batch = page.get("features")
        if not isinstance(batch, list) or len(batch) != min(PAGE_SIZE, count - offset):
            raise RuntimeError(f"Incomplete page at offset {offset}")
        return batch

    features = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        pages = [executor.submit(fetch_page, offset) for offset in range(0, count, PAGE_SIZE)]
        for completed in as_completed(pages):
            features.extend(completed.result())
            if len(features) % 500 == 0 or len(features) == count:
                print(f"Fetched {len(features)}/{count} features", flush=True)

    features.sort(key=lambda feature: feature["properties"]["OBJECTID"])

    ids = [feature["properties"]["OBJECTID"] for feature in features]
    if len(ids) != len(set(ids)) or len(ids) != count:
        raise RuntimeError("Feature IDs are duplicated or the source changed during download")
    if any(not feature.get("geometry") for feature in features):
        raise RuntimeError("A feature is missing geometry")

    collection = {"type": "FeatureCollection", "features": features}
    payload = (json.dumps(collection, separators=(",", ":")) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = {
        "source": LAYER,
        "layer_name": metadata["name"],
        "description": "Sacramento County FEMA 100 and 500 Year Floodplain map layer",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "feature_count": count,
        "output_crs": "EPSG:4326",
        "geometry_precision_decimal_places": 6,
        "maximum_geometry_offset_degrees": 0.00001,
        "fields": FIELDS.split(","),
        "sha256": digest,
        "limitations": "Mapped hazard polygons generalized by up to 0.00001 degrees for download; not historical inundation observations or verified-dry labels.",
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".geojson.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, OUTPUT)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {count} features to {OUTPUT} ({len(payload):,} bytes; sha256 {digest})")


if __name__ == "__main__":
    main()
