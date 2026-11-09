"""Fetch USGS river measurements for the 2022–23 Sacramento storm period."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
BASE = "https://waterservices.usgs.gov/nwis/iv/"
PARAMETERS = {
    "format": "json",
    "sites": "11335000,11446500,11447650",
    "startDT": "2022-12-25",
    "endDT": "2023-01-20",
    "parameterCd": "00060,00065",
    "siteStatus": "all",
}
URL = f"{BASE}?{urlencode(PARAMETERS)}"
RAW = ROOT / "data/cache/sacramento-usgs-river-observations.json"
OUTPUT = ROOT / "data/sacramento-river-observations-2022-2023.csv"
MANIFEST = ROOT / "data/sacramento-river-observations-2022-2023.source.json"


def main():
    with urlopen(URL, timeout=120) as response:
        raw = response.read()
    document = json.loads(raw)
    series = document.get("value", {}).get("timeSeries", [])
    if len(series) < 6:
        raise RuntimeError(f"Expected six site/parameter series; found {len(series)}")
    rows = []
    for item in series:
        site = item["sourceInfo"]
        site_id = site["siteCode"][0]["value"]
        location = site["geoLocation"]["geogLocation"]
        parameter = item["variable"]["variableCode"][0]["value"]
        unit = item["variable"]["unit"]["unitCode"]
        for values in item["values"]:
            for observation in values["value"]:
                if observation["value"] in {"", "-999999"}:
                    continue
                rows.append({
                    "event_id": "sacramento-storms-2022-12-2023-01",
                    "site_id": site_id,
                    "site_name": site["siteName"],
                    "lat": location["latitude"],
                    "lng": location["longitude"],
                    "observed_at": observation["dateTime"],
                    "parameter_code": parameter,
                    "parameter": "discharge" if parameter == "00060" else "gage_height",
                    "value": observation["value"],
                    "unit": unit,
                    "qualifiers": ";".join(observation.get("qualifiers", [])),
                })
    rows.sort(key=lambda row: (row["site_id"], row["observed_at"], row["parameter_code"]))
    RAW.parent.mkdir(parents=True, exist_ok=True)
    RAW.write_bytes(raw)
    with OUTPUT.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    MANIFEST.write_text(json.dumps({
        "source_url": URL,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "series_count": len(series),
        "observation_count": len(rows),
        "site_ids": sorted({row["site_id"] for row in rows}),
        "parameter_codes": ["00060 (discharge)", "00065 (gage height)"],
        "limitations": "River measurements are event context at gauges, not location-level inundation labels.",
    }, indent=2) + "\n")
    print(f"Wrote {len(rows)} river observations to {OUTPUT}")


if __name__ == "__main__":
    main()
