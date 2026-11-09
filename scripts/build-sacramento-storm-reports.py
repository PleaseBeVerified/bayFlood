"""Extract Sacramento County flood reports from NOAA Storm Events archives."""

import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
ARCHIVES = {
    2022: "StormEvents_details-ftp_v1.0_d2022_c20260625.csv.gz",
    2023: "StormEvents_details-ftp_v1.0_d2023_c20260323.csv.gz",
}
CACHE = ROOT / "data/cache/noaa-storm-events"
OUTPUT = ROOT / "data/sacramento-reported-floods-2022-2023.csv"
MANIFEST = ROOT / "data/sacramento-reported-floods-2022-2023.source.json"
START = datetime(2022, 12, 25)
END = datetime(2023, 1, 20, 23, 59, 59)


def read_archive(year, filename):
    path = CACHE / f"details-{year}.csv.gz"
    if not path.exists():
        with urlopen(SOURCE_ROOT + filename, timeout=300) as response:
            path.write_bytes(response.read())
    with gzip.open(path, "rt", newline="") as source:
        rows = list(csv.DictReader(source))
    return rows, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    selected = []
    checksums = {}
    for year, filename in ARCHIVES.items():
        rows, checksums[filename] = read_archive(year, filename)
        for row in rows:
            if row["STATE"] != "CALIFORNIA" or row["CZ_TYPE"] != "C" or row["CZ_NAME"] != "SACRAMENTO":
                continue
            if row["EVENT_TYPE"] not in {"Flood", "Flash Flood"}:
                continue
            begin = datetime.strptime(row["BEGIN_DATE_TIME"], "%d-%b-%y %H:%M:%S")
            if not START <= begin <= END:
                continue
            selected.append({
                "noaa_event_id": row["EVENT_ID"],
                "noaa_episode_id": row["EPISODE_ID"],
                "county": "Sacramento County",
                "event_type": row["EVENT_TYPE"],
                "begin_local": begin.isoformat(sep=" "),
                "end_local": datetime.strptime(row["END_DATE_TIME"], "%d-%b-%y %H:%M:%S").isoformat(sep=" "),
                "time_zone": row["CZ_TIMEZONE"],
                "begin_lat": row["BEGIN_LAT"],
                "begin_lng": row["BEGIN_LON"],
                "end_lat": row["END_LAT"],
                "end_lng": row["END_LON"],
                "begin_location": row["BEGIN_LOCATION"],
                "end_location": row["END_LOCATION"],
                "flood_cause": row["FLOOD_CAUSE"],
                "report_source": row["SOURCE"],
                "episode_narrative": row["EPISODE_NARRATIVE"],
                "event_narrative": row["EVENT_NARRATIVE"],
                "source_url": f"https://www.ncei.noaa.gov/access/storm-events-database/?id={row['EVENT_ID']}",
            })
    selected.sort(key=lambda row: (row["begin_local"], row["noaa_event_id"]))
    if not selected:
        raise RuntimeError("No Sacramento flood reports found in the expected window")
    with OUTPUT.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    MANIFEST.write_text(json.dumps({
        "source_urls": [SOURCE_ROOT + filename for filename in ARCHIVES.values()],
        "source_sha256": checksums,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "filter": "California, Sacramento County, Flood or Flash Flood, begin 2022-12-25 through 2023-01-20",
        "report_count": len(selected),
        "limitations": "Storm reports and coordinates describe incidents or affected areas; coordinates are not surveyed inundation footprints or verified point labels. No dry labels.",
    }, indent=2) + "\n")
    print(f"Wrote {len(selected)} NOAA flood reports to {OUTPUT}")


if __name__ == "__main__":
    main()
