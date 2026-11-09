"""Build a FEMA map-class point sample from Sacramento County polygons.

This is an offline map-label sample. It does not contain observed inundation or
the physical features required by the Bay Area model.
"""

import csv
import hashlib
import json
from pathlib import Path
import random
from datetime import datetime, timezone

from osgeo import gdal, ogr, osr

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FEMAdata/sacramento_county_100_500_year_floodplain.geojson"
OUTPUT = ROOT / "data/sacramento-fema-map-class-sample.csv"
MANIFEST = ROOT / "data/sacramento-fema-map-class-sample.source.json"
PER_CLASS = 1000
MIN_BOUNDARY_DISTANCE_M = 30
SEED = 20261105
CLASSES = ("minimal", "point_two_percent", "one_percent")
NLCD = ROOT / "data/nlcd/fractional-impervious-2025-full.tiff"
RAINFALL = ROOT / "rainfall/rainfall_data.tif"


def mapped_class(properties):
    zone = (properties.get("FLD_ZONE") or "").strip().upper()
    subtype = (properties.get("ZONE_SUBTY") or "").strip().upper()
    if subtype == "AREA OF MINIMAL FLOOD HAZARD":
        return "minimal"
    if subtype == "0.2 PCT ANNUAL CHANCE FLOOD HAZARD":
        return "point_two_percent"
    if properties.get("SFHA_TF") == "T" or zone in {"A", "A99", "AE", "AH", "AO"}:
        return "one_percent"
    if zone == "X" and subtype == "1 PCT DEPTH LESS THAN 1 FOOT":
        return "one_percent"
    return None


def spatial_references():
    geographic = osr.SpatialReference()
    geographic.ImportFromEPSG(4326)
    geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    projected = osr.SpatialReference()
    projected.ImportFromEPSG(3310)
    projected.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return osr.CoordinateTransformation(geographic, projected), osr.CoordinateTransformation(projected, geographic)


class RasterSampler:
    def __init__(self, path):
        self.dataset = gdal.Open(str(path))
        if self.dataset is None:
            raise RuntimeError(f"Cannot open raster {path}")
        geographic = osr.SpatialReference()
        geographic.ImportFromEPSG(4326)
        geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        raster_srs = osr.SpatialReference(wkt=self.dataset.GetProjection())
        raster_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.transform = osr.CoordinateTransformation(geographic, raster_srs)
        self.inverse_geotransform = gdal.InvGeoTransform(self.dataset.GetGeoTransform())
        self.band = self.dataset.GetRasterBand(1)
        self.nodata = self.band.GetNoDataValue()

    def sample(self, longitude, latitude):
        x, y, _ = self.transform.TransformPoint(longitude, latitude)
        transform = self.inverse_geotransform
        column = int(transform[0] + x * transform[1] + y * transform[2])
        row = int(transform[3] + x * transform[4] + y * transform[5])
        if not (0 <= column < self.dataset.RasterXSize and 0 <= row < self.dataset.RasterYSize):
            raise RuntimeError("Point lies outside source raster")
        value = float(self.band.ReadAsArray(column, row, 1, 1)[0, 0])
        if value == self.nodata:
            raise RuntimeError("Point has no source raster value")
        return value


def main():
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    to_projected, to_geographic = spatial_references()
    impervious = RasterSampler(NLCD)
    rainfall = RasterSampler(RAINFALL)
    polygons = {label: [] for label in CLASSES}
    for feature in source["features"]:
        label = mapped_class(feature["properties"])
        if label is None:
            continue
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"]))
        geometry.Transform(to_projected)
        area = geometry.GetArea()
        if area > 0:
            polygons[label].append((area, geometry, feature["properties"]))

    rng = random.Random(SEED)
    rows = []
    seen = set()
    for label in CLASSES:
        candidates = polygons[label]
        if not candidates:
            raise RuntimeError(f"No source polygons for {label}")
        weights = [item[0] for item in candidates]
        created = 0
        attempts = 0
        while created < PER_CLASS and attempts < PER_CLASS * 1000:
            attempts += 1
            _, polygon, properties = rng.choices(candidates, weights=weights, k=1)[0]
            west, east, south, north = polygon.GetEnvelope()
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint_2D(rng.uniform(west, east), rng.uniform(south, north))
            if not polygon.Contains(point) or polygon.Boundary().Distance(point) < MIN_BOUNDARY_DISTANCE_M:
                continue
            geographic = point.Clone()
            geographic.Transform(to_geographic)
            longitude, latitude, _ = geographic.GetPoint()
            key = (round(longitude, 6), round(latitude, 6))
            if key in seen:
                continue
            seen.add(key)
            created += 1
            rows.append({
                "sample_id": f"sac-fema-{len(rows) + 1:05d}",
                "county": "Sacramento County",
                "lat": f"{latitude:.6f}",
                "lng": f"{longitude:.6f}",
                "mapped_hazard_class": label,
                "current_imperviousness_pct": f"{impervious.sample(longitude, latitude):.1f}",
                "rainfall_100yr_24h_in": f"{rainfall.sample(longitude, latitude) / 1000:.2f}",
                "flood_zone_raw": properties.get("FLD_ZONE") or "",
                "zone_subtype": (properties.get("ZONE_SUBTY") or "").strip(),
                "source_object_id": properties["OBJECTID"],
                "label_source": "Sacramento County FEMA 100 and 500 Year Floodplain map layer",
            })
        if created != PER_CLASS:
            raise RuntimeError(f"Only found {created}/{PER_CLASS} interior points for {label}")

    headers = list(rows[0])
    with OUTPUT.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    MANIFEST.write_text(json.dumps({
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_url": "https://mapservices.gis.saccounty.net/arcgis/rest/services/DWR_HYDROLOGY/MapServer/7",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "points_per_class": PER_CLASS,
        "minimum_boundary_distance_m": MIN_BOUNDARY_DISTANCE_M,
        "feature_sources": {
            "current_imperviousness_pct": str(NLCD.relative_to(ROOT)),
            "rainfall_100yr_24h_in": str(RAINFALL.relative_to(ROOT)),
        },
        "sampling": "Area-weighted polygon selection, then uniform point within polygon; 30 m boundary exclusion",
        "limitations": "FEMA-derived map classes and two static raster features; no observed event outcomes, elevation, water distance, or terrain features.",
    }, indent=2) + "\n")
    print(f"Wrote {len(rows)} map-class points to {OUTPUT}")


if __name__ == "__main__":
    main()
