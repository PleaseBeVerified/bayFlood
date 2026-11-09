"""Build a model-ready FEMA map-class sample for Tracy–Lathrop–Stockton.

Run fetch-tracy-lathrop-stockton-sources.py first. This is an external geographic
test set for existing models, not observed flood-event data.
"""

import csv
import hashlib
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from osgeo import gdal, ogr, osr


gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FEMAdata/tracy_lathrop_stockton_nfhl_flood_zones.geojson"
OUTPUT = ROOT / "data/tracy-lathrop-stockton-fema-map-class-terrain-sample.csv"
MANIFEST = ROOT / "data/tracy-lathrop-stockton-fema-map-class-terrain-sample.source.json"
BOUNDS = (-121.55, 37.68, -121.15, 37.98)
CLASSES = ("minimal", "point_two_percent", "one_percent")
PER_CLASS = 150
BOUNDARY_BUFFER_M = 10
MIN_POINT_SEPARATION_M = 30
WATER_SEARCH_RADIUS_M = 5000
SEED = 20261108
WATER_FILES = (
    "bay_area_flowlines.geojson", "bay_area_water_bodies.geojson",
    "NHDArea.geojson", "NHDLine.geojson",
)
REGIONAL_FLOWLINES = "tracy_lathrop_stockton_3dhp_flowlines.geojson"


def reference(epsg):
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return srs


GEOGRAPHIC = reference(4326)
PROJECTED = reference(3310)
TO_PROJECTED = osr.CoordinateTransformation(GEOGRAPHIC, PROJECTED)
TO_GEOGRAPHIC = osr.CoordinateTransformation(PROJECTED, GEOGRAPHIC)


def mapped_class(properties):
    zone = (properties.get("FLD_ZONE") or "").strip().upper()
    subtype = (properties.get("ZONE_SUBTY") or "").strip().upper()
    if zone == "X" and subtype == "AREA OF MINIMAL FLOOD HAZARD":
        return "minimal"
    if zone == "X" and subtype == "0.2 PCT ANNUAL CHANCE FLOOD HAZARD":
        return "point_two_percent"
    if properties.get("SFHA_TF") == "T" or zone in {"A", "A99", "AE", "AH", "AO", "VE"}:
        return "one_percent"
    return None


class Raster:
    def __init__(self, path):
        self.path = path
        self.dataset = gdal.Open(str(path))
        if self.dataset is None:
            raise RuntimeError(f"Cannot open {path}")
        srs = osr.SpatialReference(wkt=self.dataset.GetProjection())
        srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.transform = osr.CoordinateTransformation(GEOGRAPHIC, srs)
        self.inverse = gdal.InvGeoTransform(self.dataset.GetGeoTransform())

    def sample(self, longitude, latitude, median_radius=0):
        x, y, _ = self.transform.TransformPoint(longitude, latitude)
        inverse = self.inverse
        column = math.floor(inverse[0] + x * inverse[1] + y * inverse[2])
        row = math.floor(inverse[3] + x * inverse[4] + y * inverse[5])
        if not (0 <= column < self.dataset.RasterXSize and 0 <= row < self.dataset.RasterYSize):
            return None
        x0 = max(0, column - median_radius)
        y0 = max(0, row - median_radius)
        width = min(self.dataset.RasterXSize, column + median_radius + 1) - x0
        height = min(self.dataset.RasterYSize, row + median_radius + 1) - y0
        values = []
        for index in range(1, self.dataset.RasterCount + 1):
            band = self.dataset.GetRasterBand(index)
            array = band.ReadAsArray(x0, y0, width, height)
            nodata = band.GetNoDataValue()
            valid = sorted(float(value) for value in array.flat
                           if math.isfinite(float(value)) and value != nodata)
            if not valid:
                return None
            values.append(valid[len(valid) // 2] if median_radius else valid[0])
        return values


def water_geometries(filenames):
    west, south, east, north = BOUNDS
    margin = 0.07  # More than 5 km at this latitude.
    result = []
    for filename in filenames:
        dataset = ogr.Open(str(ROOT / "hydrography" / filename))
        if dataset is None:
            raise RuntimeError(f"Cannot open water layer {filename}")
        layer = dataset.GetLayer(0)
        layer.SetSpatialFilterRect(west - margin, south - margin, east + margin, north + margin)
        for feature in layer:
            geometry = feature.GetGeometryRef()
            if geometry is None:
                continue
            geometry = geometry.Clone()
            geometry.Transform(TO_PROJECTED)
            result.append((geometry.GetEnvelope(), geometry))
    return result


def nearby_water_distance(point, features):
    x, y, _ = point.GetPoint()
    radius = WATER_SEARCH_RADIUS_M
    nearest = radius + 1
    for (west, east, south, north), geometry in features:
        if x < west - nearest or x > east + nearest or y < south - nearest or y > north + nearest:
            continue
        nearest = min(nearest, geometry.Distance(point))
    return nearest if nearest <= radius else None


def candidate_polygons(source):
    west, south, east, north = BOUNDS
    window = ogr.CreateGeometryFromWkt(
        f"POLYGON(({west} {south},{east} {south},{east} {north},{west} {north},{west} {south}))"
    )
    polygons = {label: [] for label in CLASSES}
    all_polygons = []
    for feature in source["features"]:
        properties = feature["properties"]
        label = mapped_class(properties)
        if label is None:
            continue
        geometry = ogr.CreateGeometryFromJson(json.dumps(feature["geometry"]))
        geometry = geometry.Intersection(window)
        if geometry.IsEmpty():
            continue
        geometry.Transform(TO_PROJECTED)
        all_polygons.append((label, geometry.GetEnvelope(), geometry))
        interior = geometry.Buffer(-BOUNDARY_BUFFER_M)
        if not interior.IsEmpty() and interior.GetArea() > 0:
            polygons[label].append((interior.GetArea(), interior, properties))
    return polygons, all_polygons


def valid_label(point, expected, all_polygons):
    x, y, _ = point.GetPoint()
    matches = set()
    for label, (west, east, south, north), geometry in all_polygons:
        if west <= x <= east and south <= y <= north and geometry.Contains(point):
            matches.add(label)
    return matches == {expected}


def model_inputs(longitude, latitude, elevation_tiles, terrain, impervious, rainfall):
    elevation = next((tile.sample(longitude, latitude, median_radius=2)
                      for tile in elevation_tiles
                      if tile.sample(longitude, latitude) is not None), None)
    slope = terrain.sample(longitude, latitude)
    landcover = impervious.sample(longitude, latitude)
    rain = rainfall.sample(longitude, latitude)
    if not elevation or not slope or not landcover or not rain:
        return None
    if slope[3] < 0 or not 0 <= landcover[0] <= 100 or rain[0] <= 0:
        return None
    return {
        "elevation_m": f"{elevation[0]:.3f}",
        "current_imperviousness_pct": f"{landcover[0]:.1f}",
        "current_100yr_24h_rainfall_mm": f"{rain[0] * 0.0254:.3f}",
        "slope_degrees": f"{slope[0]:.3f}",
        "local_elevation_250m": f"{slope[1]:.3f}",
        "local_elevation_500m": f"{slope[2]:.3f}",
        "flow_accumulation_m2": f"{slope[3]:.3f}",
        "height_above_drainage_m": f"{slope[4]:.3f}",
    }


def main():
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    polygons, all_polygons = candidate_polygons(source)
    terrain = Raster(ROOT / "terrain/runtime/terrain-features-30m.tif")
    impervious = Raster(ROOT / "data/nlcd/fractional-impervious-2025-full.tiff")
    rainfall = Raster(ROOT / "rainfall/rainfall_data.tif")
    elevation_tiles = [Raster(path) for path in sorted((ROOT / "elevation").glob("*.tif"))]
    water = water_geometries((*WATER_FILES, REGIONAL_FLOWLINES))
    rng = random.Random(SEED)
    rows = []
    selected = []
    source_ids = {label: set() for label in CLASSES}
    for label in CLASSES:
        candidates = polygons[label]
        if not candidates:
            raise RuntimeError(f"No interior FEMA polygons for {label}")
        weights = [item[0] for item in candidates]
        created = attempts = 0
        while created < PER_CLASS and attempts < PER_CLASS * 10000:
            attempts += 1
            _, polygon, properties = rng.choices(candidates, weights=weights, k=1)[0]
            west, east, south, north = polygon.GetEnvelope()
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint_2D(rng.uniform(west, east), rng.uniform(south, north))
            if not polygon.Contains(point) or not valid_label(point, label, all_polygons):
                continue
            x, y, _ = point.GetPoint()
            if any((x - px) ** 2 + (y - py) ** 2 < MIN_POINT_SEPARATION_M ** 2
                   for px, py in selected):
                continue
            distance = nearby_water_distance(point, water)
            if distance is None:
                continue
            geographic = point.Clone()
            geographic.Transform(TO_GEOGRAPHIC)
            longitude, latitude, _ = geographic.GetPoint()
            inputs = model_inputs(longitude, latitude, elevation_tiles, terrain, impervious, rainfall)
            if inputs is None:
                continue
            selected.append((x, y))
            created += 1
            source_ids[label].add(properties["OBJECTID"])
            rows.append({
                "sample_id": f"tls-fema-terrain-{len(rows) + 1:05d}",
                "region": "Tracy–Lathrop–Stockton, western San Joaquin County",
                "county": "San Joaquin County",
                "lat": f"{latitude:.6f}",
                "lng": f"{longitude:.6f}",
                "spatial_partition": "external_test",
                "mapped_hazard_class": label,
                **inputs,
                "nearest_water_available": "1",
                "nearest_water_distance_m": str(round(distance)),
                "flood_zone_raw": (properties.get("FLD_ZONE") or "").strip(),
                "zone_subtype": (properties.get("ZONE_SUBTY") or "").strip(),
                "source_object_id": properties["OBJECTID"],
                "source_dfirm_id": properties["DFIRM_ID"],
                "label_source": "FEMA NFHL Flood Hazard Zones mapped class",
            })
        if created != PER_CLASS:
            raise RuntimeError(f"Only found {created}/{PER_CLASS} valid {label} points after {attempts} attempts")
        print(f"{label}: {created} points from {len(source_ids[label])} FEMA polygons ({attempts} attempts)", flush=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "dataset_name": "Tracy–Lathrop–Stockton FEMA map-class terrain sample",
        "dataset_role": "geographically external test set for existing terrain FEMA classifiers",
        "target_column": "mapped_hazard_class",
        "target_meaning": "FEMA NFHL mapped annual-chance class; not observed event flooding",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "source_url": "https://hazards.fema.gov/arcgis/rest/services/public/NFHL/MapServer/28",
        "source_dfirm_id": "06077C",
        "study_bounds_wgs84": BOUNDS,
        "row_count": len(rows),
        "class_counts": dict(Counter(row["mapped_hazard_class"] for row in rows)),
        "distinct_source_polygons_by_class": {label: len(ids) for label, ids in source_ids.items()},
        "seed": SEED,
        "minimum_polygon_boundary_distance_m": BOUNDARY_BUFFER_M,
        "minimum_point_separation_m": MIN_POINT_SEPARATION_M,
        "water_search_radius_m": WATER_SEARCH_RADIUS_M,
        "sampling": "Area-weighted FEMA polygon selection; random point in a 10 m inward buffer; require unique class, at least 30 m between samples, complete local raster values, and a water feature within 5 km in the base-plus-USGS-3DHP network.",
        "supplemental_hydrography_sha256": hashlib.sha256(
            (ROOT / "hydrography" / REGIONAL_FLOWLINES).read_bytes()
        ).hexdigest(),
        "feature_sources": {
            "elevation_m": "elevation/USGS_13_*.tif (5x5 median, matching app)",
            "current_imperviousness_pct": str(impervious.path.relative_to(ROOT)),
            "current_100yr_24h_rainfall_mm": str(rainfall.path.relative_to(ROOT)),
            "terrain_features": str(terrain.path.relative_to(ROOT)),
            "nearest_water_distance_m": [f"hydrography/{filename}" for filename in
                                         (*WATER_FILES, REGIONAL_FLOWLINES)],
        },
        "limitations": "FEMA map labels are not observed floods or verified dry event outcomes. Balanced spatial samples do not represent natural class prevalence. Leveed or unmapped areas must not be silently labeled minimal. Points can cluster by source polygon despite 30 m separation; use this entire region as one geographic holdout, not random row splits.",
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} model-input-complete rows to {OUTPUT}")


if __name__ == "__main__":
    main()
