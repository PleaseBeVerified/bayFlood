"""Extract saved CNN raster inputs for a supported external holdout.

Run with the Homebrew Python that provides GDAL. This writes an ignored cache;
the CSV and FEMA labels are unchanged.
"""

import csv
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from osgeo import gdal, osr


gdal.UseExceptions()
osr.UseExceptions()
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/cache/terrain-cnn"
TERRAIN = ROOT / "terrain/runtime/terrain-features-30m.tif"
DEM = CACHE / "dem-30m.tif"
NLCD = CACHE / "imperviousness-30m.tif"
REGIONS = ("santa-cruz-north", "tracy-lathrop-stockton")
SIZE = 64
NODATA = -9999.0
CHANNELS = [
    "slope_degrees", "local_elevation_250m", "local_elevation_500m",
    "log_flow_accumulation_m2", "height_above_drainage_m",
    "dem_relative_to_point_m", "imperviousness_pct",
    "topography_valid", "imperviousness_valid",
]


def window(dataset, column, row):
    half = SIZE // 2
    left, top = column - half, row - half
    right, bottom = left + SIZE, top + SIZE
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(right, dataset.RasterXSize), min(bottom, dataset.RasterYSize)
    result = np.full((dataset.RasterCount, SIZE, SIZE), NODATA, dtype=np.float32)
    if x1 > x0 and y1 > y0:
        pixels = dataset.ReadAsArray(x0, y0, x1 - x0, y1 - y0)
        if pixels.ndim == 2:
            pixels = pixels[None, ...]
        result[:, y0 - top:y1 - top, x0 - left:x1 - left] = pixels
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--region", choices=REGIONS, default="santa-cruz-north")
    parser.add_argument("--three-class", action="store_true")
    arguments = parser.parse_args()
    three_class = arguments.three_class
    dataset = ROOT / f"data/{arguments.region}-fema-map-class-terrain-sample.csv"
    with dataset.open(newline="", encoding="utf-8") as source:
        rows = [row for row in csv.DictReader(source)
                if three_class or row["mapped_hazard_class"] in {"minimal", "one_percent"}]
    expected_count = 450 if three_class else 300
    if len(rows) != expected_count or {row["spatial_partition"] for row in rows} != {"external_test"}:
        raise RuntimeError(f"Expected {expected_count} {arguments.region} holdout rows")
    output_dir = CACHE / "three-class" if three_class else CACHE
    patch_path = output_dir / f"{arguments.region}-external-patches-float16.npy"
    metadata_path = output_dir / f"{arguments.region}-external-patches.json"
    rasters = [gdal.Open(str(path)) for path in (TERRAIN, DEM, NLCD)]
    if any(dataset is None for dataset in rasters):
        raise FileNotFoundError("CNN aligned rasters are missing; run npm run model:cnn:rasters")
    terrain, dem, nlcd = rasters
    if any(dataset.RasterXSize != terrain.RasterXSize
           or dataset.RasterYSize != terrain.RasterYSize
           or dataset.GetGeoTransform() != terrain.GetGeoTransform()
           for dataset in (dem, nlcd)):
        raise RuntimeError("CNN raster grids are not aligned")
    geographic = osr.SpatialReference()
    geographic.ImportFromEPSG(4326)
    geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    projected = osr.SpatialReference()
    projected.ImportFromEPSG(3310)
    projected.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    project = osr.CoordinateTransformation(geographic, projected)
    transform = terrain.GetGeoTransform()
    patches = np.zeros((len(rows), len(CHANNELS), SIZE, SIZE), dtype=np.float16)
    metadata = []
    for index, item in enumerate(rows):
        x, y, _ = project.TransformPoint(float(item["lng"]), float(item["lat"]))
        column = math.floor((x - transform[0]) / transform[1])
        row = math.floor((y - transform[3]) / transform[5])
        terrain_pixels = window(terrain, column, row)
        dem_pixels = window(dem, column, row)[0]
        nlcd_pixels = window(nlcd, column, row)[0]
        topography_valid = np.all(np.isfinite(terrain_pixels) & (terrain_pixels != NODATA), axis=0)
        topography_valid &= np.isfinite(dem_pixels) & (dem_pixels != NODATA)
        imperviousness_valid = np.isfinite(nlcd_pixels) & (nlcd_pixels >= 0) & (nlcd_pixels <= 100)
        if not topography_valid[SIZE // 2, SIZE // 2]:
            raise RuntimeError(f"Missing central terrain pixel: {item['sample_id']}")
        data = np.zeros((len(CHANNELS), SIZE, SIZE), dtype=np.float32)
        data[:5, topography_valid] = terrain_pixels[:, topography_valid]
        data[3, topography_valid] = np.log1p(np.maximum(data[3, topography_valid], 0))
        data[5, topography_valid] = dem_pixels[topography_valid] - float(item["elevation_m"])
        data[6, imperviousness_valid] = nlcd_pixels[imperviousness_valid]
        data[7] = topography_valid
        data[8] = imperviousness_valid
        patches[index] = data.astype(np.float16)
        metadata.append({
            "sample_id": item["sample_id"],
            "mapped_hazard_class": item["mapped_hazard_class"],
            "topography_coverage": float(topography_valid.mean()),
            "imperviousness_coverage": float(imperviousness_valid.mean()),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(patch_path, patches)
    metadata_path.write_text(json.dumps({
        "schemaVersion": 1,
        "dataset": str(dataset.relative_to(ROOT)),
        "datasetSha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "patchSize": SIZE,
        "resolutionMeters": 30,
        "channels": CHANNELS,
        "rows": metadata,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} CNN patches; minimum topography coverage "
          f"{min(row['topography_coverage'] for row in metadata):.3f}")


if __name__ == "__main__":
    main()
