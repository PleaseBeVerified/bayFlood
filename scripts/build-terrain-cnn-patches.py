#!/usr/bin/env python3
"""Build aligned raster patches around the existing FEMA calibration points.

Run with the Homebrew Python that provides ``osgeo``. The aligned rasters in
data/cache are used by the live CNN; the patch array is only for training.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from osgeo import gdal, osr


ROOT = Path(__file__).resolve().parents[1]
TERRAIN = ROOT / "terrain/runtime/terrain-features-30m.tif"
DEM = ROOT / "terrain/dem-10m.tif"
NLCD = ROOT / "data/nlcd/fractional-impervious-2025-full.tiff"
CACHE = ROOT / "data/cache/terrain-cnn"
TABLES = {
    "binary": ROOT / "data/dnn-training-table.csv",
    "three-class": ROOT / "data/dnn-three-class-training-table.csv",
}
THREE_CLASS_LABELS = {"minimal": 0, "one_percent": 1, "point_two_percent": 2}
PATCH_SIZE = 64
NODATA = -9999.0
CHANNELS = [
    "slope_degrees", "local_elevation_250m", "local_elevation_500m",
    "log_flow_accumulation_m2", "height_above_drainage_m",
    "dem_relative_to_point_m", "imperviousness_pct",
    "topography_valid", "imperviousness_valid",
]


def aligned_raster(source: Path, name: str, reference: gdal.Dataset, resampling: str) -> gdal.Dataset:
    target = CACHE / name
    if not target.exists():
        transform = reference.GetGeoTransform()
        width, height = reference.RasterXSize, reference.RasterYSize
        bounds = [
            transform[0], transform[3] + height * transform[5],
            transform[0] + width * transform[1], transform[3],
        ]
        result = gdal.Warp(
            str(target), str(source), format="GTiff",
            outputBounds=bounds, width=width, height=height,
            dstSRS=reference.GetProjection(), resampleAlg=resampling,
            srcNodata=250 if source == NLCD else None, dstNodata=NODATA,
            outputType=gdal.GDT_Float32,
            creationOptions=["TILED=YES", "COMPRESS=ZSTD", "PREDICTOR=3"],
            multithread=True,
        )
        if result is None:
            raise RuntimeError(f"Could not align {source}")
        result = None
    result = gdal.Open(str(target))
    if result is None or result.RasterXSize != reference.RasterXSize or result.RasterYSize != reference.RasterYSize:
        raise RuntimeError(f"Invalid aligned raster: {target}")
    return result


def window(dataset: gdal.Dataset, column: int, row: int) -> np.ndarray:
    """Read a 64-pixel window, padding outside the source with NoData."""
    half = PATCH_SIZE // 2
    left, top = column - half, row - half
    right, bottom = left + PATCH_SIZE, top + PATCH_SIZE
    x0, y0 = max(left, 0), max(top, 0)
    x1, y1 = min(right, dataset.RasterXSize), min(bottom, dataset.RasterYSize)
    result = np.full((dataset.RasterCount, PATCH_SIZE, PATCH_SIZE), NODATA, dtype=np.float32)
    if x1 > x0 and y1 > y0:
        pixels = dataset.ReadAsArray(x0, y0, x1 - x0, y1 - y0)
        if pixels.ndim == 2:
            pixels = pixels[None, ...]
        result[:, y0 - top:y1 - top, x0 - left:x1 - left] = pixels
    return result


def main() -> None:
    rasters_only = "--rasters-only" in sys.argv[1:]
    three_class = "--three-class" in sys.argv[1:]
    if any(arg not in {"--rasters-only", "--three-class"} for arg in sys.argv[1:]):
        raise ValueError("Supported options are --rasters-only and --three-class")
    if rasters_only and three_class:
        raise ValueError("--rasters-only does not need a class selection")
    table_path = TABLES["three-class" if three_class else "binary"]
    output_dir = CACHE / "three-class" if three_class else CACHE
    for source in (TERRAIN, DEM, NLCD) + (() if rasters_only else (table_path,)):
        if not source.exists():
            raise FileNotFoundError(source)
    CACHE.mkdir(parents=True, exist_ok=True)
    gdal.UseExceptions()
    terrain = gdal.Open(str(TERRAIN))
    dem = aligned_raster(DEM, "dem-30m.tif", terrain, "average")
    nlcd = aligned_raster(NLCD, "imperviousness-30m.tif", terrain, "average")
    if rasters_only:
        print(f"Aligned CNN rasters are ready in {CACHE}")
        return
    with table_path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    output_dir.mkdir(parents=True, exist_ok=True)
    array_path = output_dir / "patches-float16.npy"
    patches = np.lib.format.open_memmap(
        array_path, mode="w+", dtype=np.float16,
        shape=(len(rows), len(CHANNELS), PATCH_SIZE, PATCH_SIZE),
    )
    geographic = osr.SpatialReference()
    geographic.ImportFromEPSG(4326)
    geographic.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    projected = osr.SpatialReference()
    projected.ImportFromEPSG(3310)
    projected.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    project = osr.CoordinateTransformation(geographic, projected)
    transform = terrain.GetGeoTransform()
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
        if not topography_valid[PATCH_SIZE // 2, PATCH_SIZE // 2]:
            raise RuntimeError(f"Missing central terrain pixel: {item['location_id']}")
        data = np.zeros((len(CHANNELS), PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
        data[:5, topography_valid] = terrain_pixels[:, topography_valid]
        data[3, topography_valid] = np.log1p(np.maximum(data[3, topography_valid], 0))
        data[5, topography_valid] = dem_pixels[topography_valid] - float(item["elevation_m"])
        data[6, imperviousness_valid] = nlcd_pixels[imperviousness_valid]
        data[7] = topography_valid
        data[8] = imperviousness_valid
        patches[index] = data.astype(np.float16)
        label = (THREE_CLASS_LABELS[item["mapped_hazard_class"]] if three_class
                 else int(item["flood_risk_label"]))
        metadata.append({
            "location_id": item["location_id"], "partition": item["spatial_partition"],
            "label": label, "county": item["county"],
            "elevation_m": float(item["elevation_m"]), "x_m": x, "y_m": y,
            "topography_coverage": float(topography_valid.mean()),
            "imperviousness_coverage": float(imperviousness_valid.mean()),
        })
        if (index + 1) % 1000 == 0:
            print(f"Extracted {index + 1}/{len(rows)} patches", flush=True)
    patches.flush()
    (output_dir / "metadata.json").write_text(json.dumps({
        "schemaVersion": 1, "patchSize": PATCH_SIZE, "resolutionMeters": 30,
        "channels": CHANNELS, "classes": list(THREE_CLASS_LABELS) if three_class else ["minimal", "one_percent"],
        "rows": metadata,
    }), encoding="utf-8")
    print(f"Wrote {len(rows)} patches to {array_path}")


if __name__ == "__main__":
    main()
