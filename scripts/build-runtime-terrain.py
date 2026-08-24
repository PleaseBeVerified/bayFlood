from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TERRAIN_DIR = PROJECT_ROOT / "terrain"
RUNTIME_DIR = TERRAIN_DIR / "runtime"
NO_DATA = -9999

FEATURES = [
    {
        "name": "slope_degrees",
        "source": "slope-degrees.tif",
        "resampling": "average",
        "units": "degrees",
    },
    {
        "name": "local_elevation_250m",
        "source": "local-elevation-250m.tif",
        "resampling": "average",
        "units": "meters",
    },
    {
        "name": "local_elevation_500m",
        "source": "local-elevation-500m.tif",
        "resampling": "average",
        "units": "meters",
    },
    {
        "name": "flow_accumulation_m2",
        "source": "flow-accumulation-m2.tif",
        "resampling": "max",
        "units": "square meters",
    },
    {
        "name": "height_above_drainage_m",
        "source": "height-above-drainage-m.tif",
        "resampling": "average",
        "units": "meters",
    },
]


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def require_tools() -> None:
    for tool in ("gdalwarp", "gdalbuildvrt", "gdal_translate", "gdalinfo"):
        if shutil.which(tool) is None:
            raise RuntimeError(f"{tool} is required but was not found on PATH.")


def replace_output(partial: Path, output: Path) -> None:
    if not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"GDAL did not create {partial}.")
    os.replace(partial, output)


def build_resampled_feature(feature: dict[str, str], resolution: int) -> Path:
    source = TERRAIN_DIR / feature["source"]
    output = RUNTIME_DIR / f"{feature['name']}-{resolution}m.tif"
    if not source.is_file():
        raise FileNotFoundError(f"Missing source terrain raster: {source}")
    if output.is_file() and output.stat().st_mtime >= source.stat().st_mtime:
        print(f"SKIP {output.name}: already current.", flush=True)
        return output

    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    partial.unlink(missing_ok=True)
    run(
        [
            "gdalwarp",
            "-overwrite",
            "-multi",
            "-wo",
            "NUM_THREADS=ALL_CPUS",
            "-r",
            feature["resampling"],
            "-tr",
            str(resolution),
            str(resolution),
            "-tap",
            "-ot",
            "Float32",
            "-dstnodata",
            str(NO_DATA),
            "-of",
            "GTiff",
            "-co",
            "TILED=YES",
            "-co",
            "BLOCKXSIZE=512",
            "-co",
            "BLOCKYSIZE=512",
            "-co",
            "COMPRESS=ZSTD",
            "-co",
            "PREDICTOR=3",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "BIGTIFF=IF_SAFER",
            str(source),
            str(partial),
        ]
    )
    replace_output(partial, output)
    return output


def build_stack(feature_files: list[Path], resolution: int) -> Path:
    vrt = RUNTIME_DIR / f"terrain-features-{resolution}m.vrt"
    vrt.unlink(missing_ok=True)
    run(["gdalbuildvrt", "-separate", str(vrt), *map(str, feature_files)])

    output = RUNTIME_DIR / f"terrain-features-{resolution}m.tif"
    newest_input = max(path.stat().st_mtime for path in feature_files)
    if output.is_file() and output.stat().st_mtime >= newest_input:
        print(f"SKIP {output.name}: already current.", flush=True)
        return output

    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    partial.unlink(missing_ok=True)
    run(
        [
            "gdal_translate",
            "-of",
            "COG",
            "-ot",
            "Float32",
            "-co",
            "COMPRESS=ZSTD",
            "-co",
            "LEVEL=9",
            "-co",
            "PREDICTOR=FLOATING_POINT",
            "-co",
            "NUM_THREADS=ALL_CPUS",
            "-co",
            "BLOCKSIZE=128",
            "-co",
            "INTERLEAVE=PIXEL",
            "-co",
            "BIGTIFF=IF_SAFER",
            "-co",
            "OVERVIEWS=NONE",
            str(vrt),
            str(partial),
        ]
    )
    replace_output(partial, output)
    return output


def validate_stack(output: Path, resolution: int) -> dict:
    result = subprocess.run(
        ["gdalinfo", "-json", str(output)],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    details = json.loads(result.stdout)
    bands = details.get("bands", [])
    if len(bands) != len(FEATURES):
        raise RuntimeError(
            f"Expected {len(FEATURES)} bands in {output}, found {len(bands)}."
        )
    if any(band.get("type") != "Float32" for band in bands):
        raise RuntimeError(f"All bands in {output} must be Float32.")
    image_structure = details.get("metadata", {}).get("IMAGE_STRUCTURE", {})
    if image_structure.get("LAYOUT") != "COG":
        raise RuntimeError(f"{output} is not marked as a COG.")

    manifest = {
        "schemaVersion": 1,
        "file": output.name,
        "sourceResolutionMeters": 10,
        "runtimeResolutionMeters": resolution,
        "crs": "EPSG:3310",
        "noData": NO_DATA,
        "blockSizePixels": 128,
        "width": details["size"][0],
        "height": details["size"][1],
        "bands": [
            {
                "index": index,
                "feature": feature["name"],
                "units": feature["units"],
                "resampling": feature["resampling"],
            }
            for index, feature in enumerate(FEATURES)
        ],
    }
    manifest_path = RUNTIME_DIR / "terrain-features-manifest.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a compact multi-band Cloud-Optimized GeoTIFF for runtime sampling."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=30,
        help="Output cell size in meters (default: 30).",
    )
    parser.add_argument(
        "--keep-intermediates",
        action="store_true",
        help="Keep the separate resampled bands and VRT after the final COG is valid.",
    )
    arguments = parser.parse_args()
    if arguments.resolution < 10 or arguments.resolution % 10 != 0:
        raise ValueError("--resolution must be a multiple of 10 meters and at least 10.")

    require_tools()
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    feature_files = [
        build_resampled_feature(feature, arguments.resolution)
        for feature in FEATURES
    ]
    output = build_stack(feature_files, arguments.resolution)
    manifest = validate_stack(output, arguments.resolution)
    if not arguments.keep_intermediates:
        for feature_file in feature_files:
            feature_file.unlink(missing_ok=True)
        (RUNTIME_DIR / f"terrain-features-{arguments.resolution}m.vrt").unlink(
            missing_ok=True
        )
        print("Removed regenerable runtime intermediates.", flush=True)
    print(
        f"Runtime terrain COG complete: {output} "
        f"({output.stat().st_size / (1024 * 1024):.1f} MiB, "
        f"{manifest['width']} x {manifest['height']} x {len(FEATURES)} bands).",
        flush=True,
    )


if __name__ == "__main__":
    main()
