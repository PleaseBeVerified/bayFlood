from pathlib import Path
import os

import whitebox


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TERRAIN_DIR = PROJECT_ROOT / "terrain"
SOURCE_DEM = TERRAIN_DIR / "dem-10m.tif"
STREAM_THRESHOLD_SQUARE_METERS = 1_000_000


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required terrain file is missing: {path}")


def run_stage(label: str, output_name: str, operation) -> None:
    output = TERRAIN_DIR / output_name
    if output.is_file() and output.stat().st_size > 0:
        print(f"SKIP {label}: {output_name} already exists.")
        return

    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    if partial.exists():
        partial.unlink()

    print(f"START {label}: writing {output_name}.")
    result = operation(partial.name)
    if result != 0 or not partial.is_file() or partial.stat().st_size == 0:
        raise RuntimeError(f"{label} failed with Whitebox status {result}.")
    os.replace(partial, output)
    print(f"DONE {label}: {output_name}.")


def main() -> None:
    require_file(SOURCE_DEM)
    TERRAIN_DIR.mkdir(parents=True, exist_ok=True)

    tools = whitebox.WhiteboxTools()
    tools.set_working_dir(str(TERRAIN_DIR))
    tools.set_verbose_mode(True)

    run_stage(
        "250 m-radius local elevation",
        "local-elevation-250m.tif",
        lambda output: tools.diff_from_mean_elev(
            SOURCE_DEM.name,
            output,
            filterx=51,
            filtery=51,
        ),
    )
    run_stage(
        "500 m-radius local elevation",
        "local-elevation-500m.tif",
        lambda output: tools.diff_from_mean_elev(
            SOURCE_DEM.name,
            output,
            filterx=101,
            filtery=101,
        ),
    )
    run_stage(
        "hydrologic depression breaching",
        "dem-hydrologic.tif",
        lambda output: tools.breach_depressions_least_cost(
            SOURCE_DEM.name,
            output,
            dist=50,
            min_dist=True,
            fill=True,
        ),
    )
    require_file(TERRAIN_DIR / "dem-hydrologic.tif")
    run_stage(
        "D8 flow accumulation",
        "flow-accumulation-m2.tif",
        lambda output: tools.d8_flow_accumulation(
            "dem-hydrologic.tif",
            output,
            out_type="catchment area",
            log=False,
        ),
    )
    require_file(TERRAIN_DIR / "flow-accumulation-m2.tif")
    run_stage(
        "stream extraction",
        "derived-streams-1km2.tif",
        lambda output: tools.extract_streams(
            "flow-accumulation-m2.tif",
            output,
            threshold=STREAM_THRESHOLD_SQUARE_METERS,
            zero_background=True,
        ),
    )
    require_file(TERRAIN_DIR / "derived-streams-1km2.tif")
    run_stage(
        "height above nearest drainage",
        "height-above-drainage-m.tif",
        lambda output: tools.elevation_above_stream(
            "dem-hydrologic.tif",
            "derived-streams-1km2.tif",
            output,
        ),
    )

    print("Terrain features complete.")
    print("Keep flow accumulation untransformed on disk; the model will use log1p when sampling it.")


if __name__ == "__main__":
    main()
