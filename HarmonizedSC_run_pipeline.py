#!/usr/bin/env python
"""
Orchestrator for Spectral Complexity and Continuous Change Detection.

Serializes the 6-step processing chain:
    Step 1: HLS30_earthAccess_to_hdf5 (NASA HLS30 STAC Ingestion)
    Step 2: Tanager_STAC-imagery-downloader (Planet Tanager-1 STAC Ingestion)
    Step 3: HLST_constellation_to_hdf5 (Albers Orthorectification & Harmonization)
    Step 4: HLST_SC_calculations (Spectral Complexity Metrics Calculation)
    Step 5: harmonized_CCD_main (Continuous Change Detection Regression)
    Step 6: harmonized_CCD_vis (Continuous Change Detection Visualization)
"""

import os
import sys
import argparse
import importlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml


STEP_DEFINITIONS = {
    1: {
        "name": "HLS30_earthAccess_to_hdf5",
        "description": "NASA HLS30 STAC Download and HDF5 Ingestion",
        "module": "HLS30_earthAccess_to_hdf5",
    },
    2: {
        "name": "Tanager_STAC-imagery-downloader",
        "description": "Planet Tanager-1 STAC Basic Reflectance Download",
        "module": "Tanager_STAC-imagery-downloader",
    },
    3: {
        "name": "HLST_constellation_to_hdf5",
        "description": "Albers Equal Area Constellation Orthorectification and Harmonization",
        "module": "HLST_constellation_to_hdf5",
    },
    4: {
        "name": "HLST_SC_calculations",
        "description": "Spectral Complexity Metrics Calculation",
        "module": "HLST_SC_calculations",
    },
    5: {
        "name": "harmonized_CCD_main",
        "description": "Harmonized Continuous Change Detection Temporal Regression",
        "module": "harmonized_CCD_main",
    },
    6: {
        "name": "harmonized_CCD_vis",
        "description": "Continuous Change Detection Visualization",
        "module": "harmonized_CCD_vis",
    },
}



def load_pipeline_config(config_path: Optional[str] = None, location: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads location and pipeline configuration YAML file with fallback hierarchy.
    """
    config_file = Path(config_path) if config_path else Path(__file__).resolve().parent / "locations_config.yaml"
    config_data = None

    if config_file.exists():
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
        except Exception:
            pass

    if config_data is None:
        # Fallback default configuration for Rochesterv2 baseline
        config_data = {
            "current_run": {"location": "Rochesterv2"},
            "locations": {
                "Rochesterv2": {
                    "SOURCE_CACHE": None,
                    "ROI_LON_MIN": -77.770166,
                    "ROI_LON_MAX": -77.376776,
                    "ROI_LAT_MIN": 42.961778,
                    "ROI_LAT_MAX": 43.342135,
                    "START_DATE": "2015-01-01",
                    "END_DATE": "2026-06-01",
                    "TANAGER_AVAILABLE": True,
                }
            }
        }

    return config_data



def run_pipeline(
    location: Optional[str] = None,
    start_date: str = "2015-01-01",
    end_date: str = "2026-06-01",
    tile_size: int = 3,
    num_endmembers: int = 7,
) -> Dict[str, Any]:
    """
    Executes the 6-step spectral complexity pipeline non-interactively.

    Parameters:
        location: Target location identifier (for example, 'Rochesterv2').
        start_date: Start date string (YYYY-MM-DD), default '2015-01-01'.
        end_date: End date string (YYYY-MM-DD), default '2026-06-01'.
        tile_size: Spatial sliding window size for spectral complexity calculation.
        num_endmembers: Number of endmembers for simplex extraction.

    Returns:
        Dictionary containing execution summary and artifact paths.
    """
    active_steps = [1, 2, 3, 4, 5, 6]

    config_data = load_pipeline_config(location=location)

    if location is None:
        location = config_data.get("current_run", {}).get("location", "Rochesterv2")

    loc_settings = config_data.get("locations", {}).get(location, {})

    print("=" * 70)
    print(" HARMONIZED SPECTRAL COMPLEXITY PIPELINE ORCHESTRATOR")
    print("=" * 70)
    print(f" Target Location: {location}")
    print(f" Temporal Span:   {start_date} to {end_date}")
    print(f" Active Steps:    {active_steps}")
    print(f" Tile Size:       {tile_size}")
    print(f" Num Endmembers:  {num_endmembers}")
    print("=" * 70)

    pipeline_state: Dict[str, Any] = {
        "location": location,
        "start_date": start_date,
        "end_date": end_date,
        "executed_steps": [],
        "artifacts": {},
        "status": "RUNNING",
    }

    master_h5 = None

    for step_num in active_steps:
        step_info = STEP_DEFINITIONS[step_num]
        step_name = step_info["name"]
        step_desc = step_info["description"]

        print(f"\n{'-' * 60}")
        print(f"Step {step_num}: {step_name}")
        print(f"Purpose: {step_desc}")
        print(f"{'-' * 60}")

        # Execute Step 1: HLS30 STAC Ingestion
        if step_num == 1:
            hls_mod = importlib.import_module("HLS30_earthAccess_to_hdf5")
            native_h5 = hls_mod.main(
                target_location=location,
                start_date=start_date,
                end_date=end_date,
                config_override=config_data,
            )
            pipeline_state["artifacts"]["native_h5"] = native_h5
            print(f"Step 1 Complete: {native_h5}")

        # Execute Step 2: Tanager STAC Ingestion
        elif step_num == 2:
            tanager_mod = importlib.import_module("Tanager_STAC-imagery-downloader")
            total_scenes = tanager_mod.main(
                start_date=start_date,
                end_date=end_date,
            )
            pipeline_state["artifacts"]["tanager_scenes"] = total_scenes
            print(f"Step 2 Complete: Processed {total_scenes} Tanager scenes.")

        # Execute Step 3: Albers Orthorectification and Harmonization
        elif step_num == 3:
            harm_mod = importlib.import_module("HLST_constellation_to_hdf5")
            input_h5 = pipeline_state["artifacts"].get("native_h5")
            master_h5 = harm_mod.main(
                target_location=location,
                input_native_h5=input_h5,
                config_override=config_data,
            )
            pipeline_state["artifacts"]["master_h5"] = master_h5
            print(f"Step 3 Complete: {master_h5}")

        # Execute Step 4: Spectral Complexity Calculations (In-Place H5 Storage)
        elif step_num == 4:
            sc_mod = importlib.import_module("HLST_SC_calculations")
            sc_mod.process_ard_cube(
                master_h5,
                tile_size=tile_size,
                num_endmembers=num_endmembers,
            )
            pipeline_state["artifacts"]["sc_h5"] = master_h5
            print(f"Step 4 Complete: Updated metrics in-place in {master_h5}")

        # Execute Step 5: Continuous Change Detection Temporal Regression
        elif step_num == 5:
            ccd_mod = importlib.import_module("harmonized_CCD_main")
            ccd_res_h5 = ccd_mod.main(
                location=location,
                h5_path=master_h5,
                launch_vis=False,
            )
            pipeline_state["artifacts"]["ccd_output_h5"] = ccd_res_h5
            print(f"Step 5 Complete: CCD results saved directly to {ccd_res_h5}")

        # Execute Step 6: Continuous Change Detection Visualization
        elif step_num == 6:
            vis_mod = importlib.import_module("harmonized_CCD_vis")
            out_png = vis_mod.get_default_png_name(
                location=location
            )
            vis_mod.plot_spatial_anomaly_overlay(
                source_h5_path=master_h5,
                output_png=out_png,
                interactive=False,
            )
            pipeline_state["artifacts"]["diagnostic_png"] = out_png
            print(f"Step 6 Complete: Diagnostic map saved to {out_png}")

        pipeline_state["executed_steps"].append(step_num)

    pipeline_state["status"] = "SUCCESS"
    print("\n" + "=" * 70)
    print(" PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    return pipeline_state


def build_arg_parser() -> argparse.ArgumentParser:
    """Constructs the command line argument parser for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="Harmonized Spectral Complexity 6-Step Processing Pipeline"
    )
    parser.add_argument(

        "--location",
        type=str,
        default=None,
        help="Target location identifier (for example, 'Rochesterv2')",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2015-01-01",
        help="Temporal acquisition start date (YYYY-MM-DD, default 2015-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2026-06-01",
        help="Temporal acquisition end date (YYYY-MM-DD, default 2026-06-01)",
    )
    parser.add_argument(

        "--tile-size",
        type=int,
        default=3,
        help="Sliding window tile size for spectral complexity calculation (default 3)",
    )
    parser.add_argument(
        "--num-endmembers",
        type=int,
        default=7,
        help="Number of endmembers to extract in spectral complexity (default 7)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point returning integer status code."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        run_pipeline(
            location=args.location,
            start_date=args.start_date,
            end_date=args.end_date,
            tile_size=args.tile_size,
            num_endmembers=args.num_endmembers,
        )
        return 0
    except Exception as exc:
        print(f"Pipeline execution terminated with failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
