'''
Tanager STAC Imagery Downloader.
Downloads Tanager-1 hyperspectral surface reflectance and radiance assets from Planet STAC catalogs.
Organizes data into item directories for subsequent multi-sensor harmonization.
'''
import os
import requests
import json
from pathlib import Path
from urllib.parse import urljoin

# Pre-defined Regions of Interest (Bounding Boxes: [min_lon, min_lat, max_lon, max_lat])
REGIONS = {
    "Southern_California": [-119.503784, 33.582591, -117.686920, 34.746126],
    "Utah": [-114.05, 37.0, -109.0, 42.5],
    "Rochester_NY": [-77.72, 43.04, -77.44, 43.28],
    "BuenosAires": [-65.0, -70.0, -41.0, -30.0],
    "Global": [-180.0, -90.0, 180.0, 90.0],
    "CentralGreece": [21.0, 40.0, 22.0, 41.0],
}

DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2026-12-31"

# Download Jobs Configuration
# Points multiple jobs to the corresponding output_dir and include_bboxes.
DOWNLOAD_JOBS = [
    {
        "job_name": "ROCX_Rochester",
        "collection_url": "https://www.planet.com/data/stac/tanager-core-imagery/ROCX2025/collection.json",
        "output_dir": r"C:\satelliteImagery\Tanager\Rochesterv2_SourceData",
        "include_bboxes": [REGIONS["Rochester_NY"]],
        "exclude_bboxes": [],
        "target_assets": ['basic_sr_hdf5']
    },
    {
        "job_name": "CentralGreece",
        "collection_url": "https://www.planet.com/data/stac/tanager-core-imagery/coastal-water-bodies/collection.json",
        "output_dir": r"C:\satelliteImagery\Tanager\CentralGreece_SourceData",
        "include_bboxes": [REGIONS["CentralGreece"]],
        "exclude_bboxes": [REGIONS["Utah"]],
        "target_assets": ['basic_sr_hdf5']
    }
]


def intersects(bbox1, bbox2):
    """Evaluates whether two [min_lon, min_lat, max_lon, max_lat] bounding boxes intersect."""
    return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])


def passes_spatial_filters(item_bbox, include_bboxes, exclude_bboxes):
    """
    Evaluates an item's bounding box against explicit inclusion and exclusion regions.
    """
    if not item_bbox:
        return False

    # 1. Strict Exclusions (e.g., dropping overlapping Utah footprints)
    for ex_box in exclude_bboxes:
        if intersects(item_bbox, ex_box):
            return False

    # 2. Inclusions
    if not include_bboxes:
        return True  # If no inclusion regions are specified, accept all non-excluded items
        
    for inc_box in include_bboxes:
        if intersects(item_bbox, inc_box):
            return True

    return False


def passes_temporal_filter(item_data, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE):
    """
    Verifies that the item falls within the 2022-2026 test period.
    """
    props = item_data.get('properties', {})
    dt_str = props.get('datetime') or props.get('acquisition_time') or item_data.get('datetime')
    if not dt_str:
        return True
    item_date = str(dt_str)[:10]
    return start_date <= item_date <= end_date


def download_file(url, destination_path):
    """Streams file download to disk to handle large multi-gigabyte HDF5 assets efficiently."""
    dest = Path(destination_path)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    -> Skipping {dest.name} (Already exists)")
        return True

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(dest, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"    -> Failed to download {url}: {e}")
        return False


def execute_job(job_config, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, session=None):
    """Processes a single download job dictionary."""
    job_name = job_config["job_name"]
    collection_url = job_config["collection_url"]
    out_dir = Path(job_config["output_dir"])
    includes = job_config.get("include_bboxes", [])
    excludes = job_config.get("exclude_bboxes", [])
    target_assets = job_config.get("target_assets", ['basic_sr_hdf5'])

    print(f"\n{'=' * 50}")
    print(f"Executing Job: {job_name}")
    print(f"Target Directory: {out_dir}")
    print(f"{'=' * 50}")

    out_dir.mkdir(parents=True, exist_ok=True)

    http_client = session if session is not None else requests

    try:
        response = http_client.get(collection_url)
        response.raise_for_status()
        collection_data = response.json()
    except Exception as e:
        print(f"Failed to fetch catalog at {collection_url}: {e}")
        return 0

    item_links = [link['href'] for link in collection_data.get('links', []) if link.get('rel') == 'item']
    print(f"Discovered {len(item_links)} items in catalog.")
    
    matched_items = 0

    for idx, item_href in enumerate(item_links):
        item_url = urljoin(collection_url, item_href)
        
        try:
            item_resp = http_client.get(item_url)
            item_resp.raise_for_status()
            item_data = item_resp.json()
        except Exception as e:
            print(f"  [Error] Failed to fetch item metadata at {item_url}: {e}")
            continue

        item_id = item_data.get('id', f'item_{idx}')
        item_bbox = item_data.get('bbox')
        
        # Apply Spatial Filtering
        if not passes_spatial_filters(item_bbox, includes, excludes):
            continue

        # Apply Temporal Filtering for the 2022-2026 test period
        if not passes_temporal_filter(item_data, start_date, end_date):
            continue

        matched_items += 1
        print(f"\n  [{item_id}] Matched spatial and temporal criteria. Processing assets...")
        
        # Route to common folder structure to support the native stacker
        item_folder = out_dir / item_id
        item_folder.mkdir(exist_ok=True)

        # Download STAC JSON Metadata (Required for extracting exact Acquisition Time)
        json_file_name = f"{item_id}.json"
        json_dest = item_folder / json_file_name
        download_file(item_url, json_dest)

        # Download Specific Assets
        assets = item_data.get('assets', {})
        for asset_key in target_assets:
            if asset_key in assets:
                asset_url = assets[asset_key].get('href')
                if not asset_url.startswith('http'):
                    asset_url = urljoin(item_url, asset_url)
                
                file_name = os.path.basename(asset_url)
                dest_path = item_folder / file_name
                print(f"    -> Downloading {asset_key}...")
                download_file(asset_url, dest_path)
            else:
                print(f"    -> Asset '{asset_key}' not present in this item.")

    print(f"\nJob '{job_name}' completed. Successfully processed {matched_items} scenes.")
    return matched_items


def main(jobs=None, start_date=DEFAULT_START_DATE, end_date=DEFAULT_END_DATE, session=None):
    print("Initializing Tanager STAC Job Queue...\n")
    job_list = jobs if jobs is not None else DOWNLOAD_JOBS
    total_scenes = 0
    for job in job_list:
        total_scenes += execute_job(job, start_date=start_date, end_date=end_date, session=session)
    print(f"\nAll download jobs finished. Total scenes processed: {total_scenes}")
    return total_scenes


if __name__ == "__main__":
    main()
