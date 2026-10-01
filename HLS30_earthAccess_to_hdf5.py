'''
Downloads HLS30 data for a specified ROI and time range from NASA Earth Access. 
Detects images originating from the exact same orbital pass and geometrically 
fuses adjacent MGRS tiles back into continuous sensor swaths.
Merges downloaded imagery into a consolidated, unified h5 file. 
'''
import os
import sys
import json
import re
import warnings
import collections
from datetime import datetime, timezone
from pathlib import Path
import concurrent.futures

import numpy as np
import h5py
import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer, CRS
import pystac_client
import pystac
import earthaccess
import yaml
import hdfeos_odl

# Add parent folder and dev repo to sys.path to find hdfeos_odl
script_dir = Path(__file__).resolve().parent
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))
if str(script_dir.parent) not in sys.path:
    sys.path.insert(0, str(script_dir.parent))

LOCATION_DEFAULT = "Rochesterv2"
DEFAULT_START_DATE = "2022-01-01"
DEFAULT_END_DATE = "2026-06-01"

ASSETS_S30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']
ASSETS_L30 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask', 'SZA', 'SAA', 'VZA', 'VAA']

S30_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.865, 1.610, 2.190]
L30_SR_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201]

# Sentinel constants and fill values
FILL_VALUE_REFLECTANCE_RAW = -9999
FILL_VALUE_FMASK_RAW = 255
FILL_VALUE_ANGLES_RAW = 40000

SCALE_FACTOR_REFLECTANCE = 0.0001
SCALE_FACTOR_ANGLES = 0.01

def load_locations_config(config_path=None):
    """Loads location configuration yaml with fallbacks."""
    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    # Fallback default configuration for Rochesterv2 baseline
    return {
        "current_run": {"location": "Rochesterv2"},
        "locations": {
            "Rochesterv2": {
                "SOURCE_CACHE": None,
                "ROI_LON_MIN": -77.770166,
                "ROI_LON_MAX": -77.376776,
                "ROI_LAT_MIN": 42.961778,
                "ROI_LAT_MAX": 43.342135,
                "START_DATE": "2015-01-01",
                "END_DATE": "2026-06-01"
            }
        }
    }

def _fetch_single_band(idx, asset_key, url, window_obj):
    """Thread-safe worker block for fetching independent spectral bands."""
    if asset_key == 'Fmask':
        fill_val = FILL_VALUE_FMASK_RAW
    elif asset_key in ['SZA', 'SAA', 'VZA', 'VAA']:
        fill_val = FILL_VALUE_ANGLES_RAW
    else:
        fill_val = FILL_VALUE_REFLECTANCE_RAW

    with rasterio.open(url) as b_src:
        # Broad casting to int32 to safely accommodate both -9999 and 40000 without overflow
        data = b_src.read(1, window=window_obj, boundless=True, fill_value=fill_val).astype(np.int32)
    return idx, data


def stream_native_tiles_to_hdf5(h5f, sensor_prefix, tile_collections, expected_sr, expected_fmask_idx, wavelengths, safe_bbox):
    """
    Streams multi-temporal native HLS tiles into native HDF5 dataset structure.
    Preserves exact legacy fill values (-9999 -> NaN, 255, 40000 -> NaN) and HDF-EOS grid metadata.
    """
    dt_str = h5py.string_dtype(encoding='ascii')
    odl_blocks = []

    for tile_name, tile_data in tile_collections.items():
        if not tile_data.get('items'):
            continue

        # Sort frames chronologically
        frames = sorted(tile_data['items'].items(), key=lambda x: x[1]['acquisition_time'])
        num_frames = len(frames)
        
        # Establish the exact native bounds for THIS specific location (safe_bbox)
        first_frame_meta = frames[0][1]
        with rasterio.open(first_frame_meta['filepath']) as src:
            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
            xs, ys = transformer.transform(
                [safe_bbox[0], safe_bbox[2], safe_bbox[2], safe_bbox[0]], 
                [safe_bbox[3], safe_bbox[3], safe_bbox[1], safe_bbox[1]]
            )
            roi_minx, roi_maxx, roi_miny, roi_maxy = min(xs), max(xs), min(ys), max(ys)
            master_window = from_bounds(roi_minx, roi_miny, roi_maxx, roi_maxy, transform=src.transform).round_offsets().round_lengths()
            tile_target_transform = src.window_transform(master_window)
            tile_crs = src.crs

        w, h = master_window.width, master_window.height
        
        grid_name = f"{sensor_prefix}_{tile_name}"
        group_path = f'/HDFEOS/GRIDS/{grid_name}/Data Fields'
        grp = h5f.create_group(group_path)

        chunk_h, chunk_w = min(h, 256), min(w, 256)
        
        sr_ds = grp.create_dataset('surface_reflectance', shape=(num_frames, expected_sr, h, w), 
                                   chunks=(1, expected_sr, chunk_h, chunk_w),
                                   dtype=np.float32, shuffle=True, compression='gzip', compression_opts=6)
        
        fm_ds = grp.create_dataset('Fmask', shape=(num_frames, h, w), 
                                   chunks=(1, chunk_h, chunk_w),
                                   dtype=np.uint8, shuffle=True, compression='gzip', compression_opts=6)
                                   
        ag_ds = grp.create_dataset('solar_view_angles', shape=(num_frames, 4, h, w), 
                                   chunks=(1, 4, chunk_h, chunk_w),
                                   dtype=np.float32, shuffle=True, compression='gzip', compression_opts=6)

        meta_arrays = {'acq': [], 'space': [], 'saz': [], 'sel': [], 'cc': []}

        for idx, (img_id, meta) in enumerate(frames):
            for k, v in meta.items():
                if v is None and k != 'filepath': 
                    raise ValueError(f"CRITICAL ERROR: Metadata '{k}' for '{img_id}' is null.")

            with rasterio.open(meta['filepath']) as src:
                # Dynamically slice cached TIFF
                read_window = from_bounds(roi_minx, roi_miny, roi_maxx, roi_maxy, transform=src.transform).round_offsets().round_lengths()
                
                t_sr = src.read(list(range(1, expected_sr + 1)), window=read_window)
                t_fm = src.read(expected_fmask_idx, window=read_window)
                t_ag = src.read(list(range(expected_fmask_idx + 1, expected_fmask_idx + 5)), window=read_window)

                sr_ds[idx, ...] = np.where(t_sr != FILL_VALUE_REFLECTANCE_RAW, t_sr.astype(np.float32) * SCALE_FACTOR_REFLECTANCE, np.nan)
                
                sr_valid = t_sr[0] != FILL_VALUE_REFLECTANCE_RAW
                fm_pass = np.where((t_fm != FILL_VALUE_FMASK_RAW) & sr_valid, t_fm, FILL_VALUE_FMASK_RAW)
                fm_ds[idx, ...] = fm_pass

                ag_pass = np.where((t_ag != FILL_VALUE_ANGLES_RAW) & sr_valid, t_ag * SCALE_FACTOR_ANGLES, np.nan)
                ag_ds[idx, ...] = ag_pass

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_sza = np.nanmean(ag_pass[0])
                mean_saa = np.nanmean(ag_pass[1])

            if np.isnan(mean_sza) or np.isnan(mean_saa):
                if np.any(sr_valid):
                    print(f"WARNING: Raster-derived mean sun angles are NaN for frame {idx} despite having valid SR data.")
                mean_sza = 0.0
                mean_saa = 0.0

            meta_arrays['acq'].append(meta['acquisition_time'])
            meta_arrays['space'].append(meta['spacecraft_id'])
            meta_arrays['saz'].append(mean_saa)
            meta_arrays['sel'].append(90.0 - mean_sza)
            meta_arrays['cc'].append(meta['cloud_cover'])

        tf = tile_target_transform
        gdal_transform = np.array([tf.c, tf.a, tf.b, tf.f, tf.d, tf.e], dtype='float64')
                                   
        sr_ds.attrs['units'] = "Reflectance"
        sr_ds.attrs['_FillValue'] = np.nan
        sr_ds.attrs['wavelengths'] = wavelengths
        
        sr_ds.attrs['spatial_ref'] = tile_crs.to_wkt()
        sr_ds.attrs['GeoTransform'] = gdal_transform

        fm_ds.attrs['_FillValue'] = FILL_VALUE_FMASK_RAW
        ag_ds.attrs['_FillValue'] = np.nan
        ag_ds.attrs['band_order'] = ["SZA", "SAA", "VZA", "VAA"]
        
        sr_ds.attrs.create('spacecraft_id', data=meta_arrays['space'], dtype=dt_str)
        sr_ds.attrs['acquisition_time'] = np.array(meta_arrays['acq'], dtype='float64') 
        sr_ds.attrs['sun_azimuth'] = np.array(meta_arrays['saz'], dtype='float32')
        sr_ds.attrs['sun_elevation'] = np.array(meta_arrays['sel'], dtype='float32')
        sr_ds.attrs['cloud_cover'] = np.array(meta_arrays['cc'], dtype='float32')
        
        zone = tile_data.get('zone', 0)
        odl = hdfeos_odl.generate_earthaccess_hls_odl_grid_string(
                grid_name, w, h, tf, "GCTP_UTM", zone, [0.0] * 13, expected_sr, num_frames, False)
        odl_blocks.append(odl)
        
    return odl_blocks





def main(target_location=None, start_date=None, end_date=None, output_dir=None, config_override=None):
    # ==========================================
    # 1. CONFIGURATION & AUTHENTICATION
    # ==========================================
    cloud_threshold = 85

    print("Authenticating with NASA Earthdata...")
    try:
        earthaccess.login(strategy="all", persist=True)
    except Exception as e:
        print(f"Warning: earthaccess authentication encountered: {e}")

    # Load Configuration
    if config_override is not None:
        config_data = config_override
    else:
        config_data = load_locations_config()

    if target_location is not None:
        Location = target_location
    else:
        Location = config_data.get("current_run", {}).get("location", LOCATION_DEFAULT)

    if Location not in config_data.get("locations", {}):
        raise ValueError(f"Location '{Location}' not found in configuration.")

    config = config_data["locations"][Location]

    SOURCE_CACHE = config.get("SOURCE_CACHE")
    ROI_LON_MIN = config["ROI_LON_MIN"]
    ROI_LON_MAX = config["ROI_LON_MAX"]
    ROI_LAT_MIN = config["ROI_LAT_MIN"]
    ROI_LAT_MAX = config["ROI_LAT_MAX"]

    # Temporal filtering targeting 2022-2026 test period
    if start_date is not None:
        START_DATE = start_date
    elif "START_DATE" in config and config["START_DATE"] >= "2022-01-01":
        START_DATE = config["START_DATE"]
    else:
        START_DATE = DEFAULT_START_DATE

    if end_date is not None:
        END_DATE = end_date
    elif "END_DATE" in config:
        END_DATE = config["END_DATE"]
    else:
        END_DATE = DEFAULT_END_DATE

    if SOURCE_CACHE and SOURCE_CACHE in config_data.get("locations", {}):
        cache_config = config_data["locations"][SOURCE_CACHE]
        cache_bbox = [
            min(cache_config["ROI_LON_MIN"], cache_config["ROI_LON_MAX"]), 
            max(cache_config["ROI_LAT_MIN"], cache_config["ROI_LAT_MAX"]),
            max(cache_config["ROI_LON_MIN"], cache_config["ROI_LON_MAX"]), 
            min(cache_config["ROI_LAT_MIN"], cache_config["ROI_LAT_MAX"])
        ]
    else:
        cache_bbox = [
            min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
            max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
        ]
    cache_bbox = [min(cache_bbox[0], cache_bbox[2]), min(cache_bbox[1], cache_bbox[3]), max(cache_bbox[0], cache_bbox[2]), max(cache_bbox[1], cache_bbox[3])]

    COMBINED_OUTPUT_DIR = r"C:\satelliteImagery\HLS30"
    HLSS30_OUTPUT_DIR = os.path.join(COMBINED_OUTPUT_DIR, "HLSS30-SourceData")
    HLSL30_OUTPUT_DIR = os.path.join(COMBINED_OUTPUT_DIR, "HLSL30-SourceData")

    if SOURCE_CACHE:
        S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
        L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{SOURCE_CACHE}/STAC_CACHE")
    else:
        S30_TEMP_DIR = os.path.join(HLSS30_OUTPUT_DIR, f"{Location}/STAC_CACHE")
        L30_TEMP_DIR = os.path.join(HLSL30_OUTPUT_DIR, f"{Location}/STAC_CACHE")

    os.makedirs(S30_TEMP_DIR, exist_ok=True)
    os.makedirs(L30_TEMP_DIR, exist_ok=True)
    os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

    # Pre-MGRS native format filename
    OUTPUT_NATIVE_HDF5 = os.path.join(COMBINED_OUTPUT_DIR, f"HLS_{Location}_STAC_Native_2025.h5")

    # Defensive Topology Enforcement for Output Grid (Location ROI)
    safe_bbox = [
        min(ROI_LON_MIN, ROI_LON_MAX), max(ROI_LAT_MIN, ROI_LAT_MAX), 
        max(ROI_LON_MIN, ROI_LON_MAX), min(ROI_LAT_MIN, ROI_LAT_MAX)
    ]
    safe_bbox = [min(safe_bbox[0], safe_bbox[2]), min(safe_bbox[1], safe_bbox[3]), max(safe_bbox[0], safe_bbox[2]), max(safe_bbox[1], safe_bbox[3])]

    # ==========================================
    # 3. NASA STAC QUERY & NATIVE VIRTUAL READ (HLS)
    # ==========================================
    def stac_native_window_read(collection_id, assets_list, temp_dir):
        # 1. Setup Caching Infrastructure
        STAC_METADATA_CACHE_DIR = os.path.join(COMBINED_OUTPUT_DIR, "STAC_METADATA_CACHE")
        os.makedirs(STAC_METADATA_CACHE_DIR, exist_ok=True)
        
        cache_name = SOURCE_CACHE if SOURCE_CACHE else Location
        cache_filename = f"{cache_name}_{collection_id}_{START_DATE}_{END_DATE}_c{cloud_threshold}.json"
        cache_filepath = os.path.join(STAC_METADATA_CACHE_DIR, cache_filename)
        
        all_items_in_cache_bounds = []
        
        if os.path.exists(cache_filepath):
            print(f"\nLoading NASA STAC metadata from local cache for {collection_id}...")
            with open(cache_filepath, 'r') as f:
                cached_data = json.load(f)
                all_items_in_cache_bounds = [pystac.Item.from_dict(d) for d in cached_data]
            print(f"Loaded {len(all_items_in_cache_bounds)} items from cache.")
        else:
            print(f"\nQuerying NASA CMR STAC for {collection_id} over the SOURCE_CACHE extent...")
            catalog = pystac_client.Client.open("https://cmr.earthdata.nasa.gov/stac/LPCLOUD")
            search = catalog.search(collections=[collection_id], bbox=cache_bbox, datetime=f"{START_DATE}/{END_DATE}", limit=500)
            
            # Initial cloud filter
            all_items_in_cache_bounds = [i for i in list(search.items()) if i.properties.get('eo:cloud_cover', 100) < cloud_threshold]
            total_items = len(all_items_in_cache_bounds)
            print(f"Identified {total_items} STAC items for {collection_id} within temporal bounds and cloud thresholds.")
        
            # Fetch EarthAccess Metadata
            platform_mapping = {}
            item_ids = [i.id for i in all_items_in_cache_bounds]
            if item_ids:
                print(f"Fetching platform metadata via earthaccess for {len(item_ids)} items...")
                short_name = collection_id.split('.')[0]
                for chunk_start in range(0, len(item_ids), 100):
                    chunk = item_ids[chunk_start:chunk_start + 100]
                    try:
                        ea_results = earthaccess.search_data(short_name=short_name, granule_ur=chunk, count=len(chunk))
                        for g in ea_results:
                            plats = g.get('umm', {}).get('Platforms', [])
                            if plats:
                                platform_mapping[g['umm']['GranuleUR']] = plats[0].get('ShortName', 'UNKNOWN')
                    except Exception as e:
                        print(f"Warning: Failed to fetch earthaccess metadata for chunk: {e}")
            
            # Embed earthaccess metadata into the STAC items for caching
            serialized_items = []
            for item in all_items_in_cache_bounds:
                item.properties['platform'] = platform_mapping.get(item.id, item.properties.get('platform', 'UNKNOWN'))
                serialized_items.append(item.to_dict())
                
            # Save the cache to disk
            with open(cache_filepath, 'w') as f:
                json.dump(serialized_items, f)
            print(f"Saved STAC metadata to local cache: {cache_filename}")

        # 2. Local Geometric Filtering to the Actual Request Bounding Box (safe_bbox)
        roi_minx, roi_miny, roi_maxx, roi_maxy = safe_bbox
        
        filtered_items = []
        for item in all_items_in_cache_bounds:
            if not item.bbox:
                continue
            item_minx, item_miny, item_maxx, item_maxy = item.bbox
            
            intersects = (
                item_minx <= roi_maxx and
                item_maxx >= roi_minx and
                item_miny <= roi_maxy and
                item_maxy >= roi_miny
            )
            
            if intersects:
                filtered_items.append(item)
                
        total_filtered = len(filtered_items)
        print(f"After local filtering, {total_filtered} items intersect the exact target ROI ({Location}).")

        tile_collections = {}
    
        gdal_env = {
            'GDAL_HTTP_COOKIEFILE': os.path.expanduser('~/.urs_cookies'),
            'GDAL_HTTP_COOKIEJAR': os.path.expanduser('~/.urs_cookies'),
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': 'tif',
            'VSI_CACHE': True,
            'GDAL_HTTP_MULTIPLEX': 'YES'
        }

        with rasterio.Env(**gdal_env):
            for i, item in enumerate(filtered_items, 1):
                img_id = item.id 
                parsed_mgrs_tile = img_id.split('.')[2] 
                cloud_cov = item.properties.get('eo:cloud_cover')
            
                if cloud_cov is None: 
                    print(f"  [{i}/{total_filtered}] [{img_id}] WARNING: STAC metadata is null. Excluding frame.")
                    continue
                
                if parsed_mgrs_tile not in tile_collections:
                    tile_collections[parsed_mgrs_tile] = {'items': {}, 'transform': None, 'crs': None, 'width': None, 'height': None, 'zone': None}
                
                tile_data = tile_collections[parsed_mgrs_tile]
                out_tif = os.path.join(temp_dir, f"{img_id}.tif")
            
                tile_data['items'][img_id] = {
                    'acquisition_time': item.datetime.timestamp(),
                    'spacecraft_id': item.properties.get('platform', 'UNKNOWN'),
                    'cloud_cover': cloud_cov,
                    'filepath': out_tif
                }
            
                if os.path.exists(out_tif) and os.path.getsize(out_tif) > 0:
                    with rasterio.open(out_tif) as cached_src:
                        transformer = Transformer.from_crs("EPSG:4326", cached_src.crs, always_xy=True)
                        xs, ys = transformer.transform(
                            [cache_bbox[0], cache_bbox[2], cache_bbox[2], cache_bbox[0]], 
                            [cache_bbox[3], cache_bbox[3], cache_bbox[1], cache_bbox[1]]
                        )
                        c_minx, c_maxx, c_miny, c_maxy = min(xs), max(xs), min(ys), max(ys)
                        
                        cache_valid = (
                            cached_src.bounds.left <= c_minx + 40 and 
                            cached_src.bounds.right >= c_maxx - 40 and 
                            cached_src.bounds.bottom <= c_miny + 40 and 
                            cached_src.bounds.top >= c_maxy - 40
                        )

                    if cache_valid:
                        print(f"  [{i}/{total_filtered}] [{img_id}] Valid cache located. Skipping STAC download.")
                        if tile_data['transform'] is None:
                            with rasterio.open(out_tif) as cached_src:
                                tile_data['transform'] = cached_src.transform
                                tile_data['crs'] = cached_src.crs
                                tile_data['width'] = cached_src.width
                                tile_data['height'] = cached_src.height
                            
                                epsg_code = CRS.from_user_input(cached_src.crs).to_epsg()
                                if epsg_code is not None:
                                    tile_data['zone'] = epsg_code % 100
                                else:
                                    zone_match = re.search(r'T(\d+)', parsed_mgrs_tile)
                                    if zone_match:
                                        tile_data['zone'] = int(zone_match.group(1))
                                    else:
                                        tile_data['zone'] = cached_src.crs.to_dict().get('zone')
                        continue
                    else:
                        print(f"  [{i}/{total_filtered}] [{img_id}] Cache exists but bounds are too small for requested region. Re-downloading.")
                print(f"  [{i}/{total_filtered}] [{img_id}] Downloading {len(assets_list)} STAC assets via Concurrent Window Read...")    
                try:
                    asset_key_ref = assets_list[0]
                    with rasterio.open(item.assets[asset_key_ref].href) as src:
                        if tile_data.get('window') is None:
                            transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
                            xs, ys = transformer.transform([cache_bbox[0], cache_bbox[2], cache_bbox[2], cache_bbox[0]], [cache_bbox[3], cache_bbox[3], cache_bbox[1], cache_bbox[1]])
                            roi_minx, roi_maxx, roi_miny, roi_maxy = min(xs), max(xs), min(ys), max(ys)
                            window = from_bounds(roi_minx, roi_miny, roi_maxx, roi_maxy, transform=src.transform).round_offsets().round_lengths()
                        
                            tile_data['transform'] = src.window_transform(window)
                            tile_data['crs'] = src.crs
                            
                            epsg_code = CRS.from_user_input(src.crs).to_epsg()
                            if epsg_code is not None:
                                tile_data['zone'] = epsg_code % 100
                            else:
                                zone_match = re.search(r'T(\d+)', parsed_mgrs_tile)
                                if zone_match:
                                    tile_data['zone'] = int(zone_match.group(1))
                                else:
                                    tile_data['zone'] = src.crs.to_dict().get('zone')
                                
                            tile_data['width'] = window.width
                            tile_data['height'] = window.height
                            tile_data['window'] = window
                        else:
                            window = tile_data['window']
                
                    num_assets = len(assets_list)
                    compiled_array = np.zeros((num_assets, tile_data['height'], tile_data['width']), dtype=np.int32)
                
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(assets_list)) as executor:
                        futures = []
                        for idx, asset_key in enumerate(assets_list):
                            url = item.assets[asset_key].href
                            futures.append(executor.submit(_fetch_single_band, idx, asset_key, url, window))
                        for future in concurrent.futures.as_completed(futures):
                            b_idx, b_data = future.result()
                            compiled_array[b_idx, :, :] = b_data
                
                    profile = {'driver': 'GTiff', 'height': tile_data['height'], 'width': tile_data['width'], 'count': num_assets, 'dtype': 'int32', 'crs': tile_data['crs'], 'transform': tile_data['transform'], 'compress': 'deflate'}
                    with rasterio.open(out_tif, 'w', **profile) as dst:
                        dst.write(compiled_array)
                    
                except Exception as e:
                    print(f"  [{i}/{total_filtered}] Failed retrieval for {img_id}: {e}")
                    if img_id in tile_data['items']:
                        del tile_data['items'][img_id]
                
        if not tile_collections:
            raise ValueError(f"CRITICAL ERROR: No valid tiles found for {collection_id} after enforcing spatial bounds.")
        
        return tile_collections

    s30_collections = stac_native_window_read("HLSS30.v2.0", ASSETS_S30, S30_TEMP_DIR)
    l30_collections = stac_native_window_read("HLSL30.v2.0", ASSETS_L30, L30_TEMP_DIR)

    # ==========================================
    # 4. PHASE 1: BUILD NATIVE HDF5 (OUT-OF-CORE STREAMING)
    # ==========================================
    print(f"\nBuilding Native Truth Data HDF5 (HLS Only): {OUTPUT_NATIVE_HDF5}")

    with h5py.File(OUTPUT_NATIVE_HDF5, 'w') as h5f:
        info_grp = h5f.create_group("HDFEOS INFORMATION")
        info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"
    
        # Configuration Metadata removed

        odl_blocks = []
        odl_blocks.extend(stream_native_tiles_to_hdf5(h5f, 'HLSS30', s30_collections, 10, 11, S30_WAVELENGTHS, safe_bbox))
        odl_blocks.extend(stream_native_tiles_to_hdf5(h5f, 'HLSL30', l30_collections, 7, 8, L30_SR_WAVELENGTHS, safe_bbox))

        if odl_blocks:
            full_odl = "GROUP=SwathStructure\nEND_GROUP=SwathStructure\nGROUP=GridStructure\n" + "\n".join(odl_blocks) + "\nEND_GROUP=GridStructure\nGROUP=PointStructure\nEND_GROUP=PointStructure\nGROUP=ZaStructure\nEND_GROUP=ZaStructure\nEND\n"
            info_grp.create_dataset("StructMetadata.0", shape=(1,), dtype=h5py.string_dtype(encoding='ascii'), data=full_odl)

    print(f"\nPipeline Complete. Native Truth Data HDF5 generated successfully: {OUTPUT_NATIVE_HDF5}")
    return OUTPUT_NATIVE_HDF5


if __name__ == '__main__':
    main()
