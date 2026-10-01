"""
HLST Constellation Orthorectification and Harmonization

Combines HDF-EOS compliant grids in separate HDF5 files using a dynamically
centered Albers Equal Area coordinate grid into a single HDF5 file.
Implements an ROI bounding box for the combined data and a strict spatial
coverage guardrail to reject marginal swath overlaps.

Strictly ported from commit c7d149a3e99d995962bfc6d5839e87d2d3ec75f6
(Harmonized_SC/HLST_constellation_to_hdf5.py, harmonize_hls.py, harmonize_tanager.py).
All helper functions are implemented inline without importing SpecComplex or torch.
"""

import os
import sys
import platform
import json
import glob
import math
import warnings
from pathlib import Path
from datetime import datetime, timezone

# Windows WMI query bypass to prevent subprocess hangs
def _dummy_wmi_query(*args, **kwargs):
    raise OSError("WMI disabled to prevent hangs")
platform._wmi_query = _dummy_wmi_query

import numpy as np
import h5py

def scale_to_int16(arr, scale_factor=10000, nodata_val=-9999):
    import numpy as np
    out = np.full(arr.shape, nodata_val, dtype=np.int16)
    valid = ~np.isnan(arr) & (arr != nodata_val)
    clipped = np.clip(arr[valid] * scale_factor, -32767, 32767)
    out[valid] = np.round(clipped).astype(np.int16)
    return out

def read_scaled_int16(arr, scale_factor=10000, nodata_val=-9999):
    import numpy as np
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = (arr != nodata_val)
    out[valid] = arr[valid].astype(np.float32) / float(scale_factor)
    return out

import rasterio
from rasterio.transform import from_bounds as transform_from_bounds, Affine
from rasterio.warp import reproject, Resampling
from rasterio.control import GroundControlPoint
from pyproj import Transformer, CRS
import scipy.ndimage
import yaml
import hdfeos_odl


# =====================================================================
# CONSTANTS AND SENTINEL VALUES
# =====================================================================
TARGET_RESOLUTION = 30.0
MIN_HLS_ROI_COVERAGE_PERCENT = 20.0
MIN_TANAGER_ROI_COVERAGE_PERCENT = 25.0
SUN_ELEVATION_THRESHOLD = 20
TANAGER_SUN_ELEVATION_THRESHOLD = 30

# HLS Sentinels
HLS_SR_NODATA = -9999
HLS_FMASK_NODATA = 255
HLS_ANGLES_NODATA = 40000

# Tanager Sentinels
TANAGER_NODATA = -9999.0
TANAGER_MASK_NODATA = 255

# Pixel Quality Thresholds
HLS_CLOUD_DILATION = 0
QA_REJECT_MASK = 0b11111  # Bits 0-4: cirrus, cloud, adj cloud/shadow, cloud shadow, snow/ice
AEROSOL_ACCEPT_LEVEL = "medium"  # 'low' (0-1), 'medium' (0-2), 'high' (0-3)

TANAGER_CLOUD_DILATION = 4
TANAGER_UNCERTAINTY_THRESHOLD = 0.10
TANAGER_AEROSOL_THRESHOLD = 0.35

S30_WAVELENGTHS = [0.443, 0.490, 0.560, 0.665, 0.705, 0.740, 0.783, 0.842, 1.610, 2.190]
L30_SR_WAVELENGTHS = [0.443, 0.482, 0.561, 0.655, 0.865, 1.609, 2.201]


# =====================================================================
# INLINE HELPER FUNCTIONS (R7 Isolation: No SpecComplex or Torch)
# =====================================================================

def get_hls_mask(
    fmask_data,
    solar_view_angles=None,
    sun_elevation_threshold=20.0,
    cloud_dilation=0,
    qa_reject_mask=None,
    aerosol_accept_level="medium",
    reject_water=True,
    return_valid=True
):
    """
    Decodes HLS Fmask bit planes and solar angles into a quality mask.

    When called with a single array (fmask_data) and default return_valid=True,
    returns a boolean mask where True = Valid and False = Invalid.

    When called in the legacy style get_hls_mask(data_grp, t, ...) where the first
    argument is a dictionary or HDF5 group, extracts Fmask and solar angles at frame
    index t and returns invalid_mask where True = Invalid/Masked, False = Valid.

    Reference: HLS Product User Guide V2.0, Table 9.
    Fmask bits:
      Bit 0: Cirrus
      Bit 1: Cloud
      Bit 2: Adjacent cloud/shadow
      Bit 3: Cloud shadow
      Bit 4: Snow/ice
      Bit 5: Water
      Bits 6-7: Aerosol level (0=climatology, 1=low, 2=moderate, 3=high)
    """
    # Legacy caller pattern: get_hls_mask(data_grp, t, ...)
    if isinstance(fmask_data, (dict, h5py.Group)):
        data_grp = fmask_data
        t = solar_view_angles if isinstance(solar_view_angles, (int, np.integer)) else 0
        raw_fm = data_grp["Fmask"][t, ...]
        if raw_fm.ndim == 3 and raw_fm.shape[0] == 1:
            raw_fm = raw_fm[0]
        fmask_arr = np.asarray(raw_fm, dtype=np.uint8)

        if "solar_view_angles" in data_grp:
            angles_arr = np.asarray(data_grp["solar_view_angles"][t, ...], dtype=np.float32)
        else:
            angles_arr = None

        if qa_reject_mask is None:
            qa_reject_mask = QA_REJECT_MASK

        return_valid = False
    else:
        fmask_arr = np.asarray(fmask_data, dtype=np.uint8)
        angles_arr = np.asarray(solar_view_angles, dtype=np.float32) if solar_view_angles is not None else None

    # Squeeze leading singleton dimensions if present
    while fmask_arr.ndim > 2 and fmask_arr.shape[0] == 1:
        fmask_arr = fmask_arr[0]

    # 1. Quality flag rejection
    cirrus = (fmask_arr & (1 << 0)) > 0
    cloud = (fmask_arr & (1 << 1)) > 0
    adj_cloud = (fmask_arr & (1 << 2)) > 0
    shadow = (fmask_arr & (1 << 3)) > 0
    snow_ice = (fmask_arr & (1 << 4)) > 0
    water = (fmask_arr & (1 << 5)) > 0
    nodata = (fmask_arr == HLS_FMASK_NODATA)

    if qa_reject_mask is not None:
        qa_invalid = ((fmask_arr & qa_reject_mask) != 0) | nodata
        if reject_water and (qa_reject_mask & (1 << 5)) == 0 and not isinstance(fmask_data, (dict, h5py.Group)):
            qa_invalid = qa_invalid | water
    else:
        qa_invalid = cirrus | cloud | adj_cloud | shadow | snow_ice | water | nodata

    # 2. Aerosol level evaluation (bits 6 and 7)
    aerosol_bits = (fmask_arr >> 6) & 0b11
    if aerosol_accept_level == "low":
        aerosol_invalid = aerosol_bits > 1
    elif aerosol_accept_level == "medium":
        aerosol_invalid = aerosol_bits > 2
    elif aerosol_accept_level == "high":
        aerosol_invalid = aerosol_bits > 3
    else:
        aerosol_invalid = np.zeros_like(qa_invalid, dtype=bool)

    invalid_mask = qa_invalid | aerosol_invalid

    # 3. Solar elevation threshold evaluation
    if angles_arr is not None:
        while angles_arr.ndim > 3 and angles_arr.shape[0] == 1:
            angles_arr = angles_arr[0]
        sza = angles_arr[0, ...]
        sun_elev = 90.0 - sza
        sun_invalid = (sun_elev < sun_elevation_threshold) | np.isnan(sun_elev)
        invalid_mask = invalid_mask | sun_invalid

    # 4. Optional morphological cloud dilation
    if cloud_dilation > 0:
        kernel = np.ones((3, 3), dtype=bool)
        invalid_mask = scipy.ndimage.binary_dilation(invalid_mask, structure=kernel, iterations=cloud_dilation)

    if return_valid:
        return ~invalid_mask
    return invalid_mask


def get_tanager_mask(
    cloud_mask_data,
    nodata_mask_data=None,
    cirrus_mask_data=None,
    shape=None,
    sun_elevation_threshold=30.0,
    cloud_dilation=0,
    apply_cloud_mask=True,
    uncertainty_threshold=0.10,
    aerosol_depth_threshold=0.35,
    return_valid=True
):
    """
    Calculates composite validity mask for Tanager-1 hyperspectral data.

    When called as get_tanager_mask(cloud_mask_data, nodata_mask_data, cirrus_mask_data=None)
    with return_valid=True, returns boolean mask where True = Valid, False = Invalid.

    When called in legacy style get_tanager_mask(data_grp, f_idx, shape, ...) where the
    first argument is a dictionary or HDF5 group, evaluates all Tanager quality layers
    and returns invalid_mask where True = Invalid/Masked, False = Valid.
    """
    # Legacy caller pattern: get_tanager_mask(grp_tanager, out_idx, shape, ...)
    if isinstance(cloud_mask_data, (dict, h5py.Group)):
        data_grp = cloud_mask_data
        f_idx = nodata_mask_data if isinstance(nodata_mask_data, (int, np.integer)) else 0
        if shape is None and isinstance(cirrus_mask_data, tuple):
            eval_shape = cirrus_mask_data
        elif shape is not None:
            eval_shape = shape
        elif "beta_cloud_mask" in data_grp:
            eval_shape = data_grp["beta_cloud_mask"].shape[-2:]
        else:
            eval_shape = (1409, 1070)

        invalid_mask = np.zeros(eval_shape, dtype=bool)
        kernel = np.ones((3, 3), dtype=bool)

        if apply_cloud_mask and "beta_cloud_mask" in data_grp:
            c_mask = (data_grp["beta_cloud_mask"][f_idx, ...] == 1)
            cir_mask = (data_grp["beta_cirrus_mask"][f_idx, ...] == 1) if "beta_cirrus_mask" in data_grp else np.zeros_like(c_mask)
            combined_cloud = c_mask | cir_mask
            if cloud_dilation > 0:
                combined_cloud = scipy.ndimage.binary_dilation(combined_cloud, structure=kernel, iterations=cloud_dilation)
            invalid_mask |= combined_cloud

        if "nodata_pixels" in data_grp:
            nd_mask = (data_grp["nodata_pixels"][f_idx, ...] > 0)
            invalid_mask |= nd_mask

        if "sun_zenith" in data_grp:
            zenith_dset = data_grp["sun_zenith"]
            zenith = zenith_dset[f_idx, ...]
            if hasattr(zenith_dset, "attrs") and "scale_factor" in zenith_dset.attrs:
                zenith = read_scaled_int16(zenith, scale_factor=zenith_dset.attrs["scale_factor"], nodata_val=zenith_dset.attrs["_FillValue"])
            
            sun_invalid = (zenith == TANAGER_NODATA) | np.isnan(zenith)
            if sun_elevation_threshold is not None:
                sun_invalid |= (zenith > (90.0 - sun_elevation_threshold))
            invalid_mask |= sun_invalid

        if "aerosol_optical_depth" in data_grp:
            aod_dset = data_grp["aerosol_optical_depth"]
            aod = aod_dset[f_idx, ...]
            if hasattr(aod_dset, "attrs") and "scale_factor" in aod_dset.attrs:
                aod = read_scaled_int16(aod, scale_factor=aod_dset.attrs["scale_factor"], nodata_val=aod_dset.attrs["_FillValue"])
                
            bad_aod = (aod == TANAGER_NODATA) | np.isnan(aod)
            if aerosol_depth_threshold is not None:
                bad_aod |= (aod >= aerosol_depth_threshold)
            if cloud_dilation > 0:
                bad_aod = scipy.ndimage.binary_dilation(bad_aod, structure=kernel, iterations=cloud_dilation)
            invalid_mask |= bad_aod

        if "surface_reflectance_uncertainty" in data_grp:
            gw = None
            if "surface_reflectance" in data_grp and hasattr(data_grp["surface_reflectance"], "attrs"):
                gw = data_grp["surface_reflectance"].attrs.get("good_wavelengths", None)

            unc_dset = data_grp["surface_reflectance_uncertainty"]
            if gw is not None:
                valid_bands = np.asarray(gw, dtype=bool)
                unc_cube = unc_dset[f_idx, valid_bands, ...]
            else:
                unc_cube = unc_dset[f_idx, ...]
                
            if hasattr(unc_dset, "attrs") and "scale_factor" in unc_dset.attrs:
                unc_cube = read_scaled_int16(unc_cube, scale_factor=unc_dset.attrs["scale_factor"], nodata_val=unc_dset.attrs["_FillValue"])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                unc = np.nanmax(unc_cube, axis=0)

            unc_invalid = (unc == TANAGER_NODATA) | np.isnan(unc)
            if uncertainty_threshold is not None:
                unc_invalid |= (unc >= uncertainty_threshold)
            if cloud_dilation > 0:
                unc_invalid = scipy.ndimage.binary_dilation(unc_invalid, structure=kernel, iterations=cloud_dilation)
            invalid_mask |= unc_invalid

        return invalid_mask

    # Standard array caller
    c_arr = np.asarray(cloud_mask_data)
    invalid = (c_arr > 0)

    if nodata_mask_data is not None:
        nd_arr = np.asarray(nodata_mask_data)
        invalid = invalid | (nd_arr > 0)

    if cirrus_mask_data is not None and not isinstance(cirrus_mask_data, tuple):
        cir_arr = np.asarray(cirrus_mask_data)
        invalid = invalid | (cir_arr > 0)

    if cloud_dilation > 0:
        kernel = np.ones((3, 3), dtype=bool)
        invalid = scipy.ndimage.binary_dilation(invalid, structure=kernel, iterations=cloud_dilation)

    if return_valid:
        return ~invalid
    return invalid


def generate_rgba_image(
    r_band,
    g_band,
    b_band,
    valid_mask=None,
    low=0.5,
    high=99.5,
    gamma=1.1,
    p_min=None,
    p_max=None
):
    """
    Generates an 8-bit RGBA image from RGB reflectance bands with contrast stretching
    and alpha channel opacity.

    Returns:
        rgba_8bit: uint8 array of shape (H, W, 4) with alpha=255 for valid, alpha=0 for invalid.
    """
    r = np.asarray(r_band, dtype=np.float32)
    g = np.asarray(g_band, dtype=np.float32)
    b = np.asarray(b_band, dtype=np.float32)

    spatial_shape = r.shape

    # Handle case where all inputs are NaN
    frame_rgb = np.stack([r, g, b], axis=0)
    if np.all(np.isnan(frame_rgb)):
        out_shape = spatial_shape + (4,) if spatial_shape != () else (4,)
        return np.zeros(out_shape, dtype=np.uint8)

    # Determine pixel validity
    all_zeros = (r == 0) & (g == 0) & (b == 0)
    has_nan = np.isnan(r) | np.isnan(g) | np.isnan(b)
    base_invalid = all_zeros | has_nan

    if valid_mask is not None:
        mask_arr = np.asarray(valid_mask, dtype=bool)
        # If mask_arr is True for valid, invert to invalid
        # Context check: if mask has fewer True than False or vice versa
        # Contract: valid_mask has True for valid pixels
        pixel_valid = mask_arr & (~base_invalid)
    else:
        pixel_valid = ~base_invalid

    # Contrast stretch each band
    rgb_stretched = np.zeros(spatial_shape + (3,), dtype=np.float32)
    bands = [r, g, b]

    for band_idx, band_data in enumerate(bands):
        valid_vals = band_data[pixel_valid]
        if valid_vals.size == 0:
            continue

        if p_min is not None and p_max is not None:
            if p_max > p_min:
                scaled = (band_data - p_min) / (p_max - p_min)
                stretched = np.clip(scaled, 0.0, 1.0)
            else:
                stretched = np.zeros_like(band_data)
        else:
            p_low, p_high = np.percentile(valid_vals, (low, high))
            if p_high > p_low:
                scaled = (band_data - p_low) / (p_high - p_low)
                stretched = np.clip(scaled, 0.0, 1.0)
            else:
                stretched = np.zeros_like(band_data)

        if gamma != 1.0:
            with np.errstate(invalid="ignore", divide="ignore"):
                stretched = np.power(stretched, 1.0 / gamma)
                stretched = np.nan_to_num(stretched, nan=0.0, posinf=1.0, neginf=0.0)

        rgb_stretched[..., band_idx] = stretched

    rgb_8bit = np.clip(rgb_stretched * 255.0, 0, 255).astype(np.uint8)

    # Construct alpha channel
    alpha = np.zeros(spatial_shape, dtype=np.uint8)
    alpha[pixel_valid] = 255

    # Combine into RGBA
    rgba_8bit = np.concatenate([rgb_8bit, np.expand_dims(alpha, axis=-1)], axis=-1)
    return rgba_8bit


# =====================================================================
# MASTER GRID CALCULATION (DYNAMIC ALBERS EQUAL AREA)
# =====================================================================

def calculate_master_grid(bbox, resolution=30.0):
    """
    Calculates a Unified Master Grid using a Dynamically Centered Albers Equal Area projection.
    Standard parallels and central origins are computed using the Deetz & Adams One-Sixth Rule
    to minimize distortion across the region of interest.

    Args:
        bbox (list or tuple): Spatial bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326.
        resolution (float): Target grid resolution in meters (default 30.0m).

    Returns:
        tuple: (dst_crs, transform, width, height, proj_code, zone, gctp_params)
    """
    min_lon, min_lat, max_lon, max_lat = bbox

    # Deetz & Adams One-Sixth Rule
    central_lon = (min_lon + max_lon) / 2.0
    central_lat = (min_lat + max_lat) / 2.0
    lat_1 = min_lat + (max_lat - min_lat) / 6.0
    lat_2 = max_lat - (max_lat - min_lat) / 6.0

    proj_str = (
        f"+proj=aea +lat_1={lat_1:.6f} +lat_2={lat_2:.6f} "
        f"+lat_0={central_lat:.6f} +lon_0={central_lon:.6f} "
        f"+x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs"
    )
    dst_crs = CRS.from_string(proj_str)

    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    xs, ys = transformer.transform(
        [bbox[0], bbox[2], bbox[2], bbox[0]],
        [bbox[3], bbox[3], bbox[1], bbox[1]]
    )
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)

    # GCTP Projection Code 3 = ALBERS, Sphere Code 12 = WGS84
    gctp_params = [
        6378137.0,
        6356752.314245,
        lat_1,
        lat_2,
        central_lon,
        central_lat,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ]

    return dst_crs, transform, width, height, "GCTP_ALBERS", 0, gctp_params


def get_safe_bbox(roi_lon_min, roi_lat_min, roi_lon_max, roi_lat_max):
    """Ensures min/max coordinate ordering for bounding boxes."""
    return [
        min(roi_lon_min, roi_lon_max),
        min(roi_lat_min, roi_lat_max),
        max(roi_lon_min, roi_lon_max),
        max(roi_lat_min, roi_lat_max),
    ]

# =====================================================================
# HLS REPROJECTION AND HARMONIZATION PIPELINE
# =====================================================================

def fetch_native_hls_groups(native_h5_path, sensor_prefix):
    """Scans Native Truth HDF5 and groups temporal frames strictly by acquisition day."""
    if not os.path.exists(native_h5_path):
        raise FileNotFoundError(f"Native HLS Truth file missing at {native_h5_path}")

    daily_groups = {}
    unique_tiles = set()

    with h5py.File(native_h5_path, "r") as h5f:
        if "HDFEOS/GRIDS" not in h5f:
            return daily_groups, unique_tiles

        grid_groups = [k for k in h5f["HDFEOS/GRIDS"].keys() if k.startswith(sensor_prefix)]
        for grid_id in grid_groups:
            parts = grid_id.split("_")
            tile_name = parts[1] if len(parts) > 1 else grid_id
            unique_tiles.add(tile_name)

            sr_path = f"HDFEOS/GRIDS/{grid_id}/Data Fields/surface_reflectance"
            if sr_path not in h5f:
                continue
            sr_ds = h5f[sr_path]
            acq_times = sr_ds.attrs.get("acquisition_time", [])

            for f_idx, ts in enumerate(acq_times):
                dt_str = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
                if dt_str not in daily_groups:
                    daily_groups[dt_str] = []
                daily_groups[dt_str].append({"tile": tile_name, "grid_id": grid_id, "frame_idx": f_idx})

    return daily_groups, unique_tiles


def process_hls_master_stack(
    native_h5_path,
    daily_groups,
    expected_sr,
    master_height,
    master_width,
    master_transform,
    master_crs,
    min_roi_coverage=MIN_HLS_ROI_COVERAGE_PERCENT,
    sun_elev_thresh=SUN_ELEVATION_THRESHOLD,
    cloud_dil=HLS_CLOUD_DILATION,
    qa_reject_mask=QA_REJECT_MASK,
    aerosol_accept_level=AEROSOL_ACCEPT_LEVEL,
    out_h5f=None,
    out_group_path=None,
    wavelengths=None,
    tile_mapping_json=None
):
    """
    Harmonizes unprojected native HLS arrays into the Master Grid directly in-memory.
    Implements two-pass spatial coverage evaluation and nearest-neighbor reprojection.
    """
    sorted_dates = sorted(daily_groups.keys())
    if len(sorted_dates) == 0:
        return None

    # ---- PASS 1: Identify Valid Dates with Sufficient Coverage ----
    valid_dates = []
    with h5py.File(native_h5_path, "r") as h5f:
        for date_str in sorted_dates:
            entries = daily_groups[date_str]
            accum_sr_band0 = np.full((1, master_height, master_width), np.nan, dtype=np.float32)

            for entry in entries:
                fidx = entry["frame_idx"]
                grid_id = entry["grid_id"]
                df_path = f"HDFEOS/GRIDS/{grid_id}/Data Fields"

                sr_node = h5f[f"{df_path}/surface_reflectance"]
                src_tf = Affine.from_gdal(*sr_node.attrs["GeoTransform"])
                src_crs = CRS.from_wkt(sr_node.attrs["spatial_ref"])

                src_sr = sr_node[fidx, 0:1, :, :]
                tmp_sr_band0 = np.full((1, master_height, master_width), np.nan, dtype=np.float32)
                reproject(
                    source=src_sr,
                    destination=tmp_sr_band0,
                    src_transform=src_tf,
                    src_crs=src_crs,
                    dst_transform=master_transform,
                    dst_crs=master_crs,
                    resampling=Resampling.nearest,
                    src_nodata=np.nan,
                    dst_nodata=np.nan
                )

                mask_sr = ~np.isnan(tmp_sr_band0)
                accum_sr_band0[mask_sr] = tmp_sr_band0[mask_sr]

            valid_pixels = np.sum(~np.isnan(accum_sr_band0[0]))
            coverage = (valid_pixels / float(master_height * master_width)) * 100.0
            if coverage >= min_roi_coverage:
                valid_dates.append(date_str)

    num_valid = len(valid_dates)
    if num_valid == 0:
        return None

    # ---- PASS 2: Reproject and Write Valid Dates ----
    grp = out_h5f.create_group(out_group_path)
    if tile_mapping_json is not None:
        import json
        grp.attrs["tile_mapping"] = json.dumps(tile_mapping_json)

    chunk_h = min(master_height, 256)
    chunk_w = min(master_width, 256)
    
    gdal_transform = np.array(
        [master_transform.c, master_transform.a, master_transform.b, master_transform.f, master_transform.d, master_transform.e],
        dtype="float64"
    )

    sr_ds = grp.create_dataset(
        "surface_reflectance", 
        shape=(num_valid, expected_sr, master_height, master_width), 
        dtype="int16", 
        shuffle=True, compression="gzip", compression_opts=6, chunks=(1, 1, chunk_h, chunk_w)
    )
    sr_ds.attrs["_FillValue"] = -32768
    sr_ds.attrs["scale_factor"] = 10000.0
    sr_ds.attrs["spatial_ref"] = master_crs.to_wkt()
    sr_ds.attrs["GeoTransform"] = gdal_transform
    if wavelengths is not None:
        sr_ds.attrs["wavelengths"] = wavelengths
    
    fmask_ds = grp.create_dataset(
        "Fmask", 
        shape=(num_valid, master_height, master_width), 
        dtype="uint8", 
        shuffle=True, compression="gzip", compression_opts=6, chunks=(1, chunk_h, chunk_w)
    )
    fmask_ds.attrs["_FillValue"] = HLS_FMASK_NODATA
    
    ang_ds = grp.create_dataset(
        "solar_view_angles", 
        shape=(num_valid, 4, master_height, master_width), 
        dtype="int16", 
        shuffle=True, compression="gzip", compression_opts=6, chunks=(1, 4, chunk_h, chunk_w)
    )
    ang_ds.attrs["_FillValue"] = -32768
    ang_ds.attrs["scale_factor"] = 10000.0
    ang_ds.attrs["band_order"] = ["SZA", "SAA", "VZA", "VAA"]
    
    vis_ds = grp.create_dataset(
        "ortho_visual", 
        shape=(num_valid, 4, master_height, master_width), 
        dtype="uint8", 
        shuffle=True, compression="gzip", compression_opts=6, chunks=(1, 4, chunk_h, chunk_w)
    )
    vis_ds.attrs["spatial_ref"] = master_crs.to_wkt()
    vis_ds.attrs["GeoTransform"] = gdal_transform
    
    mask_ds = grp.create_dataset(
        "common_mask", 
        shape=(num_valid, master_height, master_width), 
        dtype=bool, 
        shuffle=True, compression="gzip", compression_opts=6, chunks=(1, chunk_h, chunk_w)
    )
    mask_ds.attrs["description"] = "True = Invalid/Masked, False = Valid."
    mask_ds.attrs["spatial_ref"] = master_crs.to_wkt()
    mask_ds.attrs["GeoTransform"] = gdal_transform

    meta_arrays = {"acq": [], "space": [], "saz": [], "sel": [], "cc": []}
    
    with h5py.File(native_h5_path, "r") as h5f:
        for idx, date_str in enumerate(valid_dates):
            entries = daily_groups[date_str]

            base_grid = entries[0]["grid_id"]
            base_fidx = entries[0]["frame_idx"]
            base_path = f"HDFEOS/GRIDS/{base_grid}/Data Fields/surface_reflectance"
            meta_arrays["acq"].append(h5f[base_path].attrs["acquisition_time"][base_fidx])

            raw_spacecraft = h5f[base_path].attrs["spacecraft_id"][base_fidx]
            spacecraft_str = raw_spacecraft.decode("utf-8") if isinstance(raw_spacecraft, bytes) else str(raw_spacecraft)
            meta_arrays["space"].append(spacecraft_str)
            meta_arrays["cc"].append(h5f[base_path].attrs["cloud_cover"][base_fidx])

            frame_sr = np.full((expected_sr, master_height, master_width), np.nan, dtype=np.float32)
            frame_fm = np.full((master_height, master_width), HLS_FMASK_NODATA, dtype=np.uint8)
            frame_ag = np.full((4, master_height, master_width), np.nan, dtype=np.float32)
            
            for entry in entries:
                fidx = entry["frame_idx"]
                grid_id = entry["grid_id"]
                df_path = f"HDFEOS/GRIDS/{grid_id}/Data Fields"

                sr_node = h5f[f"{df_path}/surface_reflectance"]
                src_tf = Affine.from_gdal(*sr_node.attrs["GeoTransform"])
                src_crs = CRS.from_wkt(sr_node.attrs["spatial_ref"])

                src_sr = sr_node[fidx]
                tmp_sr = np.full((expected_sr, master_height, master_width), np.nan, dtype=np.float32)
                reproject(
                    source=src_sr,
                    destination=tmp_sr,
                    src_transform=src_tf,
                    src_crs=src_crs,
                    dst_transform=master_transform,
                    dst_crs=master_crs,
                    resampling=Resampling.nearest,
                    src_nodata=np.nan,
                    dst_nodata=np.nan
                )
                mask_sr = ~np.isnan(tmp_sr)
                frame_sr[mask_sr] = tmp_sr[mask_sr]

                src_fm = h5f[f"{df_path}/Fmask"][fidx]
                if src_fm.ndim == 3 and src_fm.shape[0] == 1:
                    src_fm = src_fm[0]
                elif src_fm.ndim > 2:
                    src_fm = np.squeeze(src_fm)
                tmp_fm = np.full((master_height, master_width), HLS_FMASK_NODATA, dtype=np.uint8)
                reproject(
                    source=src_fm,
                    destination=tmp_fm,
                    src_transform=src_tf,
                    src_crs=src_crs,
                    dst_transform=master_transform,
                    dst_crs=master_crs,
                    resampling=Resampling.nearest,
                    src_nodata=HLS_FMASK_NODATA,
                    dst_nodata=HLS_FMASK_NODATA
                )
                mask_fm = (tmp_fm != HLS_FMASK_NODATA)
                frame_fm[mask_fm] = tmp_fm[mask_fm]

                src_ag = h5f[f"{df_path}/solar_view_angles"][fidx]
                tmp_ag = np.full((4, master_height, master_width), np.nan, dtype=np.float32)
                reproject(
                    source=src_ag,
                    destination=tmp_ag,
                    src_transform=src_tf,
                    src_crs=src_crs,
                    dst_transform=master_transform,
                    dst_crs=master_crs,
                    resampling=Resampling.nearest,
                    src_nodata=np.nan,
                    dst_nodata=np.nan
                )
                mask_ag = ~np.isnan(tmp_ag)
                frame_ag[mask_ag] = tmp_ag[mask_ag]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mean_sza = np.nanmean(frame_ag[0])
                mean_saa = np.nanmean(frame_ag[1])

            meta_arrays["saz"].append(mean_saa)
            meta_arrays["sel"].append(90.0 - mean_sza)

            temp_grp = {"Fmask": frame_fm[np.newaxis, ...], "solar_view_angles": frame_ag[np.newaxis, ...]}
            frame_mask = get_hls_mask(
                temp_grp,
                0,
                sun_elevation_threshold=sun_elev_thresh,
                cloud_dilation=cloud_dil,
                qa_reject_mask=qa_reject_mask,
                aerosol_accept_level=aerosol_accept_level,
                return_valid=False
            ).astype(bool)

            rgba_img = generate_rgba_image(
                r_band=frame_sr[3, :, :],
                g_band=frame_sr[2, :, :],
                b_band=frame_sr[1, :, :],
                valid_mask=~frame_mask
            )
            vis_frame = np.transpose(rgba_img, (2, 0, 1))

            sr_ds[idx] = scale_to_int16(frame_sr)
            fmask_ds[idx] = frame_fm
            ang_ds[idx] = scale_to_int16(frame_ag)
            vis_ds[idx] = vis_frame
            mask_ds[idx] = frame_mask

    sr_ds.attrs["acquisition_time"] = np.array(meta_arrays["acq"], dtype="float64")
    sr_ds.attrs["spacecraft_id"] = np.array(meta_arrays["space"], dtype="S")
    sr_ds.attrs["sun_azimuth"] = np.array(meta_arrays["saz"], dtype="int16")
    sr_ds.attrs["sun_elevation"] = np.array(meta_arrays["sel"], dtype="int16")
    sr_ds.attrs["cloud_cover"] = np.array(meta_arrays["cc"], dtype="int16")

    return num_valid


def write_hdf_sensor_group(h5f, group_path, data_dict, wavelengths, crs, transform, tile_mapping_json=None):
    """Writes reprojected HLS datasets into HDF-EOS compliant group."""
    if not data_dict or data_dict["count"] == 0:
        return

    grp = h5f.create_group(group_path)
    gdal_transform = np.array(
        [transform.c, transform.a, transform.b, transform.f, transform.d, transform.e],
        dtype="float64"
    )
    dt = h5py.string_dtype(encoding="ascii")

    num_frames, bands, h, w = data_dict["sr"].shape
    chunk_h, chunk_w = min(h, 256), min(w, 256)

    # 1. Surface Reflectance
    sr_ds = grp.create_dataset(
        "surface_reflectance",
        data=scale_to_int16(data_dict["sr"]),
        shuffle=True, compression="gzip",
        compression_opts=6,
        chunks=(1, bands, chunk_h, chunk_w)
    )
    sr_ds.attrs["units"] = "Reflectance"
    sr_ds.attrs["_FillValue"] = -32768
    sr_ds.attrs["scale_factor"] = 10000.0
    sr_ds.attrs["wavelengths"] = wavelengths
    sr_ds.attrs["spatial_ref"] = crs.to_wkt()
    sr_ds.attrs["GeoTransform"] = gdal_transform
    sr_ds.attrs.create("spacecraft_id", data=data_dict["meta"]["space"], dtype=dt)
    sr_ds.attrs["acquisition_time"] = np.array(data_dict["meta"]["acq"], dtype="float64")
    sr_ds.attrs["sun_azimuth"] = np.array(data_dict["meta"]["saz"], dtype="int16")
    sr_ds.attrs["sun_elevation"] = np.array(data_dict["meta"]["sel"], dtype="int16")
    sr_ds.attrs["cloud_cover"] = np.array(data_dict["meta"]["cc"], dtype="int16")

    # 2. Fmask
    fmask_ds = grp.create_dataset(
        "Fmask",
        data=data_dict["fm"][:, 0, :, :],
        dtype="uint8",
        shuffle=True, compression="gzip",
        compression_opts=6,
        chunks=(1, chunk_h, chunk_w)
    )
    fmask_ds.attrs["_FillValue"] = HLS_FMASK_NODATA

    # 3. Solar View Angles
    ang_ds = grp.create_dataset(
        "solar_view_angles",
        data=scale_to_int16(data_dict["ag"]),
        shuffle=True, compression="gzip",
        compression_opts=6,
        chunks=(1, 4, chunk_h, chunk_w)
    )
    ang_ds.attrs["_FillValue"] = -32768
    ang_ds.attrs["scale_factor"] = 10000.0
    ang_ds.attrs["band_order"] = ["SZA", "SAA", "VZA", "VAA"]

    # 4. Ortho Visual
    vis_ds = grp.create_dataset(
        "ortho_visual",
        data=data_dict["vis"],
        dtype="uint8",
        shuffle=True, compression="gzip",
        compression_opts=6,
        chunks=(1, 4, chunk_h, chunk_w)
    )
    vis_ds.attrs["spatial_ref"] = crs.to_wkt()
    vis_ds.attrs["GeoTransform"] = gdal_transform

    # 5. Common Mask (True = Invalid/Masked, False = Valid)
    mask_ds = grp.create_dataset(
        "common_mask",
        data=data_dict["mask"],
        dtype=bool,
        shuffle=True, compression="gzip",
        compression_opts=6,
        chunks=(1, chunk_h, chunk_w)
    )
    mask_ds.attrs["description"] = "True = Invalid/Masked, False = Valid."
    mask_ds.attrs["spatial_ref"] = crs.to_wkt()
    mask_ds.attrs["GeoTransform"] = gdal_transform
    mask_ds.attrs["qa_reject_mask"] = QA_REJECT_MASK
    mask_ds.attrs["cloud_dilation"] = HLS_CLOUD_DILATION
    mask_ds.attrs["aerosol_accept_level"] = AEROSOL_ACCEPT_LEVEL
    mask_ds.attrs["sun_elevation_threshold"] = SUN_ELEVATION_THRESHOLD


# =====================================================================
# TANAGER SWATH ORTHORECTIFICATION PIPELINE
# =====================================================================

def process_tanager_swaths_to_grid(
    h5f,
    tanager_source_dir,
    master_height,
    master_width,
    master_crs,
    master_transform,
    min_roi_coverage=MIN_TANAGER_ROI_COVERAGE_PERCENT,
    sun_elev_thresh=TANAGER_SUN_ELEVATION_THRESHOLD,
    cloud_dil=TANAGER_CLOUD_DILATION,
    uncert_thresh=TANAGER_UNCERTAINTY_THRESHOLD,
    aero_thresh=TANAGER_AEROSOL_THRESHOLD
):
    """
    Translates Tanager-1 Basic Swaths (unrectified pushbroom hyperspectral swaths)
    into the master Albers Equal Area grid using Thin Plate Spline (TPS) Ground Control Points.
    """
    basic_files = glob.glob(os.path.join(tanager_source_dir, "**", "*_basic_sr_hdf5.h5"), recursive=True)
    if not basic_files:
        return None

    if "HDFEOS/GRIDS/TANAGER" in h5f:
        del h5f["HDFEOS/GRIDS/TANAGER"]

    passes = {}
    for f in basic_files:
        basename = os.path.basename(f)
        parts = basename.split("_")
        if len(parts) >= 1:
            pass_ts = parts[0]
            if pass_ts not in passes:
                passes[pass_ts] = []
            passes[pass_ts].append(f)

    pass_keys = sorted(list(passes.keys()))
    total_num_frames = len(pass_keys)
    if total_num_frames == 0:
        return None

    # Inspect first chunk for band count and dataset structure
    with h5py.File(passes[pass_keys[0]][0], "r") as f_test:
        sr_test = f_test["HDFEOS/SWATHS/HYP/Data Fields/surface_reflectance"]
        band_count = sr_test.shape[0]

    grp_tanager = h5f.create_group("HDFEOS/GRIDS/TANAGER/Data Fields")
    chunk_h, chunk_w = min(master_height, 256), min(master_width, 256)
    gdal_transform = np.array(
        [master_transform.c, master_transform.a, master_transform.b, master_transform.f, master_transform.d, master_transform.e],
        dtype="float64"
    )

    datasets_created_info = []
    meta_lists = {"acq_time": [], "space_id": [], "good_wavelengths": []}

    original_dtypes = {}
    original_fills = {}

    # Initialize datasets based on swath structure
    with h5py.File(passes[pass_keys[0]][0], "r") as f_meta:
        src_df = f_meta["HDFEOS/SWATHS/HYP/Data Fields"]
        for name in src_df.keys():
            src_dset = src_df[name]
            original_dtype = src_dset.dtype
            original_fill = src_dset.attrs.get("_FillValue", TANAGER_NODATA)
            
            original_dtypes[name] = original_dtype
            original_fills[name] = original_fill
            
            is_3d = len(src_dset.shape) == 3
            bands = src_dset.shape[0] if is_3d else None

            out_shape = (total_num_frames, bands, master_height, master_width) if is_3d else (total_num_frames, master_height, master_width)
            chunks = (1, bands, chunk_h, chunk_w) if is_3d else (1, chunk_h, chunk_w)

            if original_dtype.kind in ['f', 'c'] and original_dtype != np.float64:
                dtype = np.dtype("int16")
                fill_val = -9999
            else:
                dtype = original_dtype
                fill_val = original_fill

            dset = grp_tanager.create_dataset(
                name,
                shape=out_shape,
                dtype=dtype,
                shuffle=True, compression="gzip",
                compression_opts=6,
                fillvalue=fill_val,
                chunks=chunks
            )
            
            if original_dtype.kind in ['f', 'c'] and original_dtype != np.float64:
                dset.attrs["scale_factor"] = 10000
                dset.attrs["_FillValue"] = -9999

            dim_names = ["Time", "Band", "YDim", "XDim"] if is_3d else ["Time", "YDim", "XDim"]
            datasets_created_info.append((name, dtype, len(out_shape), dim_names))

    valid_t_indices = []

    for t_idx, pass_ts in enumerate(pass_keys):
        chunks_files = passes[pass_ts]
        pass_canvases = {}
        pass_times = []

        for name in grp_tanager.keys():
            orig_dtype = original_dtypes[name]
            is_3d = len(grp_tanager[name].shape) == 4
            bands = grp_tanager[name].shape[1] if is_3d else None
            canvas_shape = (bands, master_height, master_width) if is_3d else (master_height, master_width)
            orig_fill = original_fills[name]
            pass_canvases[name] = np.full(canvas_shape, orig_fill, dtype=orig_dtype)

        meta_lists["space_id"].append("Tanager-1")
        gw_found = False

        for chunk_idx, chunk_file in enumerate(chunks_files):
            with h5py.File(chunk_file, "r") as f_chunk:
                df_grp = f_chunk["HDFEOS/SWATHS/HYP/Data Fields"]
                geo_grp = f_chunk["HDFEOS/SWATHS/HYP/Geolocation Fields"]
                lat = geo_grp["Latitude"][:]
                lon = geo_grp["Longitude"][:]
                pass_times.extend(geo_grp["Time"][:].tolist())

                if chunk_idx == 0:
                    gw = df_grp["surface_reflectance"].attrs.get("good_wavelengths")
                    if gw is not None:
                        meta_lists["good_wavelengths"].append(gw)
                        gw_found = True

                # Construct Ground Control Points (GCPs) sampled on a 10-pixel grid
                gcps = []
                step = 10
                rows = list(range(0, lat.shape[0], step))
                if rows[-1] != lat.shape[0] - 1:
                    rows.append(lat.shape[0] - 1)
                cols = list(range(0, lat.shape[1], step))
                if cols[-1] != lat.shape[1] - 1:
                    cols.append(lat.shape[1] - 1)

                for r in rows:
                    for c in cols:
                        gcps.append(GroundControlPoint(row=r, col=c, x=lon[r, c], y=lat[r, c]))

                for name in df_grp.keys():
                    if chunk_idx == 0:
                        for attr_name, attr_val in df_grp[name].attrs.items():
                            if attr_name not in grp_tanager[name].attrs:
                                grp_tanager[name].attrs[attr_name] = attr_val

                    is_3d = len(grp_tanager[name].shape) == 4
                    bands = grp_tanager[name].shape[1] if is_3d else None
                    dtype = df_grp[name].dtype
                    fill_val = original_fills[name]

                    src_data = df_grp[name][:]
                    if not is_3d:
                        src_data = src_data[np.newaxis, ...]
                        incoming = np.full((1, master_height, master_width), fill_val, dtype=dtype)
                    else:
                        incoming = np.full((bands, master_height, master_width), fill_val, dtype=dtype)

                    reproject(
                        source=src_data,
                        destination=incoming,
                        src_transform=None,
                        gcps=gcps,
                        src_crs="EPSG:4326",
                        dst_transform=master_transform,
                        dst_crs=master_crs,
                        resampling=Resampling.nearest,
                        src_nodata=fill_val,
                        dst_nodata=fill_val,
                        tps=True
                    )

                    if dtype.kind in ["f", "c"] and np.isnan(fill_val):
                        valid_mask = ~np.isnan(incoming)
                    else:
                        valid_mask = ~np.isclose(incoming, fill_val, equal_nan=True)

                    if not is_3d:
                        valid_mask = valid_mask[0]
                        pass_canvases[name][valid_mask] = incoming[0][valid_mask]
                    else:
                        pass_canvases[name][valid_mask] = incoming[valid_mask]

        if len(pass_times) > 0:
            meta_lists["acq_time"].append(np.mean(pass_times))
        else:
            meta_lists["acq_time"].append(0.0)

        if not gw_found:
            meta_lists["good_wavelengths"].append(np.zeros(band_count, dtype=bool))

        sr_fill = original_fills["surface_reflectance"]
        sr_canvas = pass_canvases["surface_reflectance"]
        valid_sr = ~np.isclose(sr_canvas[0], sr_fill, equal_nan=True)
        sr_valid_pixels = np.sum(valid_sr)

        for name in pass_canvases.keys():
            if original_dtypes[name].kind in ['f', 'c'] and original_dtypes[name] != np.float64:
                grp_tanager[name][t_idx, ...] = scale_to_int16(pass_canvases[name], scale_factor=10000, nodata_val=-9999)
            else:
                grp_tanager[name][t_idx, ...] = pass_canvases[name]

        coverage = (sr_valid_pixels / float(master_height * master_width)) * 100.0
        if coverage >= min_roi_coverage:
            valid_t_indices.append(t_idx)

    # Write global attributes
    dt_str = h5py.string_dtype(encoding="ascii")
    grp_tanager["surface_reflectance"].attrs["acquisition_time"] = np.array(meta_lists["acq_time"], dtype="float64")
    grp_tanager["surface_reflectance"].attrs.create("spacecraft_id", data=np.array(meta_lists["space_id"], dtype=dt_str))
    if len(meta_lists["good_wavelengths"]) == total_num_frames:
        grp_tanager["surface_reflectance"].attrs["all_good_wavelengths"] = np.array(meta_lists["good_wavelengths"], dtype=bool)

    if len(valid_t_indices) > 0:
        # Generate Common Mask
        mask_ds = grp_tanager.create_dataset(
            "common_mask",
            shape=(total_num_frames, master_height, master_width),
            dtype=bool,
            shuffle=True, compression="gzip",
            compression_opts=6,
            chunks=(1, chunk_h, chunk_w)
        )
        datasets_created_info.append(("common_mask", bool, 3, ["Time", "YDim", "XDim"]))
        mask_ds.attrs["spatial_ref"] = master_crs.to_wkt()
        mask_ds.attrs["GeoTransform"] = gdal_transform
        mask_ds.attrs["description"] = "True = Invalid/Masked, False = Valid."

        for out_idx in range(total_num_frames):
            invalid_mask = get_tanager_mask(
                grp_tanager,
                out_idx,
                shape=(master_height, master_width),
                sun_elevation_threshold=sun_elev_thresh,
                cloud_dilation=cloud_dil,
                apply_cloud_mask=True,
                uncertainty_threshold=uncert_thresh,
                aerosol_depth_threshold=aero_thresh,
                return_valid=False
            )
            mask_ds[out_idx, ...] = invalid_mask

        # Generate True Color RGBA visual composite
        wavelengths = grp_tanager["surface_reflectance"].attrs["wavelengths"]
        r_idx = int(np.argmin(np.abs(wavelengths - 650)))
        g_idx = int(np.argmin(np.abs(wavelengths - 550)))
        b_idx = int(np.argmin(np.abs(wavelengths - 450)))

        ortho_vis_dset = grp_tanager.create_dataset(
            "ortho_visual",
            shape=(total_num_frames, 4, master_height, master_width),
            dtype="uint8",
            shuffle=True, compression="gzip",
            compression_opts=6,
            fillvalue=0,
            chunks=(1, 4, chunk_h, chunk_w)
        )
        datasets_created_info.append(("ortho_visual", np.dtype("uint8"), 4, ["Time", "RGBABand", "YDim", "XDim"]))
        ortho_vis_dset.attrs["spatial_ref"] = master_crs.to_wkt()
        ortho_vis_dset.attrs["GeoTransform"] = gdal_transform

        sr_dset_ref = grp_tanager["surface_reflectance"]
        for out_idx in range(total_num_frames):
            r_band = sr_dset_ref[out_idx, r_idx, :, :]
            g_band = sr_dset_ref[out_idx, g_idx, :, :]
            b_band = sr_dset_ref[out_idx, b_idx, :, :]

            r_input = np.where(r_band < -1, np.nan, r_band)
            g_input = np.where(g_band < -1, np.nan, g_band)
            b_input = np.where(b_band < -1, np.nan, b_band)

            rgba_img = generate_rgba_image(
                r_band=r_input,
                g_band=g_input,
                b_band=b_input,
                valid_mask=~mask_ds[out_idx]
            )
            ortho_vis_dset[out_idx, ...] = np.transpose(rgba_img, (2, 0, 1))

        return datasets_created_info, total_num_frames, band_count

    return None


# =====================================================================
# CONFIGURATION LOADER
# =====================================================================

def load_locations_config(config_path=None):
    """Loads location bounding box and sensor availability configuration."""
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    search_dirs = [
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent.parent,
        Path.cwd(),
    ]
    for d in search_dirs:
        p = d / "locations_config.yaml"
        if p.exists():
            with open(p, "r") as f:
                return yaml.safe_load(f)

    # Standard fallback configuration for Rochesterv2 baseline
    return {
        "current_run": {"location": "Rochesterv2"},
        "locations": {
            "Rochesterv2": {
                "SOURCE_CACHE": None,
                "ROI_LON_MIN": -77.770166,
                "ROI_LON_MAX": -77.376776,
                "ROI_LAT_MIN": 42.961778,
                "ROI_LAT_MAX": 43.342135,
                "START_DATE": "2022-01-01",
                "END_DATE": "2026-06-01",
                "TANAGER_AVAILABLE": True,
            }
        }
    }


# =====================================================================
# MAIN PIPELINE EXECUTION
# =====================================================================

def main(target_location=None, input_native_h5=None, tanager_source_dir=None, output_master_h5=None, config_override=None):
    """
    Executes Albers Equal Area orthorectification and constellation harmonization.
    Produces the base HLST_{Location}_Harmonized.h5 file.
    """
    if config_override is not None:
        config_data = config_override
    else:
        config_data = load_locations_config()

    if target_location is not None:
        location = target_location
    else:
        location = config_data.get("current_run", {}).get("location", "Rochesterv2")

    config = config_data["locations"][location]
    source_cache = config.get("SOURCE_CACHE", location)
    if source_cache is None:
        source_cache = location

    roi_lon_min = config["ROI_LON_MIN"]
    roi_lon_max = config["ROI_LON_MAX"]
    roi_lat_min = config["ROI_LAT_MIN"]
    roi_lat_max = config["ROI_LAT_MAX"]
    tanager_available = config.get("TANAGER_AVAILABLE", False)

    # Establish Master Grid via Albers Equal Area Conic
    safe_bbox = get_safe_bbox(roi_lon_min, roi_lat_min, roi_lon_max, roi_lat_max)
    master_crs, master_transform, master_width, master_height, master_proj, master_zone, master_gctp = calculate_master_grid(
        safe_bbox, TARGET_RESOLUTION
    )

    # Establish Directories
    hls_source_dir = "C:/satelliteImagery/HLS30/"
    tanager_dir = tanager_source_dir or f"C:/satelliteImagery/Tanager/{source_cache}_SourceData"
    output_dir = "C:/satelliteImagery/HLST30/"

    native_h5 = input_native_h5 or os.path.join(hls_source_dir, f"HLS_{location}_STAC_Native_2025.h5")
    master_h5 = output_master_h5 or os.path.join(output_dir, f"HLST_{location}_Harmonized.h5")
    os.makedirs(os.path.dirname(os.path.abspath(master_h5)), exist_ok=True)

    print(f"Building Multi-Sensor Harmonized ARD Cube: {master_h5}")
    print(f"Master Grid: {master_width}x{master_height} at 30m Albers Equal Area")

    # Group HLS frames by date
    s30_daily, s30_tiles = {}, set()
    l30_daily, l30_tiles = {}, set()
    if os.path.exists(native_h5):
        s30_daily, s30_tiles = fetch_native_hls_groups(native_h5, "HLSS30")
        l30_daily, l30_tiles = fetch_native_hls_groups(native_h5, "HLSL30")

    unique_hls_tiles = sorted(list(s30_tiles.union(l30_tiles)))
    master_tile_mapping = {tile: i + 1 for i, tile in enumerate(unique_hls_tiles)}
    master_tile_mapping_json = json.dumps(master_tile_mapping)

    with h5py.File(master_h5, "w") as h5f:
        info_grp = h5f.create_group("HDFEOS INFORMATION")
        info_grp.attrs["HDFEOSVersion"] = "HDFEOS_5.1.16"

        # Configuration Metadata removed

        odl_blocks = []

        # 1. Harmonize HLSS30
        if len(s30_daily) > 0:
            print("Harmonizing HLSS30...")
            s30_count = process_hls_master_stack(
                native_h5,
                s30_daily,
                10,
                master_height,
                master_width,
                master_transform,
                master_crs,
                min_roi_coverage=MIN_HLS_ROI_COVERAGE_PERCENT,
                sun_elev_thresh=SUN_ELEVATION_THRESHOLD,
                cloud_dil=HLS_CLOUD_DILATION,
                qa_reject_mask=QA_REJECT_MASK,
                aerosol_accept_level=AEROSOL_ACCEPT_LEVEL,
                out_h5f=h5f,
                out_group_path="/HDFEOS/GRIDS/HLSS30/Data Fields",
                wavelengths=S30_WAVELENGTHS,
                tile_mapping_json=master_tile_mapping_json
            )
            if s30_count > 0:
                odl_blocks.append(
                    hdfeos_odl.generate_hls_odl_grid_string(
                        "HLSS30", master_width, master_height, master_transform,
                        master_proj, master_zone, master_gctp, 10, s30_count
                    )
                )

        # 2. Harmonize HLSL30
        if len(l30_daily) > 0:
            print("Harmonizing HLSL30...")
            l30_count = process_hls_master_stack(
                native_h5,
                l30_daily,
                7,
                master_height,
                master_width,
                master_transform,
                master_crs,
                min_roi_coverage=MIN_HLS_ROI_COVERAGE_PERCENT,
                sun_elev_thresh=SUN_ELEVATION_THRESHOLD,
                cloud_dil=HLS_CLOUD_DILATION,
                qa_reject_mask=QA_REJECT_MASK,
                aerosol_accept_level=AEROSOL_ACCEPT_LEVEL,
                out_h5f=h5f,
                out_group_path="/HDFEOS/GRIDS/HLSL30/Data Fields",
                wavelengths=L30_SR_WAVELENGTHS,
                tile_mapping_json=master_tile_mapping_json
            )
            if l30_count > 0:
                odl_blocks.append(
                    hdfeos_odl.generate_hls_odl_grid_string(
                        "HLSL30", master_width, master_height, master_transform,
                        master_proj, master_zone, master_gctp, 7, l30_count
                    )
                )

        # 3. Harmonize Tanager-1 (From Basic Swaths)
        if tanager_available:
            print("Harmonizing TANAGER swaths...")
            res = process_tanager_swaths_to_grid(
                h5f,
                tanager_dir,
                master_height,
                master_width,
                master_crs,
                master_transform,
                min_roi_coverage=MIN_TANAGER_ROI_COVERAGE_PERCENT,
                sun_elev_thresh=TANAGER_SUN_ELEVATION_THRESHOLD,
                cloud_dil=TANAGER_CLOUD_DILATION,
                uncert_thresh=TANAGER_UNCERTAINTY_THRESHOLD,
                aero_thresh=TANAGER_AEROSOL_THRESHOLD
            )
            if res is not None:
                datasets_created_info, total_num_frames, band_count = res
                odl_blocks.append(
                    hdfeos_odl.generate_dynamic_odl_grid_string(
                        "TANAGER", master_width, master_height, master_transform,
                        master_proj, master_zone, master_gctp, datasets_created_info,
                        total_num_frames, band_count
                    )
                )

        # Finalize StructMetadata.0 ODL string
        odl_string = "\n".join(odl_blocks)
        info_grp.create_dataset(
            "StructMetadata.0",
            data=np.array(odl_string, dtype=h5py.string_dtype(encoding="ascii"))
        )

    print(f"Master Harmonized ARD Cube successfully generated: {master_h5}")
    return master_h5


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Albers Equal Area Harmonization and Orthorectification")
    parser.add_argument("--location", type=str, default="Rochesterv2", help="Target location name")
    parser.add_argument("--native_h5", type=str, default=None, help="Input native HLS HDF5 path")
    parser.add_argument("--tanager_dir", type=str, default=None, help="Input Tanager source directory")
    parser.add_argument("--output_h5", type=str, default=None, help="Output master Harmonized HDF5 path")
    args = parser.parse_args()

    main(
        target_location=args.location,
        input_native_h5=args.native_h5,
        tanager_source_dir=args.tanager_dir,
        output_master_h5=args.output_h5
    )
