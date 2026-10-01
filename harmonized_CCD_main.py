"""
Harmonized Continuous Change Detection (CCD) Pipeline.

Ports continuous change detection harmonic regression modeling using LassoLarsIC
and Recursive Least Squares (RLS) from commit c7d149a3e99d995962bfc6d5839e87d2d3ec75f6.
Maintains pre-MGRS coordinate reference and dataset handling matching the Albers grid.
Inlines read_scaled_int16.
Zero imports of torch or legacy SpecComplex.
"""

import os
import h5py
import numpy as np
import datetime
import math
from tqdm import tqdm
from sklearn.linear_model import LassoLarsIC
import multiprocessing
import joblib
import contextlib
from scipy.stats import chi2
from joblib import Parallel, delayed


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def read_scaled_int16(arr, scale_factor=10000, nodata_val=-32768):
    """
    Invert int16 quantized array back to float32 reflectance.
    Preserves nodata sentinels as NaN without injecting synthetic fill values.
    """
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    valid = (arr != nodata_val)
    out[valid] = arr[valid].astype(np.float32) / float(scale_factor)
    return out


def scale_to_int16(arr, scale_factor=10000, nodata_val=-32768):
    """
    Quantize float array to int16 format.
    Preserves nodata sentinels without synthetic smoothing.
    """
    out = np.full(arr.shape, nodata_val, dtype=np.int16)
    valid = ~np.isnan(arr) & (arr != nodata_val)
    clipped = np.clip(arr[valid] * scale_factor, -32767, 32767)
    out[valid] = np.round(clipped).astype(np.int16)
    return out


# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input/Output
LOCATION = "Rochesterv2"
H5_PATH = f"C:/satelliteImagery/HLST30/HLST_{LOCATION}_Harmonized_SC_EM-7_Norm-None.h5"
TARGET_METRIC = 'sliding_volume_z_score'

if TARGET_METRIC == 'sliding_volume_z_score':
    TARGET_NAME = 'Spectral Complexity (Z-Score)'
elif TARGET_METRIC == 'sliding_volume_robust_scale':
    TARGET_NAME = 'Spectral Complexity (Robust)'
elif TARGET_METRIC == 'ndvi':
    TARGET_NAME = 'NDVI'
elif TARGET_METRIC == 'ndbi':
    TARGET_NAME = 'NDBI'
else:
    TARGET_NAME = TARGET_METRIC

# Anomaly Detection Thresholds
CHANGE_PROBABILITY = 0.99
CONSECUTIVE_ANOMALIES = 4
CHI2_DEGREES_OF_FREEDOM = 3

# Segment Initialization & Stability Constraints
MAX_RMSE = 1.0
MIN_RMSE_CLAMP = 1e-5
MIN_YEARS_FOR_INIT = 1.5
MIN_SAMPLES = 16

# Harmonic Regression Configurations
ENABLE_CONSTANT = True
ENABLE_LINEAR = False
ENABLE_QUADRATIC = False
TEMPORAL_PERIODS = [0.33, 0.5, 1]

# RLS Tuning
RLS_RIDGE_PENALTY = 1e-6
RLS_FORGETTING_FACTOR = 0.999


def extract_fractional_years(acq_times):
    """Converts UNIX timestamps into continuous fractional years (t)."""
    frac_years = []
    for dt in acq_times:
        val = float(dt)
        if 1970.0 <= val <= 2100.0:
            frac_years.append(val)
            continue
        try:
            dt_obj = datetime.datetime.fromtimestamp(val, tz=datetime.timezone.utc)
            year = dt_obj.year
            start_of_year = datetime.datetime(year, 1, 1, tzinfo=datetime.timezone.utc)
            start_of_next = datetime.datetime(year + 1, 1, 1, tzinfo=datetime.timezone.utc)
            year_duration = (start_of_next - start_of_year).total_seconds()
            elapsed = (dt_obj - start_of_year).total_seconds()
            frac_years.append(year + (elapsed / year_duration))
        except (OSError, OverflowError, ValueError):
            frac_years.append(1970.0 + (val / 31557600.0))
    return np.array(frac_years)


def build_harmonic_matrix(t, temporal_periods=None,
                          enable_const=ENABLE_CONSTANT,
                          enable_lin=ENABLE_LINEAR,
                          enable_quad=ENABLE_QUADRATIC):
    """
    Constructs a Fourier basis matrix with configurable polynomial trends and temporal periods.
    """
    if temporal_periods is None:
        temporal_periods = TEMPORAL_PERIODS

    cols = []

    if enable_const:
        cols.append(np.ones_like(t))

    if enable_lin:
        cols.append(t)
    if enable_quad:
        cols.append(t**2)

    w = 2.0 * math.pi
    for p in temporal_periods:
        cols.append(np.cos((w / p) * t))
        cols.append(np.sin((w / p) * t))

    return np.column_stack(cols) if cols else np.zeros((len(t), 0))


def _process_row_chunk(chunk_args):
    """Worker function for processing a row chunk across temporal frames."""
    (y_start, y_end, width, y_data_chunk, valid_mask_chunk,
     frac_years, X_full, acq_times, min_samples, change_prob, consec_anom) = chunk_args

    num_frames = y_data_chunk.shape[0]
    chunk_height = y_end - y_start
    num_features = X_full.shape[1]

    chunk_pred = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_rmse = np.full((num_frames, chunk_height, width), np.nan, dtype=np.float32)
    chunk_coef = np.full((num_frames, chunk_height, width, num_features), np.nan, dtype=np.float32)
    chunk_flags = np.zeros((num_frames, chunk_height, width), dtype=np.uint8)
    chunk_date = np.full((chunk_height, width), np.nan, dtype=np.float64)
    chunk_count = np.zeros((chunk_height, width), dtype=np.int32)

    chi2_threshold = chi2.ppf(change_prob, df=CHI2_DEGREES_OF_FREEDOM)

    for y_local in range(chunk_height):
        for x in range(width):
            pixel_valid = valid_mask_chunk[:, y_local, x]
            valid_indices = np.where(pixel_valid)[0]

            if len(valid_indices) <= min_samples:
                continue

            i = 0
            while i < len(valid_indices) - min_samples:
                # 1. Initialize segment
                segment_start_idx = i
                train_end = segment_start_idx + min_samples

                # Minimum time span constraint for stable initialization
                while train_end <= len(valid_indices):
                    time_span = frac_years[valid_indices[train_end - 1]] - frac_years[valid_indices[segment_start_idx]]
                    if time_span >= MIN_YEARS_FOR_INIT:
                        break
                    train_end += 1

                if train_end > len(valid_indices):
                    # Could not satisfy minimum initialization time span
                    break

                active_pool = list(range(segment_start_idx, train_end))
                min_dof = X_full.shape[1] + 1
                absolute_min_samples = max(min_dof, min_samples)

                initialization_successful = False
                lasso_model = LassoLarsIC(criterion='bic', fit_intercept=True)

                while len(active_pool) >= absolute_min_samples:
                    X_train = X_full[valid_indices[active_pool], :]
                    Y_train = y_data_chunk[valid_indices[active_pool], y_local, x]

                    try:
                        lasso_model.fit(X_train, Y_train)
                        y_train_pred = lasso_model.predict(X_train)
                        rmse = np.sqrt(np.mean((Y_train - y_train_pred)**2))
                        rmse = max(rmse, MIN_RMSE_CLAMP)
                    except Exception:
                        break

                    errors = np.abs(Y_train - y_train_pred)
                    max_error_idx = int(np.argmax(errors))
                    max_error = errors[max_error_idx]

                    chi2_val_init = (max_error / rmse)**2
                    if chi2_val_init > chi2_threshold:
                        active_pool.pop(max_error_idx)
                    else:
                        initialization_successful = True
                        break

                if not initialization_successful:
                    i += 1
                    continue

                # Hand-off to OLS for RLS initialization
                active_features = np.where(lasso_model.coef_ != 0)[0]

                # If constant term is enabled, force it to be active
                if ENABLE_CONSTANT and 0 not in active_features:
                    active_features = np.insert(active_features, 0, 0)

                if len(active_features) == 0:
                    active_features = np.array([0])

                X_train_active = X_full[valid_indices[active_pool]][:, active_features]
                Y_train_active = y_data_chunk[valid_indices[active_pool], y_local, x]

                # Initialize Full RLS State (track all terms)
                X_train_all = X_full[valid_indices[active_pool]]
                num_total_features = X_train_all.shape[1]

                try:
                    # OLS only on active features to prevent overfitting on initialization
                    P_active = np.linalg.inv(X_train_active.T @ X_train_active + np.eye(len(active_features)) * RLS_RIDGE_PENALTY)
                    theta_active = P_active @ X_train_active.T @ Y_train_active

                    # Map to full theta vector (inactive terms start at 0)
                    theta = np.zeros(num_total_features)
                    theta[active_features] = theta_active

                    # Initialize P matrix on all features so RLS can adapt them later
                    P = np.linalg.inv(X_train_all.T @ X_train_all + np.eye(num_total_features) * RLS_RIDGE_PENALTY)
                except np.linalg.LinAlgError:
                    i += 1
                    continue

                Y_pred_init = X_train_all @ theta
                SSE = np.sum((Y_train_active - Y_pred_init)**2)
                dof = len(active_features)
                n_points = len(active_pool)
                rmse = np.sqrt(SSE / max(1, n_points - dof))
                rmse = max(rmse, MIN_RMSE_CLAMP)

                # Transient Event Rejection: If the initialization window is volatile, slide forward
                if rmse > MAX_RMSE:
                    i += 1
                    continue

                # Backfill predictions for initialization window
                for global_idx in valid_indices[segment_start_idx:train_end]:
                    x_target = X_full[global_idx, :]
                    chunk_pred[global_idx, y_local, x] = np.dot(x_target, theta)
                    chunk_rmse[global_idx, y_local, x] = rmse
                    chunk_coef[global_idx, y_local, x, :] = theta

                consecutive_count = 0
                break_detected = False

                # 2. Forward Expanding Phase
                for j in range(train_end, len(valid_indices)):
                    target_idx = valid_indices[j]

                    x_target = X_full[target_idx, :]
                    y_pred = np.dot(x_target, theta)

                    actual = y_data_chunk[target_idx, y_local, x]
                    error = abs(actual - y_pred)

                    chunk_pred[target_idx, y_local, x] = y_pred
                    chunk_rmse[target_idx, y_local, x] = rmse
                    chunk_coef[target_idx, y_local, x, :] = theta

                    chi2_val = (error / rmse)**2
                    is_anomaly = chi2_val > chi2_threshold

                    if is_anomaly:
                        chunk_flags[target_idx, y_local, x] = 1
                        consecutive_count += 1

                        if consecutive_count >= consec_anom:
                            chunk_count[y_local, x] += 1
                            if np.isnan(chunk_date[y_local, x]):
                                first_anomaly_idx = valid_indices[j - consec_anom + 1]
                                chunk_date[y_local, x] = acq_times[first_anomaly_idx]

                            break_detected = True
                            break
                    else:
                        consecutive_count = 0
                        # RLS Update Step
                        x_target_2d = x_target.reshape(-1, 1)
                        Px = P @ x_target_2d
                        denom = RLS_FORGETTING_FACTOR + (x_target_2d.T @ Px)[0, 0]
                        K = Px / denom

                        e = actual - y_pred

                        theta = theta + (K.flatten() * e)
                        P = (P - (K @ (x_target_2d.T @ P))) / RLS_FORGETTING_FACTOR
                        SSE = SSE + (e**2) / denom
                        n_points += 1
                        new_rmse = np.sqrt(SSE / max(1, n_points - dof))

                if break_detected:
                    i = j - consec_anom + 1
                else:
                    break

    return y_start, y_end, chunk_pred, chunk_rmse, chunk_coef, chunk_flags, chunk_date, chunk_count


def resolve_h5_input(location=LOCATION, h5_path=None):
    """Finds existing HDF5 file containing the target harmonized datasets."""
    if h5_path and os.path.exists(h5_path):
        return h5_path

    candidates = [
        f"C:/satelliteImagery/HLST30/HLST_{location}_Harmonized.h5",
        f"C:/satelliteImagery/HLST30/HLST_{location}_Harmonized_SC_EM-7_Norm-None.h5",
        rf"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data\HLST_{location}_Harmonized.h5",
        rf"F:\Resilio\IMGS 890 Research\JSTARS-Spectral-Complexity\source_data\HLST_{location}_Harmonized_SC_EM-7_Norm-None.h5",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return H5_PATH


def main(enable_const=ENABLE_CONSTANT,
         enable_lin=ENABLE_LINEAR,
         enable_quad=ENABLE_QUADRATIC,
         temporal_periods=None,
         target_metric=TARGET_METRIC,
         launch_vis=True,
         location=LOCATION,
         h5_path=None,
         output_h5=None,
         n_jobs=None,
         chunk_size=4):
    """
    Main driver for continuous change detection regression analysis.
    """
    if temporal_periods is None:
        temporal_periods = TEMPORAL_PERIODS

    min_samples = MIN_SAMPLES

    if h5_path is None:
        h5_path = resolve_h5_input(location=location, h5_path=H5_PATH)

    _term_str = f"C{int(enable_const)}L{int(enable_lin)}Q{int(enable_quad)}"
    _period_str = f"P{len(temporal_periods)}"

    if output_h5 is None:
        output_dir = "C:/satelliteImagery/HLST30/CCD"
        output_h5 = os.path.join(
            output_dir,
            f"{location}_CCD_Harmonized_Change_Detection_{target_metric}_{_term_str}_{_period_str}.h5"
        )

    print(f"Loading data from {h5_path}...")
    with h5py.File(h5_path, 'r') as f:
        data_grp = f['/HDFEOS/GRIDS/HARMONIZED/Data Fields']
        metric_ds = data_grp[target_metric]

        acq_times = metric_ds.attrs.get('acquisition_time')
        if acq_times is None and 'acquisition_time' in data_grp.attrs:
            acq_times = data_grp.attrs['acquisition_time']
        elif acq_times is None and 'common_mask' in data_grp and 'acquisition_time' in data_grp['common_mask'].attrs:
            acq_times = data_grp['common_mask'].attrs['acquisition_time']
        elif acq_times is None and 'acquisition_time' in f.attrs:
            acq_times = f.attrs['acquisition_time']

        if acq_times is not None:
            acq_times = np.array(acq_times[:], dtype=np.float64)

        y_data = metric_ds[...]
        if np.issubdtype(y_data.dtype, np.integer):
            y_data = read_scaled_int16(y_data)

        # Determine valid mask
        common_mask = data_grp['common_mask'][...]
        valid_mask = (common_mask == 0) & ~np.isnan(y_data)

        geo_transform = metric_ds.attrs.get('GeoTransform')
        if geo_transform is None and 'GeoTransform' in f.attrs:
            geo_transform = f.attrs['GeoTransform']

        spatial_ref = metric_ds.attrs.get('spatial_ref')
        if spatial_ref is None and 'spatial_ref' in f.attrs:
            spatial_ref = f.attrs['spatial_ref']

    num_frames, height, width = y_data.shape

    if acq_times is None:
        acq_times = np.linspace(1640995200.0, 1767225600.0, num_frames, dtype=np.float64)

    frac_years = extract_fractional_years(acq_times)

    # Sort chronologically
    sort_idx = np.argsort(acq_times)
    acq_times = acq_times[sort_idx]
    frac_years = frac_years[sort_idx]
    y_data = y_data[sort_idx, ...]
    valid_mask = valid_mask[sort_idx, ...]

    print(f"Dataset shape: {num_frames} frames, {height}x{width} pixels")

    # Output arrays
    change_date_map = np.zeros((height, width), dtype=np.float64)
    change_date_map[:] = np.nan
    change_count_map = np.zeros((height, width), dtype=np.int32)

    predicted_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    rmse_series = np.full((num_frames, height, width), np.nan, dtype=np.float32)
    anomaly_flags = np.zeros((num_frames, height, width), dtype=np.uint8)

    X_full = build_harmonic_matrix(frac_years, temporal_periods=temporal_periods,
                                   enable_const=enable_const, enable_lin=enable_lin, enable_quad=enable_quad)

    num_features = X_full.shape[1]
    coef_series = np.full((num_frames, height, width, num_features), np.nan, dtype=np.float32)

    print("\nExecuting Sliding Window LASSO Harmonic Regression...")

    if n_jobs is None:
        n_jobs = min(8, multiprocessing.cpu_count())
    print(f"Using {n_jobs} cores for parallel processing.")

    num_chunks = max(1, math.ceil(height / chunk_size))
    chunks = []
    for y_start in range(0, height, chunk_size):
        y_end = min(y_start + chunk_size, height)
        chunk_args = (
            y_start, y_end, width,
            y_data[:, y_start:y_end, :], valid_mask[:, y_start:y_end, :],
            frac_years, X_full, acq_times,
            min_samples, CHANGE_PROBABILITY, CONSECUTIVE_ANOMALIES
        )
        chunks.append(chunk_args)

    if n_jobs > 1 and len(chunks) > 1:
        with tqdm_joblib(tqdm(desc="Processing row chunks", total=len(chunks))):
            results = Parallel(n_jobs=n_jobs, backend='loky', return_as='generator')(
                delayed(_process_row_chunk)(chunk) for chunk in chunks
            )
        for y_start, y_end, c_pred, c_rmse, c_coef, c_flags, c_date, c_count in results:
            predicted_series[:, y_start:y_end, :] = c_pred
            rmse_series[:, y_start:y_end, :] = c_rmse
            coef_series[:, y_start:y_end, :, :] = c_coef
            anomaly_flags[:, y_start:y_end, :] = c_flags
            change_date_map[y_start:y_end, :] = c_date
            change_count_map[y_start:y_end, :] = c_count
    else:
        for chunk in tqdm(chunks, desc="Processing row chunks"):
            y_start, y_end, c_pred, c_rmse, c_coef, c_flags, c_date, c_count = _process_row_chunk(chunk)
            predicted_series[:, y_start:y_end, :] = c_pred
            rmse_series[:, y_start:y_end, :] = c_rmse
            coef_series[:, y_start:y_end, :, :] = c_coef
            anomaly_flags[:, y_start:y_end, :] = c_flags
            change_date_map[y_start:y_end, :] = c_date
            change_count_map[y_start:y_end, :] = c_count

    print(f"\nSaving Results directly to {h5_path}...")
    with h5py.File(h5_path, 'a') as out_file:
        if 'CCD_Results' in out_file:
            del out_file['CCD_Results']
        ccd_grp = out_file.create_group('CCD_Results')

        if spatial_ref is not None:
            ccd_grp.attrs['spatial_ref'] = spatial_ref
        if geo_transform is not None:
            ccd_grp.attrs['GeoTransform'] = geo_transform

        ccd_grp.attrs['CHANGE_PROBABILITY'] = CHANGE_PROBABILITY
        ccd_grp.attrs['CONSECUTIVE_ANOMALIES'] = CONSECUTIVE_ANOMALIES
        ccd_grp.attrs['CHI2_DEGREES_OF_FREEDOM'] = CHI2_DEGREES_OF_FREEDOM
        ccd_grp.attrs['MAX_RMSE'] = MAX_RMSE
        ccd_grp.attrs['MIN_RMSE_CLAMP'] = MIN_RMSE_CLAMP
        ccd_grp.attrs['MIN_YEARS_FOR_INIT'] = MIN_YEARS_FOR_INIT
        ccd_grp.attrs['MIN_SAMPLES'] = min_samples
        ccd_grp.attrs['TEMPORAL_PERIODS'] = temporal_periods
        ccd_grp.attrs['ENABLE_CONSTANT'] = enable_const
        ccd_grp.attrs['ENABLE_LINEAR'] = enable_lin
        ccd_grp.attrs['ENABLE_QUADRATIC'] = enable_quad
        ccd_grp.attrs['TARGET_METRIC'] = target_metric
        ccd_grp.attrs['SOURCE_DATA'] = h5_path

        ccd_grp.create_dataset('predicted_series', data=predicted_series, shuffle=True, compression='gzip', compression_opts=6)
        ccd_grp.create_dataset('rmse_series', data=rmse_series, shuffle=True, compression='gzip', compression_opts=6)
        ccd_grp.create_dataset('coef_series', data=coef_series, shuffle=True, compression='gzip', compression_opts=6)
        ccd_grp.create_dataset('anomaly_flags', data=anomaly_flags, shuffle=True, compression='gzip', compression_opts=6)
        ccd_grp.create_dataset('change_date_timestamp', data=change_date_map, shuffle=True, compression='gzip', compression_opts=6)
        ccd_grp.create_dataset('change_count', data=change_count_map, shuffle=True, compression='gzip', compression_opts=6)

    print("Harmonized CCD Pipeline Complete!")

    if launch_vis:
        print("\nLaunching visualization...")
        try:
            from harmonized_CCD_vis import plot_spatial_anomaly_overlay
            plot_spatial_anomaly_overlay(h5_path)
        except ImportError as e:
            print(f"Could not import visualization module: {e}")
        except Exception as e:
            print(f"An error occurred while launching visualization: {e}")

    return h5_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Harmonized CCD Main")
    parser.add_argument('--location', type=str, default=LOCATION, help="Location name")
    parser.add_argument('--h5-path', type=str, default=None, help="Input source HDF5 path")
    parser.add_argument('--output-h5', type=str, default=None, help="Output CCD HDF5 path")
    parser.add_argument('--metric', type=str, default=TARGET_METRIC, help="Target metric")
    parser.add_argument('--no-vis', action='store_true', help="Skip launching visualization")
    args = parser.parse_args()

    main(
        location=args.location,
        h5_path=args.h5_path,
        output_h5=args.output_h5,
        target_metric=args.metric,
        launch_vis=not args.no_vis
    )
