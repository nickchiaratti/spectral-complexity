import os
import math
import numpy as np
from scipy import ndimage
from scipy.ndimage import uniform_filter
from warnings import simplefilter


def maximumDistance(data, num_endmembers, chunk_size=50000):
    '''
    Memory-optimized MaxD geometric simplex extraction.
    Utilizes strict float32 typing and chunked vector broadcasting to 
    prevent ArrayMemoryErrors on hyperspectral (300+ band) datasets.
    
    Returns:
        endmembers [bands, num_endmembers]
        endmembers_index [1, num_endmembers]
    '''      
    image2D = np.reshape(data, (data.shape[0] * data.shape[1], data.shape[2]), order="F")

    if np.min(image2D) < 0:
        warnings.warn('Data contains negative values')
        image2D = np.clip(image2D, 0, 2)
    if np.max(image2D) > 1:
        warnings.warn('Data contains values greater than 1')
        image2D = np.clip(image2D, 0, 1)

    valid_mask = ~np.isnan(image2D).any(axis=1)
    
    # Strict NaN Enforcement: Reject entirely if ANY pixels are missing
    if not valid_mask.all():
        return np.full((image2D.shape[1], num_endmembers), np.nan), np.full((1, num_endmembers), np.nan)
        
    if np.sum(valid_mask) < num_endmembers:
        return np.full((image2D.shape[1], num_endmembers), np.nan), np.full((1, num_endmembers), np.nan)

    valid_data = image2D[valid_mask].astype(np.float32)
    valid_indices = np.where(valid_mask)[0]

    data_t = np.transpose(valid_data)
    num_bands, num_pix = data_t.shape

    magnitude = np.linalg.norm(data_t, axis=0)
    idx1 = np.argmax(magnitude)
    idx2 = np.argmin(magnitude)

    endmembers = np.zeros([num_bands, num_endmembers], dtype=np.float32)
    endmembers_index = np.zeros([1, num_endmembers], dtype=int)   

    endmembers[:, 0] = data_t[:, idx1]
    endmembers[:, 1] = data_t[:, idx2]
    
    endmembers_index[0, 0] = valid_indices[idx1]
    endmembers_index[0, 1] = valid_indices[idx2]

    # Pre-allocate strictly as float32 to prevent memory doubling
    data_proj = data_t.copy()
    identity_matrix = np.identity(num_bands, dtype=np.float32)

    for i in range(2, num_endmembers):
        diff = data_proj[:, idx2:idx2+1] - data_proj[:, idx1:idx1+1]
        
        # Enforce float32 on the pseudoinverse to prevent float64 upcasting during matmul
        pseudo = np.linalg.pinv(diff).astype(np.float32)
        proj_operator = (identity_matrix - np.matmul(diff, pseudo)).astype(np.float32)

        # EVIDENCE-BASED FIX: Chunked In-Place Projection
        # Applies the projection matrix in memory-safe chunks rather than generating 
        # a new massive array across the entire image space simultaneously.
        for c in range(0, num_pix, chunk_size):
            c_end = min(c + chunk_size, num_pix)
            data_proj[:, c:c_end] = np.matmul(proj_operator, data_proj[:, c:c_end])

        idx1 = idx2
        vec = data_proj[:, idx2:idx2+1] 
            
        # EVIDENCE-BASED FIX: Chunked Distance Calculation
        # Prevents NumPy from allocating a massive intermediate array for np.square()
        diff_new = np.zeros(num_pix, dtype=np.float32)
        for c in range(0, num_pix, chunk_size):
            c_end = min(c + chunk_size, num_pix)
            chunk = data_proj[:, c:c_end]
            diff_new[c:c_end] = np.sum(np.square(vec - chunk), axis=0)
            
        idx2 = np.argmax(diff_new)

        endmembers[:, i] = data_t[:, idx2]
        endmembers_index[0, i] = valid_indices[idx2]

    return endmembers, endmembers_index

def calcGramLocalVolumes(endmembers, localization_vector):
    """
    Calculates the Local Gram matrix.
    1. Subtracts the localization vector from all other endmembers (centering the simplex on x).
    2. Calculates the Gram matrix of these centered vectors.
    3. Calculates the parallelotope volume estimate for 1 through N endmembers
    4. Returns volume values in an array of length N
    """
    # Reduce to current number of endmembers
    # Shape: (Bands, N)
    localized_vectors = endmembers - localization_vector[:, np.newaxis]

    # Calculate Gram Matrix
    # G = V^T * V (Shape: N x N)
    gram = np.matmul(localized_vectors.T, localized_vectors)
    
    # Initialize array to store the volume sequence
    N = gram.shape[0]
    volumes = np.zeros(N)
    
    # Calculate the parallelotope volume estimate for 1 through N endmembers
    for i in range(1, N + 1):
        # Extract the i x i top-left submatrix
        sub_gram = gram[:i, :i]
        
        # Calculate the Gramian determinant
        det = np.linalg.det(sub_gram)
        
        # Guard against floating-point inaccuracies that can cause tiny 
        # negative determinants near the linear dependence threshold
        if det < 0:
            det = 0.0
            
    # Volume is the square root of the Gramian determinant
        volumes[i-1] = np.sqrt(det)
        
    return volumes

def process_volume_sliding_tile(frame_data, tile_size, stride, num_endmembers):
    """
    Sliding window processing.
    Strict Validity: Window is only processed if ALL pixels are valid.
    Output is masked with NaN for any pixel identified as invalid.
    """
    bands, height, width = frame_data.shape
    img = np.transpose(frame_data, (1, 2, 0))
    
    out_h = (height - tile_size) // stride + 1
    out_w = (width - tile_size) // stride + 1
    center_offset = tile_size // 2
    
    neighborhood_map = np.full((height, width), np.nan, dtype=np.float32)
    
    for y_start in range(0, height - tile_size + 1, stride):
        for x_start in range(0, width - tile_size + 1, stride):
            y_end, x_end = y_start + tile_size, x_start + tile_size
            
            tile = img[y_start:y_end, x_start:x_end, :]
            
            # Pre-emptive validity check to prevent inpainting/smearing
            if np.isnan(tile).any():
                continue
                
            tile_flat = np.reshape(tile, (tile_size * tile_size, bands), order="F").T
            endmembers, volumes = maximumDistance_volumes(tile_flat, num_endmembers)
            
            if len(volumes) > 2:
                vol_val = np.max(volumes[2:])
            else:
                vol_val = 0.0
                
            cy = y_start + center_offset
            cx = x_start + center_offset
            neighborhood_map[cy, cx] = vol_val

    clean_neighborhood = np.nan_to_num(neighborhood_map, nan=0.0)
    valid_binary = ~np.isnan(neighborhood_map)
    
    sum_map = uniform_filter(clean_neighborhood, size=tile_size, mode='constant', cval=0.0) 
    count_map = uniform_filter(valid_binary.astype(np.float32), size=tile_size, mode='constant', cval=0.0)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        final_map = np.where(count_map > 0, sum_map / count_map, np.nan)
        
    return final_map

def calculate_global_z_score(volume_array, valid_pixel_mask):
    """
    Calculates the global Z-score for an entire frame of spectral complexity volumes.
    Decouples the evaluation space from the background statistics space by strictly 
    calculating the mean and standard deviation from radiometrically valid pixels, 
    preventing artifacts from skewing the background model.
    """
    #print("Calculating global Z-score for frame")
    height, width = volume_array.shape
    z_scores = np.full((height, width), np.nan, dtype=np.float32)
    
    # Identify globally valid pixels (strictly positive for log transform)
    global_valid_mask = volume_array > 0.0
    
    # Intersect with radiometrically valid pixels for the statistical background model
    stats_mask = global_valid_mask & valid_pixel_mask
    
    # Extract subset volumes strictly for statistical estimation
    stats_vols = volume_array[stats_mask]

    # Graceful fallback per user directive: Return NaNs for entire frame if no valid background exists.
    if len(stats_vols) < 9 or len(np.unique(stats_vols)) <= 1:
        warnings.warn("calculate_global_z_score warning: Insufficient valid pixels (< 9) or zero variance found. Returning NaNs.")
        return z_scores, np.nan, np.nan
        
    log_stats_vols = np.log(stats_vols)
    
    # Calculate global scene statistics (using ddof=1 for unbiased sample estimator)
    global_mean = np.mean(log_stats_vols)
    global_std = np.std(log_stats_vols, ddof=1)
    
    # Strict failure handling: Prevent training on synthetically flat frames
    if global_std == 0:
        raise ValueError("calculate_global_z_score failed: Global standard deviation of the radiometrically valid subset is exactly zero.")
        
    # Evaluate ALL geometrically valid pixels using the pure background model
    apply_vols = volume_array[global_valid_mask]
    log_apply_vols = np.log(apply_vols)
    
    # Apply standard Z-score equation
    z_scores[global_valid_mask] = (log_apply_vols - global_mean) / global_std
    
    return z_scores, global_mean, global_std