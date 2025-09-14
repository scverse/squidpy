import numpy as np
import dask.array as da
import enum
import xarray as xr
from dask.diagnostics import ProgressBar
from scipy import ndimage as ndi
from scipy.fft import fft2, fftfreq
import itertools
from typing import Literal, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
from spatialdata._logging import logger as logg
import pandas as pd
from anndata import AnnData
from spatialdata.models import TableModel, ShapesModel
from shapely.geometry import Polygon
import geopandas as gpd
from enum import Enum
from numba import njit
import numba

# Disable Numba parallelization globally to avoid conflicts with Dask
numba.set_num_threads(1)

def _ensure_yxc(img_da: xr.DataArray) -> xr.DataArray:
    """
    Ensure dims are (y, x, c). SpatialData often uses (c, y, x).
    Adds a length-1 "c" if missing.
    """
    dims = list(img_da.dims)
    if "y" not in dims or "x" not in dims:
        raise ValueError(f"Expected dims to include \"y\" and \"x\". Found dims={dims}")
    
    # Handle case where dims are (c, y, x) - transpose to (y, x, c)
    if "c" in dims and dims[0] == "c":
        return img_da.transpose("y", "x", "c")
    elif "c" in dims:
        return img_da.transpose("y", "x", "c")
    
    # If no "c" dimension, add one
    return img_da.expand_dims({"c": [0]}).transpose("y", "x", "c")


def _get_image_data(sdata, image_key: str, scale: str = "scale0"):
    """
    Extract image data from SpatialData object, handling both datatree and direct DataArray images.
    
    Parameters
    ----------
    sdata : SpatialData
        SpatialData object
    image_key : str
        Key in sdata.images
    scale : str
        Multiscale level, e.g. "scale0"
    
    Returns
    -------
    xr.DataArray
        Image data in (y, x, c) format
    """
    img_node = sdata.images[image_key]
    
    # Check if the image is a datatree (has multiple scales) or a direct DataArray
    if hasattr(img_node, 'keys') and len(img_node.keys()) > 1:
        # It's a datatree with multiple scales
        available_scales = list(img_node.keys())
        
        if scale not in available_scales:
            logg.warning(f"Scale '{scale}' not found. Available scales: {available_scales}")
            if available_scales:
                scale = available_scales[0]  # Use first available scale
                logg.info(f"Using scale '{scale}' instead")
            else:
                raise ValueError(f"No scales available for image '{image_key}'")
        
        img_da = img_node[scale].image
    else:
        # It's a direct DataArray (single scale)
        if hasattr(img_node, 'image'):
            img_da = img_node.image
        else:
            # The node itself is the DataArray
            img_da = img_node
    
    # Ensure proper format
    return _ensure_yxc(img_da)


def _to_gray_dask_yx(img_yxc: xr.DataArray, weights=(0.2126, 0.7152, 0.0722)) -> da.Array:
    """
    Dask-native grayscale conversion. Expects (y, x, c).
    For RGBA, ignores alpha; for single-channel, returns it as-is.
    """
    arr = img_yxc.data
    if arr.ndim != 3:
        raise ValueError("Expected a 3D array (y, x, c).")
    c = arr.shape[2]
    if c == 1:
        return arr[..., 0].astype(np.float32, copy=False)
    rgb = arr[..., :3].astype(np.float32, copy=False)
    w = da.from_array(np.asarray(weights, dtype=np.float32), chunks=(3,))
    gray = da.tensordot(rgb, w, axes=([2], [0]))  # -> (y, x)
    return gray.astype(np.float32, copy=False)


@njit(fastmath=True, cache=True, parallel=False)
def _sobel_energy_numba(block_2d: np.ndarray) -> np.ndarray:
    """
    Numba-optimized kernel for Sobel energy (Tenengrad): gx^2 + gy^2.
    Simplified version for better memory efficiency.
    Returns per-pixel energy values (not normalized by tile size).
    """
    h, w = block_2d.shape
    out = np.empty_like(block_2d, dtype=np.float32)
    
    # Simplified Sobel implementation with boundary handling
    for i in range(h):
        for j in range(w):
            # Calculate Gx (vertical gradient) - simplified
            gx = 0.0
            if j > 0 and j < w - 1:
                # Use central difference for vertical gradient
                gx = block_2d[i, j+1] - block_2d[i, j-1]
            elif j == 0:
                gx = block_2d[i, j+1] - block_2d[i, j]
            else:  # j == w - 1
                gx = block_2d[i, j] - block_2d[i, j-1]
            
            # Calculate Gy (horizontal gradient) - simplified
            gy = 0.0
            if i > 0 and i < h - 1:
                # Use central difference for horizontal gradient
                gy = block_2d[i+1, j] - block_2d[i-1, j]
            elif i == 0:
                gy = block_2d[i+1, j] - block_2d[i, j]
            else:  # i == h - 1
                gy = block_2d[i, j] - block_2d[i-1, j]
            
            # Calculate Sobel energy: gx^2 + gy^2
            energy = gx * gx + gy * gy
            
            # Clip to prevent overflow
            if energy > 1e6:
                energy = 1e6
            elif energy < 0.0:
                energy = 0.0
                
            out[i, j] = energy
    
    return out


def _sobel_energy_np(block_2d: np.ndarray) -> np.ndarray:
    """
    NumPy kernel for Sobel energy (Tenengrad): gx^2 + gy^2.
    """
    gx = ndi.sobel(block_2d, axis=0, mode="reflect")
    gy = ndi.sobel(block_2d, axis=1, mode="reflect")
    out = gx.astype(np.float32) ** 2 + gy.astype(np.float32) ** 2
    return out


@njit(fastmath=True, cache=True, parallel=False)
def _laplace_square_numba(block_2d: np.ndarray) -> np.ndarray:
    """
    Numba-optimized kernel for variance of Laplacian.
    Calculates the Laplacian at each pixel, then returns the variance of all Laplacian values.
    """
    h, w = block_2d.shape
    n = h * w
    
    # First pass: calculate Laplacian at each pixel
    laplacian_values = np.empty((h, w), dtype=np.float32)
    
    for i in range(h):
        for j in range(w):
            # Calculate Laplacian using second differences
            lap = 0.0
            
            # Horizontal second difference
            if j > 0 and j < w - 1:
                lap += block_2d[i, j+1] - 2.0 * block_2d[i, j] + block_2d[i, j-1]
            elif j == 0:
                lap += block_2d[i, j+1] - block_2d[i, j]
            else:  # j == w - 1
                lap += block_2d[i, j] - block_2d[i, j-1]
            
            # Vertical second difference
            if i > 0 and i < h - 1:
                lap += block_2d[i+1, j] - 2.0 * block_2d[i, j] + block_2d[i-1, j]
            elif i == 0:
                lap += block_2d[i+1, j] - block_2d[i, j]
            else:  # i == h - 1
                lap += block_2d[i, j] - block_2d[i-1, j]
            
            laplacian_values[i, j] = lap
    
    # Second pass: calculate mean of Laplacian values
    total = 0.0
    for i in range(h):
        for j in range(w):
            total += laplacian_values[i, j]
    mean_lap = total / n
    
    # Third pass: calculate variance of Laplacian values
    var_sum = 0.0
    for i in range(h):
        for j in range(w):
            diff = laplacian_values[i, j] - mean_lap
            var_sum += diff * diff
    var_lap = var_sum / n
    
    # Clip to prevent overflow
    if var_lap > 1e6:
        var_lap = 1e6
    elif var_lap < 0.0:
        var_lap = 0.0
    
    # Fill output array with variance value
    out = np.empty_like(block_2d, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = var_lap
    
    return out


def _laplace_square_np(block_2d: np.ndarray) -> np.ndarray:
    """
    NumPy kernel for variance of Laplacian.
    """
    lap = ndi.laplace(block_2d, mode="reflect").astype(np.float32)
    var_lap = np.var(lap, dtype=np.float32)
    return np.full_like(block_2d, var_lap, dtype=np.float32)


@njit(fastmath=True, cache=True, parallel=False)
def _variance_numba(block_2d: np.ndarray) -> np.ndarray:
    """
    Numba-optimized kernel for variance metric.
    Returns the variance of the block as a constant array.
    """
    h, w = block_2d.shape
    n = h * w
    
    # Calculate mean
    total = 0.0
    for i in range(h):
        for j in range(w):
            total += block_2d[i, j]
    mean_val = total / n
    
    # Calculate variance
    var_sum = 0.0
    for i in range(h):
        for j in range(w):
            diff = block_2d[i, j] - mean_val
            var_sum += diff * diff
    var_val = var_sum / n
    
    
    # Fill output array with variance value
    out = np.empty_like(block_2d, dtype=np.float32)
    for i in range(h):
        for j in range(w):
            out[i, j] = var_val
    
    return out


def _variance_np(block_2d: np.ndarray) -> np.ndarray:
    """
    NumPy kernel for variance metric.
    Returns the variance of the block as a constant array.
    """
    var_val = np.var(block_2d, dtype=np.float32)
    return np.full_like(block_2d, var_val, dtype=np.float32)


def _fft_high_freq_energy_np(block_2d: np.ndarray) -> np.ndarray:
    """
    NumPy kernel for FFT high-frequency energy ratio.
    Calculates ratio of high-frequency energy to total energy.
    """
    # Normalize input to prevent overflow
    block_normalized = block_2d.astype(np.float64)  # Use float64 for precision
    block_mean = np.mean(block_normalized)
    block_std = np.std(block_normalized)
    
    if block_std > 0:
        block_normalized = (block_normalized - block_mean) / block_std
    else:
        block_normalized = block_normalized - block_mean
    
    # Compute 2D FFT
    fft_coeffs = fft2(block_normalized)
    fft_magnitude = np.abs(fft_coeffs)
    
    # Create frequency grids
    h, w = block_2d.shape
    freq_y = fftfreq(h)
    freq_x = fftfreq(w)
    freq_grid_y, freq_grid_x = np.meshgrid(freq_y, freq_x, indexing='ij')
    freq_radius = np.sqrt(freq_grid_y**2 + freq_grid_x**2)
    
    # Define high-frequency mask (exclude DC and very low frequencies)
    # Use 10% of Nyquist frequency as threshold
    high_freq_mask = freq_radius > 0.1
    
    # Calculate energies with overflow protection
    fft_mag_sq = fft_magnitude**2
    total_energy = np.sum(fft_mag_sq)
    high_freq_energy = np.sum(fft_mag_sq[high_freq_mask])
    
    # Calculate ratio (avoid division by zero and extreme values)
    if total_energy > 1e-10 and np.isfinite(total_energy) and np.isfinite(high_freq_energy):
        ratio = high_freq_energy / total_energy
        # Clip to reasonable range to prevent extreme values
        ratio = np.clip(ratio, 0.0, 1.0)
    else:
        ratio = 0.0
    
    # Ensure finite value
    if not np.isfinite(ratio):
        ratio = 0.0
    
    return np.full_like(block_2d, ratio, dtype=np.float32)


def _haar_wavelet_energy_np(block_2d: np.ndarray) -> np.ndarray:
    """
    NumPy kernel for Haar wavelet detail-band energy.
    Calculates energy in LH/HL/HH bands normalized by total energy.
    Implements manual 2D Haar wavelet transform.
    """
    try:
        # Normalize input to prevent overflow
        block_normalized = block_2d.astype(np.float64)  # Use float64 for precision
        block_mean = np.mean(block_normalized)
        block_std = np.std(block_normalized)
        
        if block_std > 0:
            block_normalized = (block_normalized - block_mean) / block_std
        else:
            block_normalized = block_normalized - block_mean
        
        # Ensure even dimensions for Haar wavelet
        h, w = block_normalized.shape
        data = block_normalized.copy()
        
        # Pad to even dimensions if needed
        if h % 2 == 1:
            data = np.vstack([data, data[-1:, :]])
        if w % 2 == 1:
            data = np.hstack([data, data[:, -1:]])
        
        # Manual 2D Haar wavelet transform
        h_new, w_new = data.shape
        
        # Step 1: Horizontal decomposition (rows)
        # Low-pass: (even + odd) / 2
        cA_h = (data[::2, :] + data[1::2, :]) / 2  # Approximation rows
        # High-pass: (even - odd) / 2  
        cH_h = (data[::2, :] - data[1::2, :]) / 2  # Detail rows
        
        # Step 2: Vertical decomposition (columns) on both subbands
        # On approximation subband
        cA = (cA_h[:, ::2] + cA_h[:, 1::2]) / 2  # LL (approximation)
        cH = (cA_h[:, ::2] - cA_h[:, 1::2]) / 2  # LH (horizontal detail)
        
        # On detail subband  
        cV = (cH_h[:, ::2] + cH_h[:, 1::2]) / 2  # HL (vertical detail)
        cD = (cH_h[:, ::2] - cH_h[:, 1::2]) / 2  # HH (diagonal detail)
        
        # Calculate energies with overflow protection
        cA_sq = cA**2
        cH_sq = cH**2
        cV_sq = cV**2
        cD_sq = cD**2
        
        total_energy = np.sum(cA_sq) + np.sum(cH_sq) + np.sum(cV_sq) + np.sum(cD_sq)
        detail_energy = np.sum(cH_sq) + np.sum(cV_sq) + np.sum(cD_sq)  # LH, HL, HH bands
        
        # Calculate normalized detail energy ratio
        if (total_energy > 1e-10 and 
            np.isfinite(total_energy) and 
            np.isfinite(detail_energy)):
            ratio = detail_energy / total_energy
            # Clip to reasonable range to prevent extreme values
            ratio = np.clip(ratio, 0.0, 1.0)
        else:
            ratio = 0.0
            
    except Exception as e:
        # Fallback if wavelet transform fails
        ratio = 0.0
    
    # Ensure finite value
    if not np.isfinite(ratio):
        ratio = 0.0
    
    return np.full_like(block_2d, ratio, dtype=np.float32)


def _detect_tissue_rgb(img_da, tile_indices: np.ndarray, tiles_y: int, tiles_x: int, ty: int, tx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Internal function to detect tissue vs background using RGB values with Dask optimization.
    
    Parameters:
    -----------
    img_da : dask.array
        Image data in (y, x, c) format
    tile_indices : np.ndarray
        Array of shape (n_tiles, 2) with [y_idx, x_idx] for each tile
    tiles_y, tiles_x : int
        Grid dimensions
    ty, tx : int
        Tile size dimensions
    
    Returns:
    --------
    tissue_mask : np.ndarray
        Boolean array where True indicates tissue tiles
    background_mask : np.ndarray
        Boolean array where True indicates background tiles
    tissue_similarities : np.ndarray
        Similarity scores to tissue reference
    background_similarities : np.ndarray
        Similarity scores to background reference
    """
    from spatialdata._logging import logger as logg
    from dask.diagnostics import ProgressBar

    H, W, C = img_da.shape
    n_tiles = len(tile_indices)

    if n_tiles > 0:
        # Collect all tile mean operations as Dask arrays
        tile_means = []
        for y_idx, x_idx in tile_indices:
            # Calculate tile boundaries
            y0 = y_idx * ty
            y1 = min((y_idx + 1) * ty, H)
            x0 = x_idx * tx
            x1 = min((x_idx + 1) * tx, W)

            # Extract tile as Dask array and compute mean
            tile = img_da[y0:y1, x0:x1]
            tile_mean = tile.mean(axis=(0, 1))
            tile_means.append(tile_mean)

        stacked_means = da.stack(tile_means, axis=0)  # Shape: (n_tiles, C)

        with ProgressBar():
            rgb_data = stacked_means.compute()  # Single progress bar for all tiles

    else:
        logg.warning("No tiles found for processing")
        rgb_data = np.zeros((0, C), dtype=np.float32)

    # Identify reference tiles
    # Corner tiles (background reference)
    corner_positions = [
        (0, 0),                    # Top-left
        (0, tiles_x - 1),          # Top-right  
        (tiles_y - 1, 0),          # Bottom-left
        (tiles_y - 1, tiles_x - 1) # Bottom-right
    ]

    # Center tiles (tissue reference)
    center_y_start = int(tiles_y * 0.3)
    center_y_end = int(tiles_y * 0.7)
    center_x_start = int(tiles_x * 0.3)
    center_x_end = int(tiles_x * 0.7)

    # Get corner tile indices
    corner_indices = []
    for y_pos, x_pos in corner_positions:
        corner_idx = np.where((tile_indices[:, 0] == y_pos) & (tile_indices[:, 1] == x_pos))[0]
        if len(corner_idx) > 0:
            corner_indices.append(corner_idx[0])

    # Get center tile indices
    center_indices = []
    for y_pos, x_pos in itertools.product(range(center_y_start, center_y_end), range(center_x_start, center_x_end)):
        center_idx = np.where((tile_indices[:, 0] == y_pos) & (tile_indices[:, 1] == x_pos))[0]
        if len(center_idx) > 0:
            center_indices.append(center_idx[0])

    if len(corner_indices) < 2 or len(center_indices) < 2:
        logg.warning("Not enough reference tiles found, classifying all as tissue")
        tissue_mask = np.ones(n_tiles, dtype=bool)
        background_mask = ~tissue_mask
        tissue_similarities = np.ones(n_tiles)
        background_similarities = np.zeros(n_tiles)
    else:
        # Get reference RGB profiles
        corner_rgb = rgb_data[corner_indices]  # Shape: (n_corners, n_channels)
        center_rgb = rgb_data[center_indices]  # Shape: (n_center, n_channels)

        # Calculate mean reference profiles
        background_reference = np.mean(corner_rgb, axis=0)
        tissue_reference = np.mean(center_rgb, axis=0)

        # Calculate similarity to both references for all tiles using cosine similarity
        with ProgressBar():
            rgb_norm = rgb_data / (np.linalg.norm(rgb_data, axis=1, keepdims=True) + 1e-8)
            bg_norm = background_reference / (np.linalg.norm(background_reference) + 1e-8)
            tissue_norm = tissue_reference / (np.linalg.norm(tissue_reference) + 1e-8)

            # Calculate similarities
            background_similarities = np.dot(rgb_norm, bg_norm)
            tissue_similarities = np.dot(rgb_norm, tissue_norm)

        # Higher similarity to corners = background, higher similarity to center = tissue
        background_mask = background_similarities > tissue_similarities
        tissue_mask = ~background_mask

        n_background = np.sum(background_mask)
        n_tissue = np.sum(tissue_mask)

    return tissue_mask, background_mask, tissue_similarities, background_similarities


def detect_tissue(sdata, image_key: str, scale: str = "scale0", tile_size: Union[Literal["auto"], Tuple[int, int]] = "auto") -> AnnData:
    """
    Detect tissue vs background tiles using RGB values from corner (background) vs center (tissue) references.
    
    Parameters:
    -----------
    sdata : SpatialData
        Your SpatialData object.
    image_key : str
        Key in sdata.images.
    scale : str
        Multiscale level, e.g. "scale0".
    tile_size : "auto" or (int, int)
        Tile size dimensions. When "auto", creates roughly 100x100 grid overlay
        with minimum 100x100 pixel tiles.
    
    Returns:
    --------
    AnnData
        AnnData object with tissue classification results in obs columns:
        - "is_tissue": Boolean indicating tissue tiles
        - "is_background": Boolean indicating background tiles
        - "tissue_similarity": Similarity to tissue reference (0-1)
        - "background_similarity": Similarity to background reference (0-1)
    """
    from spatialdata._logging import logger as logg
    
    # Get image data using helper function
    img_da = _get_image_data(sdata, image_key, scale)
    H, W, C = img_da.shape
    
    logg.info(f"Preparing sharpness metrics calculation.")
    logg.info(f"- Image size (x, y) is ({W}, {H}), channels: {C}.")
    
    # Calculate tile size
    if tile_size == "auto":
        ty, tx = _calculate_auto_tile_size(H, W)
    else:
        ty, tx = tile_size
        if len(tile_size) == 3:
            raise ValueError("tile_size must be 2D (y, x), not 3D")
    
    logg.info(f"Using tiles with size (x, y): ({tx}, {ty})")
    
    # Calculate number of tiles
    tiles_y = (H + ty - 1) // ty
    tiles_x = (W + tx - 1) // tx
    n_tiles = tiles_y * tiles_x
    
    # Create tile indices and boundaries using helper function
    tile_indices, obs_names, pixel_bounds = _create_tile_indices_and_bounds(
        tiles_y, tiles_x, ty, tx, H, W
    )
    
    # Use shared tissue detection function
    tissue_mask, background_mask, tissue_similarities, background_similarities = _detect_tissue_rgb(
        img_da, tile_indices, tiles_y, tiles_x, ty, tx
    )
    
    # Create AnnData object
    adata = AnnData(X=np.zeros((n_tiles, 1)))  # Dummy X matrix
    adata.var_names = ["dummy"]
    adata.obs_names = obs_names
    
    # Add tissue classification results
    adata.obs["is_tissue"] = pd.Categorical(tissue_mask.astype(str), categories=["False", "True"])
    adata.obs["is_background"] = pd.Categorical(background_mask.astype(str), categories=["False", "True"])
    adata.obs["tissue_similarity"] = tissue_similarities
    adata.obs["background_similarity"] = background_similarities
    
    # Add tile grid indices
    adata.obs["tile_y"] = tile_indices[:, 0]
    adata.obs["tile_x"] = tile_indices[:, 1]
    
    # Add tile boundaries in pixels (already calculated by helper function)
    adata.obs["pixel_y0"] = pixel_bounds[:, 0]
    adata.obs["pixel_x0"] = pixel_bounds[:, 1]
    adata.obs["pixel_y1"] = pixel_bounds[:, 2]
    adata.obs["pixel_x1"] = pixel_bounds[:, 3]
    
    # Add metadata
    adata.uns["tissue_detection"] = {
        "image_key": image_key,
        "scale": scale,
        "tile_size_y": ty,
        "tile_size_x": tx,
        "image_height": H,
        "image_width": W,
        "n_tiles_y": tiles_y,
        "n_tiles_x": tiles_x,
        "n_tissue_tiles": int(np.sum(tissue_mask)),
        "n_background_tiles": int(np.sum(background_mask)),
        "method": "rgb_similarity"
    }
    
    return adata


def _detect_sharpness_outliers(X: np.ndarray, method: str = "iqr", tissue_mask: Optional[np.ndarray] = None, var_names: Optional[list] = None) -> np.ndarray:
    """
    Detect tiles with low sharpness (blurry/out-of-focus) using parameter-free methods.
    
    Parameters:
    -----------
    X : np.ndarray
        Sharpness metrics array of shape (n_tiles, n_metrics)
    method : str
        Method to use: "iqr", "zscore", "tenengrad_tissue", or "pvalue"
    tissue_mask : np.ndarray, optional
        Boolean mask indicating tissue tiles. Required for "tenengrad_tissue" method.
    
    Returns:
    --------
    outlier_labels : np.ndarray
        Array of -1 (low sharpness outlier) or 1 (normal sharpness) labels
    """
    # Clean the data to prevent infinite values
    X_clean = _clean_sharpness_data(X)
    
    if method == "tenengrad_tissue":
        return _detect_tenengrad_tissue_outliers(X_clean, tissue_mask, var_names)
    elif method == "pvalue":
        return _detect_outliers_pvalue(X_clean, tissue_mask, var_names)
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    if method == "iqr":
        return _detect_outliers_iqr(X_scaled)
    elif method == "zscore":
        return _detect_outliers_zscore(X_scaled)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr', 'zscore', 'tenengrad_tissue', or 'pvalue'.")


def _clean_sharpness_data(X: np.ndarray) -> np.ndarray:
    """
    Clean sharpness data by handling infinite and extreme values.
    
    Parameters:
    -----------
    X : np.ndarray
        Sharpness metrics array of shape (n_tiles, n_metrics)
    
    Returns:
    --------
    np.ndarray
        Cleaned array with finite values
    """
    X_clean = X.copy()
    
    # Replace infinite values with NaN
    X_clean[np.isinf(X_clean)] = np.nan
    
    # For each metric, replace NaN values with the median of that metric
    for i in range(X_clean.shape[1]):
        metric_data = X_clean[:, i]
        if np.any(np.isnan(metric_data)):
            median_val = np.nanmedian(metric_data)
            X_clean[np.isnan(metric_data), i] = median_val
    
    # Clip extreme values to prevent overflow
    # Use 99.9th percentile as upper bound
    for i in range(X_clean.shape[1]):
        metric_data = X_clean[:, i]
        upper_bound = np.percentile(metric_data, 99.9)
        lower_bound = np.percentile(metric_data, 0.1)
        X_clean[:, i] = np.clip(metric_data, lower_bound, upper_bound)
    
    return X_clean


def _detect_outliers_iqr(X_scaled: np.ndarray) -> np.ndarray:
    """Detect outliers using Interquartile Range (IQR) method - only low sharpness."""
    # Calculate IQR for each metric
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    
    # Define outlier bounds (1.5 * IQR rule) - only lower bound for low sharpness
    lower_bound = Q1 - 1.5 * IQR
    
    # Flag outliers only if sharpness is LOW (below lower bound)
    # High sharpness is good, so we don't flag upper outliers
    outlier_mask = np.any(X_scaled < lower_bound, axis=1)
    
    # Convert to -1/1 format
    outlier_labels = np.where(outlier_mask, -1, 1)
    
    return outlier_labels


def _detect_outliers_zscore(X_scaled: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using Z-score method - only low sharpness."""
    # Calculate Z-scores (X_scaled is already standardized)
    # Only check for LOW sharpness (negative z-scores)
    z_scores = X_scaled  # X_scaled is already standardized
    
    # Flag tiles that have LOW sharpness (negative z-scores below threshold)
    # High sharpness (positive z-scores) is good, so we don't flag those
    outlier_mask = np.any(z_scores < -threshold, axis=1)
    
    # Convert to -1/1 format
    outlier_labels = np.where(outlier_mask, -1, 1)
    
    return outlier_labels


def _detect_tenengrad_tissue_outliers(
    X: np.ndarray, 
    tissue_mask: Optional[np.ndarray] = None,
    var_names: Optional[list] = None
) -> np.ndarray:
    """
    Calculate unfocus likelihood scores for tissue tiles using Tenengrad scores with background reference.
    
    Logic:
    1. Use background tiles as reference for "blurry" appearance
    2. Compare tissue tiles to both background and other tissue tiles
    3. Calculate continuous scores (0-1) where higher values indicate more likely to be unfocused
    4. Scores are normalized: 0 = very sharp, 1 = very blurry (background-like)
    
    Parameters
    ----------
    X : np.ndarray
        Sharpness scores array of shape (n_tiles, n_metrics)
    tissue_mask : np.ndarray, optional
        Boolean mask indicating tissue tiles
    var_names : list, optional
        Variable names for the metrics
        
    Returns
    -------
    np.ndarray
        Array of unfocus likelihood scores (0-1) for all tiles
    """
    from sklearn.mixture import GaussianMixture
    
    if tissue_mask is None:
        # If no tissue mask, fall back to standard zscore method
        return _detect_outliers_zscore(X)
    
    # Get background and tissue tiles
    background_mask = ~tissue_mask
    tissue_tiles = X[tissue_mask]
    background_tiles = X[background_mask]
    
    if len(tissue_tiles) == 0 or len(background_tiles) == 0:
        return np.zeros(len(X))  # No unfocus if no tissue or background
    
    # Find Tenengrad metric index
    tenengrad_idx = None
    for i, var_name in enumerate(var_names):
        if "tenengrad" in var_name.lower():
            tenengrad_idx = i
            break
    
    if tenengrad_idx is None:
        # Fallback to standard zscore method if Tenengrad not found
        return _detect_outliers_zscore(X)
    
    # Extract Tenengrad scores using the correct index
    tissue_tenengrad = tissue_tiles[:, tenengrad_idx]
    background_tenengrad = background_tiles[:, tenengrad_idx]
    
    # Step 1: Analyze background distribution
    background_mean = np.mean(background_tenengrad)
    background_std = np.std(background_tenengrad)
    
    # Step 2: Analyze tissue distribution
    tissue_mean = np.mean(tissue_tenengrad)
    tissue_std = np.std(tissue_tenengrad)
    
    # Step 3: Determine if tissue has one or two classes
    # Use Gaussian Mixture Model to test for bimodality
    if len(tissue_tenengrad) >= 4:  # Need at least 4 samples for GMM
        # Try 1-component and 2-component models
        gmm_1 = GaussianMixture(n_components=1, random_state=42)
        gmm_2 = GaussianMixture(n_components=2, random_state=42)
        
        try:
            gmm_1.fit(tissue_tenengrad.reshape(-1, 1))
            gmm_2.fit(tissue_tenengrad.reshape(-1, 1))
            
            # Use BIC to determine best model
            bic_1 = gmm_1.bic(tissue_tenengrad.reshape(-1, 1))
            bic_2 = gmm_2.bic(tissue_tenengrad.reshape(-1, 1))
            
            # If 2-component model is significantly better, use it
            bic_improvement = bic_1 - bic_2
            use_two_components = bic_improvement > 10  # Threshold for significant improvement
            
        except:
            # If GMM fails, use simple threshold method
            use_two_components = False
    else:
        use_two_components = False
    
    # Step 4: Calculate continuous unfocus likelihood scores
    if use_two_components:
        # Two-class case: use GMM probabilities
        gmm_2.fit(tissue_tenengrad.reshape(-1, 1))
        tissue_probs = gmm_2.predict_proba(tissue_tenengrad.reshape(-1, 1))
        
        # Determine which component is "blurry" (closer to background)
        component_means = gmm_2.means_.flatten()
        blurry_component = np.argmin(np.abs(component_means - background_mean))
        
        # Use probability of belonging to blurry component as unfocus score
        unfocus_scores = tissue_probs[:, blurry_component]
        
    else:
        # One-class case: calculate distance-based scores
        # Normalize tissue scores relative to background distribution
        # Score = 1 - (tissue_score - background_min) / (background_max - background_min)
        background_min = np.min(background_tenengrad)
        background_max = np.max(background_tenengrad)
        background_range = background_max - background_min
        
        if background_range > 0:
            # Normalize tissue scores to [0, 1] where 0 = sharp, 1 = background-like
            normalized_scores = (tissue_tenengrad - background_min) / background_range
            # Invert so higher Tenengrad = lower unfocus score
            unfocus_scores = 1.0 - np.clip(normalized_scores, 0, 1)
        else:
            # If background has no variation, use simple threshold
            tissue_background_ratio = tissue_mean / (background_mean + 1e-10)
            if tissue_background_ratio > 1.5:
                # Tissue is sharp - score based on distance from tissue mean
                unfocus_scores = np.clip((tissue_mean - tissue_tenengrad) / (tissue_std + 1e-10), 0, 1)
            else:
                # Tissue is blurry - score based on similarity to background
                unfocus_scores = np.clip((tissue_tenengrad - background_mean) / (background_std + 1e-10), 0, 1)
    
    # Step 5: Create full unfocus scores array
    full_scores = np.zeros(len(X))  # Start with all sharp (0)
    tissue_indices = np.where(tissue_mask)[0]
    full_scores[tissue_indices] = unfocus_scores  # Set tissue unfocus scores
    
    return full_scores


def _detect_outliers_pvalue(
    X: np.ndarray, 
    tissue_mask: Optional[np.ndarray] = None,
    var_names: Optional[list] = None,
    alpha: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers using p-value method based on background distribution.
    
    For each tile, calculates the p-value of observing its sharpness score
    given the distribution of background tiles. Tiles with p-values below alpha
    are considered outliers (unlikely to come from background distribution).
    
    Parameters:
    -----------
    X : np.ndarray
        Sharpness metrics array of shape (n_tiles, n_metrics)
    tissue_mask : np.ndarray, optional
        Boolean mask indicating tissue tiles. If None, uses all tiles.
    var_names : list, optional
        List of metric names. If None, uses all metrics.
    alpha : float
        Significance level for outlier detection (default: 0.05)
    
    Returns:
    --------
    outlier_labels : np.ndarray
        Array of -1 (outlier) or 1 (normal) labels
    p_values : np.ndarray
        Array of p-values for each tile
    """
    from scipy import stats
    
    if tissue_mask is None:
        # If no tissue mask, use all tiles
        tissue_mask = np.ones(len(X), dtype=bool)
    
    if var_names is None:
        var_names = [f"metric_{i}" for i in range(X.shape[1])]
    
    # Separate tissue and background tiles
    tissue_tiles = X[tissue_mask]
    background_tiles = X[~tissue_mask]
    
    if len(background_tiles) < 10:
        # Not enough background tiles for reliable statistics
        return np.ones(len(X), dtype=int), np.ones(len(X))
    
    # Calculate p-values for each metric
    p_values = np.ones((len(tissue_tiles), len(var_names)))
    
    for i, var_name in enumerate(var_names):
        tissue_scores = tissue_tiles[:, i]
        background_scores = background_tiles[:, i]
        
        # Fit normal distribution to background scores
        bg_mean = np.mean(background_scores)
        bg_std = np.std(background_scores)
        
        if bg_std < 1e-10:
            # No variation in background, skip this metric
            continue
        
        # Calculate p-values for tissue scores
        # For sharpness metrics, we typically want to detect LOW values (blurry tiles)
        # So we calculate the probability of observing a value as low or lower
        p_values[:, i] = stats.norm.cdf(tissue_scores, loc=bg_mean, scale=bg_std)
    
    # Combine p-values across metrics using Fisher's method or minimum p-value
    # Using minimum p-value approach (more conservative)
    min_p_values = np.min(p_values, axis=1)
    
    # Create full p-values array
    full_p_values = np.ones(len(X))
    tissue_indices = np.where(tissue_mask)[0]
    full_p_values[tissue_indices] = min_p_values
    
    # Flag outliers: p-value < alpha (unlikely to come from background)
    outlier_mask = full_p_values < alpha
    
    # Convert to -1/1 format
    outlier_labels = np.where(outlier_mask, -1, 1)
    
    return outlier_labels, full_p_values


def _calculate_auto_tile_size(height: int, width: int) -> Tuple[int, int]:
    """
    Calculate tile size for auto mode: roughly 100x100 grid overlay with 100px minimum.
    Creates square tiles that evenly divide the image dimensions.
    
    Parameters
    ----------
    height : int
        Image height in pixels
    width : int
        Image width in pixels
        
    Returns
    -------
    Tuple[int, int]
        Tile size as (y, x) in pixels (always square)
    """
    # Calculate how many tiles we want (roughly 100 in the larger dimension)
    target_tiles = 100

    # Calculate tile size for each dimension
    tile_size_y = height // target_tiles
    tile_size_x = width // target_tiles

    # Use the smaller tile size to ensure both dimensions are covered
    tile_size = min(tile_size_y, tile_size_x)

    # Ensure minimum 100x100 pixel tiles
    return (100, 100) if tile_size < 100 else (tile_size, tile_size)


def _make_tiles(
    arr_yx: da.Array,
    tile_size: Union[Literal["auto"], Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Decide tile size based on tile_size parameter.
    
    Parameters
    ----------
    arr_yx : da.Array
        Dask array with (y, x) dimensions
    tile_size : "auto" or (int, int)
        Tile size dimensions. When "auto", creates roughly 100x100 grid overlay.
        
    Returns
    -------
    Tuple[int, int]
        Tile size as (y, x) in pixels
    """
    
    if tile_size == "auto":
        return _calculate_auto_tile_size(arr_yx.shape[0], arr_yx.shape[1])
    
    if isinstance(tile_size, tuple):
        if len(tile_size) == 2:
            return int(tile_size[0]), int(tile_size[1])
        else:
            raise ValueError(f"tile_size tuple must have exactly 2 dimensions (y, x), got {len(tile_size)}")
    
    raise ValueError(f"tile_size must be 'auto' or a 2-tuple, got {type(tile_size)}")


def _create_tile_indices_and_bounds(
    tiles_y: int, 
    tiles_x: int, 
    tile_size_y: int, 
    tile_size_x: int, 
    image_height: int, 
    image_width: int
) -> Tuple[np.ndarray, list, np.ndarray]:
    """
    Create tile indices, observation names, and pixel boundaries.
    
    Parameters
    ----------
    tiles_y, tiles_x : int
        Grid dimensions
    tile_size_y, tile_size_x : int
        Tile size dimensions
    image_height, image_width : int
        Image dimensions
        
    Returns
    -------
    Tuple[np.ndarray, list, np.ndarray]
        Tuple containing:
        - tile_indices: Array of shape (n_tiles, 2) with [y_idx, x_idx]
        - obs_names: List of observation names
        - pixel_bounds: Array of shape (n_tiles, 4) with [y0, x0, y1, x1]
    """
    tile_indices = []
    obs_names = []
    pixel_bounds = []
    
    for y_idx in range(tiles_y):
        for x_idx in range(tiles_x):
            tile_indices.append([y_idx, x_idx])
            obs_names.append(f"tile_x{x_idx}_y{y_idx}")
            
            # Calculate tile boundaries ensuring complete coverage
            y0 = y_idx * tile_size_y
            x0 = x_idx * tile_size_x
            y1 = (y_idx + 1) * tile_size_y if y_idx < tiles_y - 1 else image_height
            x1 = (x_idx + 1) * tile_size_x if x_idx < tiles_x - 1 else image_width
            pixel_bounds.append([y0, x0, y1, x1])
    
    return np.array(tile_indices), obs_names, np.array(pixel_bounds)


def _create_tile_polygons(
    scores: np.ndarray, 
    tile_size_y: int, 
    tile_size_x: int, 
    image_height: int, 
    image_width: int
) -> Tuple[np.ndarray, list]:
    """
    Create rectangular polygons for each tile in the grid.
    
    Parameters
    ----------
    scores : np.ndarray
        2D array of tile scores with shape (tiles_y, tiles_x)
    tile_size_y : int
        Height of each tile in pixels
    tile_size_x : int
        Width of each tile in pixels
    image_height : int
        Total image height in pixels
    image_width : int
        Total image width in pixels
        
    Returns
    -------
    Tuple[np.ndarray, list]
        Tuple containing:
        - 2D array of tile centroids (y, x coordinates)
        - List of Polygon objects for each tile
    """
    tiles_y, tiles_x = scores.shape
    centroids = []
    polygons = []
    
    for y_idx in range(tiles_y):
        for x_idx in range(tiles_x):
            # Calculate tile boundaries in pixel coordinates
            # Ensure complete coverage by extending last tiles to image boundaries
            y0 = y_idx * tile_size_y
            x0 = x_idx * tile_size_x
            y1 = (y_idx + 1) * tile_size_y if y_idx < tiles_y - 1 else image_height
            x1 = (x_idx + 1) * tile_size_x if x_idx < tiles_x - 1 else image_width
            
            # Calculate centroid
            centroid_y = (y0 + y1) / 2
            centroid_x = (x0 + x1) / 2
            centroids.append([centroid_y, centroid_x])
            
            # Create rectangular polygon for this tile
            # Note: Polygon expects (x, y) coordinates, not (y, x)
            polygon = Polygon([
                (x0, y0),  # bottom-left
                (x1, y0),  # bottom-right
                (x1, y1),  # top-right
                (x0, y1),  # top-left
                (x0, y0)   # close polygon
            ])
            polygons.append(polygon)
    
    return np.array(centroids), polygons

class SHARPNESS_METRICS(Enum):
    TENENGRAD = enum.auto()
    VAR_OF_LAPLACIAN = enum.auto()
    VARIANCE = enum.auto()
    FFT_HIGH_FREQ_ENERGY = enum.auto()
    HAAR_WAVELET_ENERGY = enum.auto()

def qc_sharpness(
    sdata,
    image_key: str,
    scale: str = "scale0",
    metrics: SHARPNESS_METRICS | list[SHARPNESS_METRICS] = [SHARPNESS_METRICS.TENENGRAD, SHARPNESS_METRICS.VAR_OF_LAPLACIAN],
    tile_size: Union[Literal["auto"], Tuple[int, int]] = "auto",
    detect_outliers: bool = True,
    detect_tissue: bool = True,
    outlier_method: str = "pvalue",
    outlier_cutoff: float = 0.1,
    progress: bool = True,
) -> None:
    """
    Compute tilewise sharpness scores for multiple metrics and print the values.

    Parameters
    ----------
    sdata : SpatialData
        Your SpatialData object.
    image_key : str
        Key in sdata.images.
    scale : str
        Multiscale level, e.g. "scale0".
    metrics : SHARPNESS_METRICS or "all"
        Sharpness metrics to compute. If "all", computes all available metrics.
    tile_size : "auto" or (int, int)
        Tile size dimensions. When "auto", creates roughly 100x100 grid overlay
        with minimum 100x100 pixel tiles.
    detect_outliers : bool
        If True, identify tiles with abnormal sharpness scores. If `detect_tissue=True`,
        outlier detection will only run on tissue tiles. Adds `sharpness_outlier` column
        to the AnnData table.
    detect_tissue : bool
        Only evaluated if `detect_outliers=True`. If True, classify tiles as background 
        vs tissue using RGB values from corner (background) vs center (tissue) 
        references. Adds `is_tissue`, `is_background`, `tissue_similarity`, and 
        `background_similarity` columns to the AnnData table.
    outlier_method : str
        Method for detecting low sharpness tiles: "iqr", "zscore", "tenengrad_tissue", or "pvalue".
        - "iqr": Interquartile Range method (parameter-free)
        - "zscore": Z-score method (parameter-free) 
        - "tenengrad_tissue": Tenengrad-based method using background reference (requires detect_tissue=True)
        - "pvalue": P-value method based on background distribution (requires detect_tissue=True, falls back to zscore if not available)
        Default "pvalue" uses P-value-based method. Only flags tiles with LOW sharpness.
    outlier_cutoff : float
        Threshold for binarizing continuous unfocus scores into outlier labels. 
        Tiles with unfocus_score >= outlier_cutoff are marked as outliers.
    progress : bool
        Show a Dask progress bar for the compute step.

    Returns
    -------
    None
        Prints results to stdout and adds TableModel and ShapesModel to sdata.
        Table key: "qc_img_{image_key}_sharpness"
        Shapes key: "qc_img_{image_key}_sharpness_grid"
        When `detect_outliers=True`, adds "sharpness_outlier" column to AnnData.obs.
        When `detect_tissue=True` and `detect_outliers=True`, also adds "is_tissue", 
        "is_background", "tissue_similarity", and "background_similarity" columns.
        "sharpness_outlier" flags tiles with low sharpness (blurry/out-of-focus).
    """
    from spatialdata import SpatialData

    # 1) Get image data using helper function
    img_da = _get_image_data(sdata, image_key, scale)

    # 2) Ensure dims and grayscale
    img_yxc = _ensure_yxc(img_da)
    gray = _to_gray_dask_yx(img_yxc)  # (y, x), float32 dask array
    H, W = gray.shape

    # 3) Determine which metrics to compute

    # 4) Calculate tile size
    ty, tx = _make_tiles(gray, tile_size)
    logg.info(f"Preparing sharpness metrics calculation.")
    logg.info(f"- Image size (x, y) is ({W}, {H}), using tiles with size (x, y): ({tx}, {ty}).")

    # Calculate tile grid dimensions
    tiles_y = (H + ty - 1) // ty
    tiles_x = (W + tx - 1) // tx
    n_tiles = tiles_y * tiles_x
    logg.info(f"- Resulting tile grid has shape (x, y): ({tiles_x}, {tiles_y}).")

    # 5) Compute sharpness scores for all metrics
    all_scores = {}

    print("")
    logg.info(f"Calculating sharpness metrics.")
    metrics_to_compute = metrics if isinstance(metrics, list) else [metrics]
    for metric in metrics_to_compute:
        metric_name = metric.name.lower() if isinstance(metric, SHARPNESS_METRICS) else metric
        logg.info(f"- Computing sharpness metric '{metric_name}'.")

        # Force Dask to use smaller chunks that match tile size
        # This prevents large chunks from creating uniform blocks
        gray_rechunked = gray.rechunk((ty, tx))

        # Per-pixel sharpness via map_overlap (no overlap for adjacent tiles)
        if metric_name == "tenengrad":
            sharp_field = da.map_overlap(
                _sobel_energy_numba, gray_rechunked, depth=0, boundary="reflect", dtype=np.float32
            )
        elif metric_name == "var_of_laplacian":
            sharp_field = da.map_overlap(
                _laplace_square_numba, gray_rechunked, depth=0, boundary="reflect", dtype=np.float32
            )
        elif metric_name == "variance":
            sharp_field = da.map_overlap(
                _variance_numba, gray_rechunked, depth=0, boundary="reflect", dtype=np.float32
            )
        elif metric_name == "fft_high_freq_energy":
            sharp_field = da.map_overlap(
                    _fft_high_freq_energy_np, gray_rechunked, depth=0, boundary="reflect", dtype=np.float32
            )
        elif metric_name == "haar_wavelet_energy":
            sharp_field = da.map_overlap(
                    _haar_wavelet_energy_np, gray_rechunked, depth=0, boundary="reflect", dtype=np.float32
            )
        else:
            raise ValueError(f"- Unknown metric '{metric_name}'.")

        # Pad the array to make it divisible by tile size
        pad_y = (tiles_y * ty) - sharp_field.shape[0]
        pad_x = (tiles_x * tx) - sharp_field.shape[1]

        if pad_y > 0 or pad_x > 0:
            # Pad with edge values
            padded = da.pad(sharp_field, ((0, pad_y), (0, pad_x)), mode='edge')
        else:
            padded = sharp_field

        # Now coarsen with trim_excess=False since dimensions are divisible
        # For additive metrics, we need to normalize by tile area to make them tile-size independent
        if metric_name in ["tenengrad"]:
            # Sum across tile, then normalize by tile area
            tile_scores = da.coarsen(np.sum, padded, {0: ty, 1: tx}, trim_excess=False)
            tile_area = ty * tx
            tile_scores = tile_scores / tile_area
        else:
            # For other metrics (variance, entropy, fft, haar), use mean (already normalized)
            tile_scores = da.coarsen(np.mean, padded, {0: ty, 1: tx}, trim_excess=False)

        # Compute scores with progress bar if requested
        if progress:
            with ProgressBar():
                scores = tile_scores.compute()  # 2D numpy array
        else:
            scores = tile_scores.compute()

        all_scores[metric_name] = scores

    # Get dimensions from first metric
    first_metric = list(all_scores.keys())[0]
    scores = all_scores[first_metric]
    tiles_y, tiles_x = scores.shape

    # Use original tile dimensions since padding ensures divisibility
    actual_ty = ty
    actual_tx = tx

    # Generate keys based on naming convention
    table_key = f"qc_img_{image_key}_sharpness"
    shapes_key = f"qc_img_{image_key}_sharpness_grid"

    # Create tile polygons and centroids using actual tile dimensions
    centroids, polygons = _create_tile_polygons(scores, actual_ty, actual_tx, H, W)

    # Create AnnData object with sharpness scores as variables (genes)
    n_tiles = len(centroids)
    n_metrics = len(all_scores)

    # Create X matrix with sharpness scores (tiles x metrics)
    X_data = np.zeros((n_tiles, n_metrics))
    var_names = []
    for i, (metric_name, metric_scores) in enumerate(all_scores.items()):
        X_data[:, i] = metric_scores.ravel()
        var_names.append(f"sharpness_{metric_name}")

    # Create tile indices and boundaries using helper function
    tile_indices, obs_names, pixel_bounds = _create_tile_indices_and_bounds(
        tiles_y, tiles_x, actual_ty, actual_tx, H, W
    )

    # Initialize default values
    outlier_labels = np.ones(len(X_data), dtype=int)  # All normal by default

    adata = AnnData(X=X_data)
    adata.var_names = var_names
    adata.obs_names = obs_names
    adata.obs["centroid_y"] = centroids[:, 0]
    adata.obs["centroid_x"] = centroids[:, 1]
    adata.obsm["spatial"] = centroids

    # Perform outlier detection if requested
    if detect_outliers:
        print("")
        logg.info(f"Detecting outlier tiles using method '{outlier_method}'...")

        if detect_tissue:
            logg.info("- Classifying tiles as tissue vs background.")
            tissue_mask = np.ones(len(X_data), dtype=bool)
            background_mask = np.zeros(len(X_data), dtype=bool)
            tissue_similarities = np.zeros(len(X_data), dtype=np.float32)
            background_similarities = np.zeros(len(X_data), dtype=np.float32)

            # Use the already loaded and processed image data
            img_da = img_yxc  # Already in (y, x, c) format from line 704

            # Use shared tissue detection function
            tissue_mask, background_mask, tissue_similarities, background_similarities = _detect_tissue_rgb(
                img_da, tile_indices, tiles_y, tiles_x, actual_ty, actual_tx
            )
            n_background = np.sum(background_mask)
            n_tissue = np.sum(tissue_mask)

            logg.info(f"- Classified {n_background} tiles as background and {n_tissue} tiles as tissue.")

        # Outlier detection
        if detect_tissue and np.sum(tissue_mask) > 0:
            logg.info("- Running outlier detection on tissue tiles only.")
            if outlier_method == "pvalue":
                # Use p-value method with tissue mask
                outlier_labels, pvalues = _detect_outliers_pvalue(X_data, tissue_mask=tissue_mask, var_names=var_names)
                # Convert p-values to unfocus scores (0 = focused, 1 = unfocused)
                # Lower p-values (more unlikely to be background) = higher unfocus scores
                unfocus_scores = 1.0 - pvalues
                min_score = np.min(unfocus_scores)
                max_score = np.max(unfocus_scores)
                if max_score > min_score:  # Avoid division by zero
                    unfocus_scores = (unfocus_scores - min_score) / (max_score - min_score)
                else:
                    unfocus_scores = np.zeros_like(unfocus_scores)
                # Use outlier_cutoff to binarize unfocus scores
                outlier_labels = np.where(unfocus_scores >= outlier_cutoff, -1, 1)
            elif outlier_method == "tenengrad_tissue":
                # Use the new Tenengrad-based method with tissue mask (returns continuous scores)
                unfocus_scores = _detect_tenengrad_tissue_outliers(X_data, tissue_mask=tissue_mask, var_names=var_names)
                # Use outlier_cutoff to binarize unfocus scores
                outlier_labels = np.where(unfocus_scores >= outlier_cutoff, -1, 1)
            else:
                # Use standard method on tissue tiles only
                tissue_data = X_data[tissue_mask]
                unfocus_scores = _detect_sharpness_outliers(tissue_data, method=outlier_method)

                # Map back to all tiles
                outlier_labels[tissue_mask] = unfocus_scores

            n_outliers = np.sum(outlier_labels == -1)
            logg.info(f"- Classified {n_outliers} tiles as outliers ({n_outliers/np.sum(tissue_mask)*100:.1f}%).")
        else:
            logg.info("- Running outlier detection on all tiles.")
            if outlier_method == "pvalue":
                # P-value method requires tissue detection, fall back to zscore
                logg.warning("P-value method requires tissue detection. Falling back to zscore method.")
                outlier_labels = _detect_sharpness_outliers(X_data, method="zscore", var_names=var_names)
                unfocus_scores = None
            else:
                outlier_labels = _detect_sharpness_outliers(X_data, method=outlier_method, var_names=var_names)

            n_outliers = np.sum(outlier_labels == -1)
            logg.info(f"- Classified {n_outliers} tiles as outliers ({n_outliers/len(outlier_labels)*100:.1f}%).")
            unfocus_scores = None  # No continuous scores for standard methods

        if detect_tissue:
            adata.obs["is_tissue"] = pd.Categorical(tissue_mask.astype(str), categories=["False", "True"])
            adata.obs["is_background"] = pd.Categorical(background_mask.astype(str), categories=["False", "True"])

        adata.obs["tissue_similarity"] = tissue_similarities
        adata.obs["background_similarity"] = background_similarities
        adata.obs["sharpness_outlier"] = pd.Categorical((outlier_labels == -1).astype(str), categories=["False", "True"])

        # Add continuous unfocus scores if available
        if unfocus_scores is not None:
            adata.obs["unfocus_score"] = unfocus_scores



    # Add metadata
    adata.uns["qc_sharpness"] = {
        "metrics": list(all_scores.keys()),
        "tile_size_y": actual_ty,
        "tile_size_x": actual_tx,
        "image_height": H,
        "image_width": W,
        "n_tiles_y": tiles_y,
        "n_tiles_x": tiles_x,
        "image_key": image_key,
        "scale": scale,
        "detect_tissue": detect_tissue,
        "outlier_method": outlier_method,
        "n_tissue_tiles": int(np.sum(tissue_mask)),
        "n_background_tiles": int(np.sum(background_mask)),
        "n_outlier_tiles": int(np.sum(outlier_labels == -1)),
    }

    # Add results to SpatialData
    table_model = TableModel.parse(adata)
    sdata.tables[table_key] = table_model

    logg.info(f"- Stored tiles as sdata.tables['{table_key}'].")

    # Create GeoDataFrame with tile polygons (ensuring complete coverage)
    tile_data = []
    for y_idx, x_idx in tile_indices:
        y0 = y_idx * actual_ty
        x0 = x_idx * actual_tx
        y1 = (y_idx + 1) * actual_ty if y_idx < tiles_y - 1 else H
        x1 = (x_idx + 1) * actual_tx if x_idx < tiles_x - 1 else W

        # Create rectangular polygon for this tile
        polygon = Polygon([
            (x0, y0),  # bottom-left
            (x1, y0),  # bottom-right
            (x1, y1),  # top-right
            (x0, y1),  # top-left
            (x0, y0)   # close polygon
        ])

        # Create tile data (without metrics - they're only in the table)
        tile_info = {
            'tile_id': f"tile_x{x_idx}_y{y_idx}",
            'tile_y': y_idx,
            'tile_x': x_idx,
            'pixel_y0': y0,
            'pixel_x0': x0,
            'pixel_y1': y1,
            'pixel_x1': x1,
            'geometry': polygon
        }

        tile_data.append(tile_info)

    tile_gdf = gpd.GeoDataFrame(tile_data, geometry='geometry')
    shapes_model = ShapesModel.parse(tile_gdf)
    sdata.shapes[shapes_key] = shapes_model

    sdata.tables[table_key].uns["spatialdata_attrs"] = {
        "region": shapes_key,
        "region_key": "grid_name", 
        "instance_key": "tile_id", 
    }
    sdata.tables[table_key].obs["grid_name"] = pd.Categorical([shapes_key] * len(sdata.tables[table_key]))
    sdata.tables[table_key].obs["tile_id"] = shapes_model.index

    logg.info(f"- Stored sharpness metrics as sdata.shapes['{shapes_key}'].")
