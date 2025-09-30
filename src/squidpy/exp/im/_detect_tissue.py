from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

import numpy as np
import spatialdata as sd
import xarray as xr
from dask_image.ndinterp import affine_transform as da_affine
from scipy import ndimage
from skimage import filters, measure
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, disk
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from spatialdata._logging import logger as logg
from spatialdata.models import Labels2DModel
from spatialdata.transformations import get_transformation

from squidpy._utils import _get_scale_factors, _yx_from_shape

from ._utils import _flatten_channels, _get_image_data


class DETECT_TISSUE_METHOD(enum.Enum):
    OTSU = enum.auto()
    FELZENSZWALB = enum.auto()


@dataclass(slots=True)
class BackgroundDetectionParams:
    """
    Which corners are background, and how large the corner boxes should be.
    If no corners are flagged True, the orientation falls back to 'bright background'.
    """

    ymin_xmin_is_bg: bool = True
    ymax_xmin_is_bg: bool = True
    ymin_xmax_is_bg: bool = True
    ymax_xmax_is_bg: bool = True
    corner_size_pct: float = 0.01  # size of each corner box as fraction of img height/width

    @property
    def any_corner(self) -> bool:
        return any((self.ymin_xmin_is_bg, self.ymax_xmin_is_bg, self.ymin_xmax_is_bg, self.ymax_xmax_is_bg))


@dataclass(slots=True)
class FelzenszwalbParams:
    """
    Size-aware superpixel defaults for felzenszwalb segmentation.
    """

    grid_rows: int = 100  # ~desired grid, only for scale heuristics
    grid_cols: int = 100
    sigma_frac: float = 0.008  # blur = this * short side (clipped to [1,5] px)
    scale_coef: float = 0.25  # scale = coef * target_area
    min_size_coef: float = 0.20  # min_size = coef * target_area


def detect_tissue(
    sdata: sd.SpatialData,
    image_key: str,
    scale: str = "auto",
    method: DETECT_TISSUE_METHOD | str = DETECT_TISSUE_METHOD.FELZENSZWALB,
    corners_are_background: bool = True,
    mask_smoothing_cycles: int = 0,
    new_labels_key: str | None = None,
    inplace: bool = True,
    **kwargs: Any,
) -> xr.DataArray:
    """
    Detect tissue and return a boolean mask (y,x) where True = specimen.

    Parameters
    ----------
    sdata : sd.SpatialData
        SpatialData object containing the image
    image_key : str
        Key of the image in sdata.images
    scale : str, default "auto"
        - If a specific scale key is provided (e.g., "scale0", "scale1", ...),
          that image scale is used verbatim.
        - If "auto": pick the smallest available scale. If that smallest scale
          exceeds the pixel threshold, it is further downscaled to be under the
          threshold for calculations.
    method : str or DETECT_TISSUE_METHOD, default FELZENSZWALB
        Method to use ("otsu" or "felzenszwalb")
    corners_are_background : bool, default True
        Whether all corners should be treated as background
    mask_smoothing_cycles : int, default 0
        Number of cycles of (2D) morphological closing to apply to the mask
    new_labels_key : str | None, default None
        Key to store the new labels in the SpatialData object
    inplace: bool, default True
        Whether to store the new labels in the SpatialData object or return the mask.
        If the mask is saved to the SpatialData object, it will inherit the scale_factors
        of the image, if present.
    **kwargs
        Optional keyword arguments:

        channel_format : {"infer", "rgb", "rgba", "multichannel"}, default "infer"
            How to interpret image channels for grey conversion:
            - "infer": Auto-detect (3 ch → RGB luminance, others → mean)
            - "rgb": Force RGB treatment (requires exactly 3 channels)
            - "rgba": Force RGBA treatment (requires exactly 4 channels)
            - "multichannel": Force mean across all channels

        background_detection_params : BackgroundDetectionParams, optional
            Custom background detection configuration. If provided, overrides
            the `corners_are_background` parameter. Default creates config
            based on `corners_are_background`.

        felzenszwalb_params : FelzenszwalbParams, optional
            Felzenszwalb segmentation parameters (only used when method="felzenszwalb").
            Default: FelzenszwalbParams(grid_rows=50, grid_cols=50, ...)

        min_specimen_area_frac : float, default 0.01
            Minimum area of a specimen as fraction of image area.

        n_samples : int | None, default None
            Number of specimens to keep. If provided, the n_samples largest components will be kept.
            If not provided, the specimens will be filtered by area using Otsu thresholding on log10(area).

        auto_max_pixels : int, default 2_000_000
            Target maximum number of pixels (H*W) for the image when scale="auto".


    Returns
    -------
    xr.DataArray
        Boolean mask with shape (y, x) where True indicates tissue

    Notes
    -----
    This function uses a simple pipeline:
    - OTSU: Global Otsu thresholding (with background orientation from corners)
    - FELZENSZWALB: Felzenszwalb superpixels -> per-superpixel Otsu (with background orientation)
    Both methods are followed by morphology to solidify, area-based filtering to keep only real specimens,
    and optional mask smoothing to refine boundaries.
    """

    # Set up background detection
    background_detection_params = kwargs.get("background_detection_params", None)
    if background_detection_params is None:
        background_detection_params = BackgroundDetectionParams(
            ymin_xmin_is_bg=corners_are_background,
            ymax_xmin_is_bg=corners_are_background,
            ymin_xmax_is_bg=corners_are_background,
            ymax_xmax_is_bg=corners_are_background,
        )

    # Convert string method to enum
    if isinstance(method, str):
        try:
            method = DETECT_TISSUE_METHOD[method.upper()]
        except KeyError as e:
            raise ValueError("method must be 'otsu' or 'felzenszwalb'") from e

    manual_scale = scale.lower() != "auto"

    if manual_scale:
        # Respect the user's scale verbatim
        img_src = _get_image_data(sdata, image_key, scale=scale)
    else:
        img_src = _get_image_data(sdata, image_key, scale="auto")

    img_src_h, img_src_w = _yx_from_shape(img_src.shape)
    n_source_pixels = img_src_h * img_src_w

    # 1) deal with channel dimension
    img_grey: xr.DataArray = _flatten_channels(img=img_src, channel_format=kwargs.get("channel_format", "infer"))

    # decide working resolution
    auto_max_pixels = kwargs.get("auto_max_pixels", 5_000_000)
    need_downscale = (not manual_scale) and (n_source_pixels > auto_max_pixels)

    if need_downscale:
        # Compute the array via Dask (if dask-backed) and show a progress bar
        logg.info("Downscaling for faster computation.")
        img_grey = _downscale_with_dask(img_grey=img_grey, target_pixels=auto_max_pixels)
    else:
        # No additional downscaling; use the smallest scale (or manual scale) as-is
        img_grey = img_grey.values  # may trigger compute without explicit progress bar

    # 2) first-pass foreground
    if method == DETECT_TISSUE_METHOD.OTSU:
        img_fg_mask = _segment_otsu(img_grey=img_grey, params=background_detection_params)
    elif method == DETECT_TISSUE_METHOD.FELZENSZWALB:
        labels = _segment_felzenszwalb(
            img_grey=img_grey,
            params=kwargs.get("felzenszwalb_params", FelzenszwalbParams()),
        )
        img_fg_mask = _mask_from_labels_via_corners(
            img_grey=img_grey, labels=labels, params=background_detection_params
        )
    else:
        raise ValueError(f"Method {method} not implemented")
    # 3) solidify
    img_fg_mask = _make_solid(img_fg_mask)

    # 4) keep only specimen-sized components (Otsu on areas)
    img_fg_mask = _filter_by_area(
        mask=img_fg_mask,
        min_specimen_area_frac=kwargs.get("min_specimen_area_frac", 0.01),
        n_samples=kwargs.get("n_samples", None),
    )

    # 5) smooth mask boundaries (optional)
    img_fg_mask = _smooth_mask(img_fg_mask, mask_smoothing_cycles)

    # 6) Upscale to full resolution of the source image
    target_shape = _get_target_upscale_shape(sdata, image_key)
    scale_matrix = _get_scaling_matrix(img_fg_mask.shape, target_shape)
    img_fg_mask_upscaled = da_affine(
        img_fg_mask,
        matrix=scale_matrix,
        offset=(0.0, 0.0),
        output_shape=target_shape,
        order=0,
        mode="constant",
        cval=0,
        output=np.int32,
    )

    if inplace:
        if new_labels_key is None:
            new_labels_key = f"{image_key}_tissue"

        source_scale_factors = _get_scale_factors(sdata.images[image_key])

        sdata.labels[new_labels_key] = Labels2DModel.parse(
            data=img_fg_mask_upscaled,
            dims=("y", "x"),
            transformations=get_transformation(sdata.images[image_key], get_all=True),
            scale_factors=source_scale_factors,
        )

        return None

    return np.array(img_fg_mask_upscaled)


def _get_scaling_matrix(current_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
    """
    Get the scaling matrix for upscaling the mask back to the original image size.
    """
    scale_y = 1 / (target_shape[0] / current_shape[0])
    scale_x = 1 / (target_shape[1] / current_shape[1])
    return np.array([[scale_y, 0.0], [0.0, scale_x]], dtype=float)


def _get_target_upscale_shape(
    sdata: sd.SpatialData,
    image_key: str,
) -> tuple[int, int]:
    """
    Get the target shape for upscaling the mask back to the original image size.
    """
    if not hasattr(sdata.images[image_key], "keys"):
        return _yx_from_shape(sdata.images[image_key].shape)

    target_scale = list(sdata.images[image_key].keys())[0]

    return _yx_from_shape(sdata.images[image_key][target_scale].image.shape)


def _downscale_with_dask(img_grey: xr.DataArray, target_pixels: int) -> np.ndarray:
    """
    Downscale (y,x) with Dask-backed xarray.coarsen (mean) until H*W <= target_pixels.
    Returns a NumPy array of the *downscaled* image and its shape. Shows a Dask ProgressBar.
    """
    img_grey_h, img_grey_w = img_grey.shape
    n_source_pixels = img_grey_h * img_grey_w
    if n_source_pixels <= target_pixels:
        # Nothing to do; still compute lazily with progress bar
        return _dask_compute(_ensure_dask(img_grey))

    # Desired continuous scale
    scale = float(np.sqrt(target_pixels / float(n_source_pixels)))  # 0 < s < 1
    target_h = max(1, int(img_grey_h * scale))
    target_w = max(1, int(img_grey_w * scale))

    # Integer coarsen factors (mean-pooling); ensure we don't exceed target
    coarsen_factor_y = max(1, int(np.ceil(img_grey_h / target_h)))
    coarsen_factor_x = max(1, int(np.ceil(img_grey_w / target_w)))

    # Ensure Dask backing (if not already)
    img_grey_small_da = (
        _ensure_dask(img_grey)
        .coarsen(y=coarsen_factor_y, x=coarsen_factor_x, boundary="trim")
        .mean()  # anti-aliased downscale
    )

    # Compute the *downscaled* array only
    img_grey_small = _dask_compute(img_grey_small_da)

    return np.asarray(img_grey_small)


def _ensure_dask(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure the DataArray is Dask-backed (chunked). If it's already Dask, return as-is.
    """
    try:
        import dask.array as dask_array

        if isinstance(da.data, dask_array.Array):
            return da
        # Chunk to reasonable tiles; adjust if you have known tile sizes
        return da.chunk({"y": 2048, "x": 2048})
    except ImportError:
        # Dask not available; just return original (compute will be eager)
        return da


def _dask_compute(img_grey_da: xr.DataArray) -> np.ndarray:
    """
    Compute an xarray DataArray (possibly Dask-backed) to a NumPy array with a Dask ProgressBar if available.
    """
    result: np.ndarray
    try:
        import dask.array as dask_array
        from dask.diagnostics import ProgressBar

        if isinstance(img_grey_da.data, dask_array.Array):
            with ProgressBar():
                result = img_grey_da.data.compute()
        result = img_grey_da.values
    except ImportError:
        result = img_grey_da.values
    return result


def _segment_otsu(img_grey: np.ndarray, params: BackgroundDetectionParams) -> np.ndarray:
    """
    Otsu binarization with orientation from background corners:
    - If corners (flagged) are brighter than global median -> foreground is darker (I <= t)
    - Else -> foreground is brighter (I >= t)
    If no corners are flagged, assume bright background (common for brightfield).
    """
    img_f = img_as_float(img_grey)
    t = threshold_otsu(img_f)
    bright_bg = _background_is_bright(img_f, params)
    return np.array((img_f <= t) if bright_bg else (img_f >= t))


def _segment_felzenszwalb(img_grey: np.ndarray, params: FelzenszwalbParams) -> np.ndarray:
    img_grey_h, img_grey_w = img_grey.shape

    # Parameters computed on the image resolution
    short = min(img_grey_h, img_grey_w)
    sigma = float(np.clip(params.sigma_frac * short, 1.0, 5.0))
    img_s = img_as_float(gaussian(img_grey, sigma=sigma))

    target_regions = max(1, params.grid_rows * params.grid_cols)
    target_area = (img_grey_h * img_grey_w) / target_regions
    scale = float(max(1.0, params.scale_coef * target_area))
    min_size = int(max(1, params.min_size_coef * target_area))

    return np.array(
        felzenszwalb(
            img_s,
            scale=scale,
            sigma=sigma,
            min_size=min_size,
            channel_axis=None,
        ).astype(np.int32)
    )


def _mask_from_labels_via_corners(
    img_grey: np.ndarray, labels: np.ndarray, params: BackgroundDetectionParams
) -> np.ndarray:
    """
    Turn superpixels into a mask via Otsu on per-label mean intensity, oriented by corners.
    """
    labels = labels.astype(np.int32)
    n_labels = int(labels.max())
    if n_labels == 0:
        return np.zeros_like(img_grey, bool)

    labels_flat = labels.ravel()
    img_grey_flat = img_as_float(img_grey).ravel()

    counts = np.bincount(labels_flat, minlength=n_labels + 1).astype(np.float64)
    sums = np.bincount(labels_flat, weights=img_grey_flat, minlength=n_labels + 1)
    means = np.zeros(n_labels + 1, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]

    valid_means = means[1:][means[1:] > 0]
    thr = threshold_otsu(valid_means) if valid_means.size > 1 else float(valid_means.min()) - 1.0

    bright_bg = _background_is_bright(img_as_float(img_grey), params)
    keep = (means <= thr) if bright_bg else (means >= thr)
    keep[0] = False
    return np.array(keep[labels])


def _background_is_bright(img_grey: np.ndarray, params: BackgroundDetectionParams) -> bool:
    """
    Decide if background is bright using only the corners flagged True in `bg`.
    If none are flagged, return True (bright background).
    """
    H, W = img_grey.shape
    ch = max(1, int(params.corner_size_pct * H))
    cw = max(1, int(params.corner_size_pct * W))

    if not params.any_corner:
        return True

    corner_mask = np.zeros((H, W), bool)
    if params.ymin_xmin_is_bg:
        corner_mask[:ch, :cw] = True
    if params.ymin_xmax_is_bg:
        corner_mask[:ch, -cw:] = True
    if params.ymax_xmin_is_bg:
        corner_mask[-ch:, :cw] = True
    if params.ymax_xmax_is_bg:
        corner_mask[-ch:, -cw:] = True

    if not corner_mask.any():
        return True
    corner_mean = float(img_grey[corner_mask].mean())
    global_median = float(np.median(img_grey))
    return corner_mean >= global_median


def _make_solid(mask: np.ndarray) -> np.ndarray:
    """
    Make mask solid by connecting nearby regions and filling enclosed holes.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)

    # fill fully enclosed holes
    return np.array(ndimage.binary_fill_holes(mask))


def _smooth_mask(mask: np.ndarray, cycles: int) -> np.ndarray:
    """
    Apply morphological closing cycles to smooth the mask boundaries.

    Parameters
    ----------
    mask : np.ndarray
        Integer mask to smooth (0 = background, >0 = specimen labels)
    cycles : int
        Number of closing cycles to apply (0 = no smoothing)

    Returns
    -------
    np.ndarray
        Smoothed integer mask
    """
    if cycles <= 0:
        return mask

    # Convert to boolean for morphological operations
    binary_mask = mask > 0

    # Calculate adaptive radius based on image size
    H, W = mask.shape
    min_dim = min(H, W)
    # Use 1-5 pixels radius depending on image size for more noticeable smoothing
    radius = max(1, min(5, min_dim // 100))

    # Apply smoothing with progressive radius increase
    smoothed_binary = binary_mask.copy()
    for i in range(cycles):
        # Slightly increase radius with each cycle for more effective smoothing
        current_radius = radius + i
        smoothed_binary = binary_closing(smoothed_binary, disk(current_radius))

    # Convert back to integer labels, preserving the original label values
    result = np.zeros_like(mask, dtype=mask.dtype)
    for label_id in np.unique(mask[mask > 0]):
        label_mask = mask == label_id
        # Apply smoothing to this specific label
        smoothed_label = binary_closing(label_mask, disk(radius))
        result[smoothed_label] = label_id

    return result


def _filter_by_area(
    mask: np.ndarray,
    min_specimen_area_frac: float,
    n_samples: int | None = None,
) -> np.ndarray:
    """
    Keep only specimen-sized components, returning integer labels for multiple specimens.

    If n_samples is provided:
        - Remove tiny artifacts (relative min-area).
        - Keep the n_samples largest components (or all if fewer are present).

    Else:
        - Remove tiny artifacts.
        - Apply Otsu on log10(areas) to separate specimen-sized from small artifacts.
    """
    labels = measure.label(mask, connectivity=2)
    n = labels.max()
    if n == 0:
        return np.zeros_like(mask, dtype=np.int32)

    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    ids = np.arange(1, n + 1)

    # Remove very small components (likely noise/artifacts)
    H, W = mask.shape
    min_area = max(1, int(min_specimen_area_frac * H * W))
    big_enough = areas >= min_area

    if not np.any(big_enough):
        return np.zeros_like(mask, dtype=np.int32)

    areas_big = areas[big_enough]
    ids_big = ids[big_enough]

    if n_samples is not None:
        # Keep the n_samples largest components
        order = np.argsort(areas_big)[::-1]
        keep = ids_big[order[:n_samples]]
        # Create a mapping from old labels to new labels
        result = np.zeros_like(labels, dtype=np.int32)
        for new_id, old_id in enumerate(keep, 1):
            result[labels == old_id] = new_id
        return result

    # Otsu on log(area) if no explicit sample count
    la = np.log10(areas_big + 1e-9)
    thr = filters.threshold_otsu(la) if la.size > 1 else la.min() - 1.0
    keep_ids = ids_big[la > thr]

    if keep_ids.size == 0:
        return np.zeros_like(mask, dtype=np.int32)

    # Create a mapping from old labels to new labels
    result = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids, 1):
        result[labels == old_id] = new_id

    return result
