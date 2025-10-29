from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Literal

import numpy as np
import spatialdata as sd
import xarray as xr
from dask_image.ndinterp import affine_transform as da_affine
from skimage import measure
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from spatialdata._logging import logger as logg
from spatialdata.models import Labels2DModel
from spatialdata.transformations import get_transformation

from squidpy._utils import _ensure_dim_order, _get_scale_factors, _yx_from_shape

from ._utils import _flatten_channels, _get_element_data


class DETECT_TISSUE_METHOD(enum.Enum):
    OTSU = enum.auto()
    FELZENSZWALB = enum.auto()


@dataclass(slots=True)
class BackgroundDetectionParams:
    """
    Which corners are background, and how large the corner boxes should be.
    If no corners are flagged True, orientation falls back to bright background.
    """

    ymin_xmin_is_bg: bool = True
    ymax_xmin_is_bg: bool = True
    ymin_xmax_is_bg: bool = True
    ymax_xmax_is_bg: bool = True
    corner_size_pct: float = 0.01  # fraction of height/width

    @property
    def any_corner(self) -> bool:
        return any(
            (
                self.ymin_xmin_is_bg,
                self.ymax_xmin_is_bg,
                self.ymin_xmax_is_bg,
                self.ymax_xmax_is_bg,
            )
        )


@dataclass(slots=True)
class FelzenszwalbParams:
    """
    Size-aware superpixel defaults for felzenszwalb segmentation.
    """

    grid_rows: int = 100
    grid_cols: int = 100
    sigma_frac: float = 0.008  # blur = this * short side, clipped to [1, 5] px
    scale_coef: float = 0.25  # scale = coef * target_area
    min_size_coef: float = 0.20  # min_size = coef * target_area


def detect_tissue(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    scale: str = "auto",
    method: DETECT_TISSUE_METHOD | str = DETECT_TISSUE_METHOD.OTSU,
    channel_format: Literal["infer", "rgb", "rgba", "multichannel"] = "infer",
    background_detection_params: BackgroundDetectionParams | None = None,
    corners_are_background: bool = True,
    min_specimen_area_frac: float = 0.01,
    n_samples: int | None = None,
    auto_max_pixels: int = 5_000_000,
    close_holes_smaller_than_frac: float = 0.0001,
    mask_smoothing_cycles: int = 0,
    new_labels_key: str | None = None,
    inplace: bool = True,
    felzenszwalb_params: FelzenszwalbParams | None = None,
) -> np.ndarray | None:
    """
    Detect tissue regions in an image and optionally store an integer-labeled mask.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the image in ``sdata.images`` to detect tissue from.
    scale
        Scale level to use for processing. If `"auto"`, uses the smallest available scale.
        Otherwise, must be a valid scale level present in the image.
    method
        Tissue detection method. Valid options are:

            - `DETECT_TISSUE_METHOD.OTSU` or `"otsu"` - Otsu thresholding with background detection.
            - `DETECT_TISSUE_METHOD.FELZENSZWALB` or `"felzenszwalb"` - Felzenszwalb superpixel segmentation.

    channel_format
        Expected format of image channels. Valid options are:

            - `"infer"` - Automatically infer from image shape.
            - `"rgb"` - RGB image.
            - `"rgba"` - RGBA image.
            - `"multichannel"` - Multi-channel image.

    background_detection_params
        Parameters for background detection via corner regions. If `None`, uses corners
        specified by `corners_are_background` for all four corners.
    corners_are_background
        Whether corners are considered background regions. Used for orienting threshold
        if `background_detection_params` is `None`.
    min_specimen_area_frac
        Minimum fraction of image area for a region to be considered a specimen.
    n_samples
        Maximum number of specimen regions to keep. If `None`, uses Otsu thresholding
        on log10(area) to separate specimens from artifacts.
    auto_max_pixels
        Maximum number of pixels to process automatically. Images larger than this
        will be downscaled before processing.
    close_holes_smaller_than_frac
        Fraction of image area below which holes in the tissue mask are filled.
    mask_smoothing_cycles
        Number of morphological closing cycles to apply for boundary smoothing.
    new_labels_key
        Key to store the resulting labels in ``sdata.labels``. If `None`, uses
        `"{image_key}_tissue"`.
    inplace
        If `True`, stores labels in ``sdata.labels``. If `False`, returns the mask array.
    felzenszwalb_params
        Parameters for Felzenszwalb superpixel segmentation. If `None`, uses default
        size-aware parameters. Only used when `method` is `"felzenszwalb"`.

    Returns
    -------
    If `inplace = False`, returns a NumPy array of shape `(y, x)` with integer labels
    where `0` represents background and `1..K` represent different specimen regions.
    Otherwise, returns `None` and stores the labels in ``sdata.labels``.

    Notes
    -----
    The function produces an integer-labeled mask where:

        - Label `0` represents background.
        - Labels `1..K` represent different specimen regions.

    Processing is performed at an appropriate resolution and then upscaled to match
    the original image dimensions.
    """
    # Normalize method
    if isinstance(method, str):
        try:
            method = DETECT_TISSUE_METHOD[method.upper()]
        except KeyError as e:
            raise ValueError('method must be "otsu" or "felzenszwalb"') from e

    # Background params
    bgp = background_detection_params or BackgroundDetectionParams(
        ymin_xmin_is_bg=corners_are_background,
        ymax_xmin_is_bg=corners_are_background,
        ymin_xmax_is_bg=corners_are_background,
        ymax_xmax_is_bg=corners_are_background,
    )

    manual_scale = scale.lower() != "auto"

    # Load smallest available or explicit scale
    img_node = sdata.images[image_key]
    img_da = _get_element_data(img_node, scale if manual_scale else "auto", "image", image_key)
    img_src = _ensure_dim_order(img_da, "yxc")
    src_h, src_w = _yx_from_shape(img_src.shape)
    n_src_px = src_h * src_w

    # Channel flattening
    img_grey_da: xr.DataArray = _flatten_channels(img=img_src, channel_format=channel_format)

    # Decide working resolution
    need_downscale = (not manual_scale) and (n_src_px > auto_max_pixels)
    if need_downscale:
        logg.info("Downscaling for faster computation.")
        img_grey = _downscale_with_dask(img_grey=img_grey_da, target_pixels=auto_max_pixels)
    else:
        img_grey = img_grey_da.values  # may compute

    # First-pass foreground
    if method == DETECT_TISSUE_METHOD.OTSU:
        img_fg_mask_bool = _segment_otsu(img_grey=img_grey, params=bgp)
    else:
        p = felzenszwalb_params or FelzenszwalbParams()
        labels_sp = _segment_felzenszwalb(img_grey=img_grey, params=p)
        img_fg_mask_bool = _mask_from_labels_via_corners(img_grey=img_grey, labels=labels_sp, params=bgp)

    # Solidify holes
    if close_holes_smaller_than_frac > 0:
        img_fg_mask_bool = _make_solid(img_fg_mask_bool, close_holes_smaller_than_frac)

    # Keep specimen-sized components → integer labels
    img_fg_labels = _filter_by_area(
        mask=img_fg_mask_bool,
        min_specimen_area_frac=min_specimen_area_frac,
        n_samples=n_samples,
    )

    # Optional smoothing → relabel once
    img_fg_labels = _smooth_mask(img_fg_labels, mask_smoothing_cycles)

    # Upscale to full resolution
    target_shape = _get_target_upscale_shape(sdata, image_key)
    scale_matrix = _get_scaling_matrix(img_fg_labels.shape, target_shape)
    img_fg_labels_up = _affine_upscale_nearest(img_fg_labels, scale_matrix, target_shape)

    if inplace:
        lk = new_labels_key or f"{image_key}_tissue"
        sf = _get_scale_factors(sdata.images[image_key])

        sdata.labels[lk] = Labels2DModel.parse(
            data=img_fg_labels_up,
            dims=("y", "x"),
            transformations=get_transformation(sdata.images[image_key], get_all=True),
            scale_factors=sf,
        )
        return None

    # If dask-backed, return a NumPy array to honor the signature
    try:
        import dask.array as da  # noqa: F401

        if hasattr(img_fg_labels_up, "compute"):
            return np.asarray(img_fg_labels_up.compute())
    except (ImportError, AttributeError, TypeError):
        pass
    return np.asarray(img_fg_labels_up)


# Line 182 - convert dask array to numpy
def _affine_upscale_nearest(labels: np.ndarray, scale_matrix: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbor affine upscaling using dask-image. Returns dask array if available, else NumPy.
    """
    try:
        import dask.array as da

        lbl_da = da.from_array(labels, chunks="auto")
        result = da_affine(
            lbl_da,
            matrix=scale_matrix,
            offset=(0.0, 0.0),
            output_shape=target_shape,
            order=0,
            mode="constant",
            cval=0,
            output=np.int32,
        )

        return np.asarray(result)
    except (ImportError, AttributeError, TypeError):
        sy = target_shape[0] / labels.shape[0]
        sx = target_shape[1] / labels.shape[1]
        yi = np.clip((np.arange(target_shape[0]) / sy).round().astype(int), 0, labels.shape[0] - 1)
        xi = np.clip((np.arange(target_shape[1]) / sx).round().astype(int), 0, labels.shape[1] - 1)
        return np.asarray(labels[yi[:, None], xi[None, :]].astype(np.int32, copy=False))


def _get_scaling_matrix(current_shape: tuple[int, int], target_shape: tuple[int, int]) -> np.ndarray:
    """
    Affine matrix mapping output coords to input coords for scipy/dask-image.
    """
    cy, cx = current_shape
    ty, tx = target_shape
    scale_y = cy / float(ty)
    scale_x = cx / float(tx)
    return np.array([[scale_y, 0.0], [0.0, scale_x]], dtype=float)


def _get_target_upscale_shape(sdata: sd.SpatialData, image_key: str) -> tuple[int, int]:
    """
    Select the first multiscale level (assumed largest) or the single-scale shape.
    """
    img = sdata.images[image_key]

    # Image2DModel-like
    if hasattr(img, "image"):
        return _yx_from_shape(img.image.shape)

    # Multiscale dict-like: first key is largest by convention
    if hasattr(img, "keys"):
        keys = list(img.keys())
        target_scale = keys[0]
        h, w = _yx_from_shape(img[target_scale].image.shape)
        return (h, w)

    # Raw array fallback
    return _yx_from_shape(img.shape)


def _downscale_with_dask(img_grey: xr.DataArray, target_pixels: int) -> np.ndarray:
    """
    Downscale (y, x) with xarray.coarsen(mean) until H*W <= target_pixels. Returns NumPy array.
    """
    h, w = img_grey.shape
    n = h * w
    if n <= target_pixels:
        return _dask_compute(_ensure_dask(img_grey))

    scale = float(np.sqrt(target_pixels / float(n)))  # 0 < scale < 1
    target_h = max(1, int(h * scale))
    target_w = max(1, int(w * scale))

    fy = max(1, int(np.ceil(h / target_h)))
    fx = max(1, int(np.ceil(w / target_w)))
    logg.info(f"Downscaling from {h}×{w} with coarsen={fy}×{fx} to ≤{target_pixels} px.")

    da_small = _ensure_dask(img_grey).coarsen(y=fy, x=fx, boundary="trim").mean()
    return np.asarray(_dask_compute(da_small))


def _ensure_dask(da: xr.DataArray) -> xr.DataArray:
    """
    Ensure DataArray is dask-backed. If not, chunk to reasonable tiles.
    """
    try:
        import dask.array as dask_array  # noqa: F401

        if hasattr(da, "data") and isinstance(da.data, dask_array.Array):
            return da
        return da.chunk({"y": 2048, "x": 2048})
    except (ImportError, AttributeError):
        return da


def _dask_compute(img_da: xr.DataArray) -> np.ndarray:
    """
    Compute an xarray DataArray (possibly dask-backed) to a NumPy array with a ProgressBar if available.
    """
    try:
        import dask.array as dask_array  # noqa: F401
        from dask.diagnostics import ProgressBar

        if hasattr(img_da, "data") and isinstance(img_da.data, dask_array.Array):
            with ProgressBar():
                computed = img_da.data.compute()
                return np.asarray(computed)
        return np.asarray(img_da.values)
    except (ImportError, AttributeError, TypeError):
        return np.asarray(img_da.values)


def _segment_otsu(img_grey: np.ndarray, params: BackgroundDetectionParams) -> np.ndarray:
    """
    Otsu binarization with orientation from background corners.
    """
    img_f = img_as_float(img_grey)
    t = threshold_otsu(img_f)
    bright_bg = _background_is_bright(img_f, params)
    return np.array((img_f <= t) if bright_bg else (img_f >= t))


def _segment_felzenszwalb(img_grey: np.ndarray, params: FelzenszwalbParams) -> np.ndarray:
    """
    Felzenszwalb superpixels with size-aware parameters.
    """
    h, w = img_grey.shape
    short = min(h, w)
    sigma = float(np.clip(params.sigma_frac * short, 1.0, 5.0))
    img_s = img_as_float(gaussian(img_grey, sigma=sigma))

    target_regions = max(1, params.grid_rows * params.grid_cols)
    target_area = (h * w) / float(target_regions)
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
    labels = labels.astype(np.int32, copy=False)
    max_lab = int(labels.max())
    if max_lab <= 0:
        return np.zeros_like(img_grey, dtype=bool)

    flat = labels.ravel()
    imgf = img_as_float(img_grey).ravel()

    counts = np.bincount(flat, minlength=max_lab + 1).astype(np.float64)
    sums = np.bincount(flat, weights=imgf, minlength=max_lab + 1)
    means = np.zeros(max_lab + 1, dtype=np.float64)
    nz = counts > 0
    means[nz] = sums[nz] / counts[nz]

    valid = means[1:]
    if valid.size > 1:
        thr = threshold_otsu(valid)
    elif valid.size == 1:
        thr = float(valid[0]) - 1.0
    else:
        thr = 0.0

    bright_bg = _background_is_bright(img_as_float(img_grey), params)
    keep = (means <= thr) if bright_bg else (means >= thr)
    keep[0] = False
    return np.array(keep[labels], dtype=bool)


def _background_is_bright(img_grey: np.ndarray, params: BackgroundDetectionParams) -> bool:
    """
    Decide if background is bright using flagged corners.
    If none are flagged or mask ends up empty, return True.
    """
    H, W = img_grey.shape
    ch = max(1, int(params.corner_size_pct * H))
    cw = max(1, int(params.corner_size_pct * W))

    if not params.any_corner:
        return True

    corner_mask = np.zeros((H, W), dtype=bool)
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


def _make_solid(mask: np.ndarray, close_holes_smaller_than_frac: float = 0.01) -> np.ndarray:
    """
    Fill holes smaller than the provided fraction of image area.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool, copy=False)

    max_hole_area = int(close_holes_smaller_than_frac * mask.size)
    return np.array(remove_small_holes(mask, area_threshold=max_hole_area))


def _smooth_mask(mask: np.ndarray, cycles: int) -> np.ndarray:
    """
    Apply morphological closing cycles to smooth boundaries, then relabel once.
    """
    if cycles <= 0:
        return mask.astype(np.int32, copy=False)

    binary = mask > 0
    H, W = mask.shape
    r0 = max(1, min(5, min(H, W) // 100))

    sm = binary
    for i in range(cycles):
        sm = binary_closing(sm, disk(r0 + i))

    return np.asarray(measure.label(sm, connectivity=2).astype(np.int32, copy=False))


def _filter_by_area(
    mask: np.ndarray,
    min_specimen_area_frac: float,
    n_samples: int | None = None,
) -> np.ndarray:
    """
    Keep specimen-sized connected components. Returns int32 labels.
    If n_samples is given, keep top-n by area after min-area filtering.
    Else, Otsu on log10(area) separates specimens from small artifacts.
    """
    labels = measure.label(mask.astype(bool, copy=False), connectivity=2)
    n = int(labels.max())
    if n == 0:
        return np.zeros_like(labels, dtype=np.int32)

    areas = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(np.int64)
    ids = np.arange(1, n + 1, dtype=np.int32)

    H, W = labels.shape
    min_area = max(1, int(min_specimen_area_frac * H * W))
    big_enough = areas >= min_area
    if not np.any(big_enough):
        return np.zeros_like(labels, dtype=np.int32)

    areas_big = areas[big_enough]
    ids_big = ids[big_enough]

    if n_samples is not None:
        order = np.argsort(areas_big)[::-1]
        keep = ids_big[order[:n_samples]]
        out = np.zeros_like(labels, dtype=np.int32)
        for new_id, old_id in enumerate(keep, 1):
            out[labels == old_id] = new_id
        return out

    la = np.log10(areas_big + 1e-9)
    thr = threshold_otsu(la) if la.size > 1 else la.min() - 1.0
    keep_ids = ids_big[la > thr]
    if keep_ids.size == 0:
        return np.zeros_like(labels, dtype=np.int32)

    out = np.zeros_like(labels, dtype=np.int32)
    for new_id, old_id in enumerate(keep_ids, 1):
        out[labels == old_id] = new_id
    return out
