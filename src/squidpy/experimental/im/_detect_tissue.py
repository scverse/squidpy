from __future__ import annotations

import enum
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import dask.array as da
import numpy as np
import spatialdata as sd
import xarray as xr
from dask.base import is_dask_collection
from dask_image.ndinterp import affine_transform as da_affine
from skimage import feature, future, measure
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import binary_closing, disk, remove_small_holes
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from spatialdata._logging import logger
from spatialdata.models import Labels2DModel
from spatialdata.transformations import get_transformation

from squidpy._utils import _ensure_dim_order, _get_scale_factors, _yx_from_shape

from ._utils import _flatten_channels, _get_element_data


class DetectTissueMethod(enum.Enum):
    OTSU = enum.auto()
    FELZENSZWALB = enum.auto()
    WEKA = enum.auto()


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


@dataclass(slots=True)
class WekaParams:
    """
    Parameters for WEKA-like trainable segmentation.
    """

    sigma_min: float = 1.0
    sigma_max: float = 16.0
    edges: bool = True
    pseudo_tissue_percentile: float = 90.0  # percentile of distance-from-bg to label as tissue
    pseudo_min_pixels: int = 50  # minimum number of tissue pixels to seed
    rf_estimators: int = 100
    rf_max_depth: int | None = 10
    rf_max_samples: float = 0.05
    random_state: int | None = 0

    # Second-stage refinement with a simple classifier
    refine_with_classifier: bool = True
    refine_n_samples_per_class: int = 50_000
    refine_bg_prob_threshold: float = 0.6  # only drop pixels very likely to be background


def _normalize_margins(
    margins_px: int | Sequence[int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Normalize border margins to a 4-tuple (top, bottom, left, right) of non-negative ints.
    If the requested margins would eclipse the image, returns all zeros.
    """
    if isinstance(margins_px, (str, bytes)):
        raise TypeError("`border_margin_px` must be an int or a sequence of 4 ints, not a string.")

    if isinstance(margins_px, Sequence):
        margins_list = list(margins_px)
        if len(margins_list) != 4:
            raise ValueError("`border_margin_px` must be an int or a sequence of 4 ints (top, bottom, left, right).")
        try:
            t, b, l, r = (int(x) for x in margins_list)
        except (TypeError, ValueError) as e:  # pragma: no cover - defensive
            raise TypeError("`border_margin_px` entries must be convertible to int.") from e
    else:
        t = b = l = r = int(margins_px)

    if any(v < 0 for v in (t, b, l, r)):
        raise ValueError("`border_margin_px` values must be non-negative.")

    H, W = shape
    if t + b >= H or l + r >= W:
        return (0, 0, 0, 0)
    return (t, b, l, r)


def _is_zero_margin(margins_px: int | Sequence[int]) -> bool:
    """
    Check whether margins resolve to zero everywhere.
    """
    try:
        if isinstance(margins_px, Sequence) and not isinstance(margins_px, (str, bytes)):
            return all(int(x) == 0 for x in margins_px)
        return int(margins_px) == 0
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return False


def _build_inner_mask(shape: tuple[int, int], margins: tuple[int, int, int, int]) -> np.ndarray:
    """
    Mask that excludes a border defined by (top, bottom, left, right).
    """
    H, W = shape
    t, b, l, r = margins
    mask = np.ones((H, W), dtype=bool)
    if t > 0:
        mask[:t, :] = False
    if b > 0:
        mask[-b:, :] = False
    if l > 0:
        mask[:, :l] = False
    if r > 0:
        mask[:, -r:] = False
    return mask


def _apply_border_margin(mask: np.ndarray, margins: tuple[int, int, int, int]) -> np.ndarray:
    """
    Zero-out a border defined by (top, bottom, left, right) around a boolean mask.
    """
    if margins == (0, 0, 0, 0):
        return mask

    t, b, l, r = margins
    out = np.array(mask, copy=True)
    if t > 0:
        out[:t, :] = False
    if b > 0:
        out[-b:, :] = False
    if l > 0:
        out[:, :l] = False
    if r > 0:
        out[:, -r:] = False
    return out


def _rescale_margins(
    margins: tuple[int, int, int, int],
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """
    Rescale margins defined for `from_shape` so they apply to `to_shape`.
    """
    fy, fx = from_shape
    ty, tx = to_shape
    if fy == 0 or fx == 0:
        return (0, 0, 0, 0)

    sy = ty / float(fy)
    sx = tx / float(fx)

    t, b, l, r = margins
    scaled = (
        int(round(t * sy)),
        int(round(b * sy)),
        int(round(l * sx)),
        int(round(r * sx)),
    )
    # Ensure margins do not eclipse the target shape
    ty_eff = max(0, ty - (scaled[0] + scaled[1]))
    tx_eff = max(0, tx - (scaled[2] + scaled[3]))
    if ty_eff <= 0 or tx_eff <= 0:
        return (0, 0, 0, 0)
    return scaled


def detect_tissue(
    sdata: sd.SpatialData,
    image_key: str,
    *,
    scale: str = "auto",
    method: DetectTissueMethod | str = DetectTissueMethod.OTSU,
    method_params: FelzenszwalbParams | WekaParams | Mapping[str, Any] | None = None,
    channel_format: Literal["infer", "rgb", "rgba", "multichannel"] = "infer",
    background_detection_params: BackgroundDetectionParams | None = None,
    corners_are_background: bool = True,
    border_margin_px: int | Sequence[int] = 0,
    min_specimen_area_frac: float = 0.01,
    n_samples: int | None = None,
    auto_max_pixels: int = 5_000_000,
    close_holes_smaller_than_frac: float = 0.0001,
    mask_smoothing_cycles: int = 0,
    new_labels_key: str | None = None,
    inplace: bool = True,
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

            - `DetectTissueMethod.OTSU` or `"otsu"` - Otsu thresholding with background detection.
            - `DetectTissueMethod.FELZENSZWALB` or `"felzenszwalb"` - Felzenszwalb superpixel segmentation.
            - `DetectTissueMethod.WEKA` or `"weka"` - Trainable segmentation with corner background priors and RGB multiscale features.
    method_params
        Optional parameters specific to the selected method. For `"felzenszwalb"`, provide a
        :class:`FelzenszwalbParams` instance or a mapping of its fields. For `"weka"`, provide a
        :class:`WekaParams` instance or mapping. Passing values when ``method="otsu"`` is not supported.
    border_margin_px
        Ignore a border when seeding and predicting tissue. Can be:

            - a single int applied to all sides, or
            - a sequence of four ints ``(top, bottom, left, right)``.

        Useful for masking out fiducial rings or slide edges. Applied consistently across all methods.

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
            method = DetectTissueMethod[method.upper()]
        except KeyError as e:
            raise ValueError('method must be "otsu", "felzenszwalb", or "weka"') from e

    logger.info(f"Detecting tissue with method: {method}")

    if method == DetectTissueMethod.WEKA and not corners_are_background:
        raise ValueError("WEKA tissue detection requires corner background priors; set corners_are_background=True.")

    if method == DetectTissueMethod.OTSU:
        if method_params is not None:
            raise ValueError("`method_params` are not supported for OTSU tissue detection.")
        resolved_method_params = None
    elif method == DetectTissueMethod.FELZENSZWALB:
        if method_params is None:
            resolved_method_params = FelzenszwalbParams()
        elif isinstance(method_params, FelzenszwalbParams):
            resolved_method_params = method_params
        elif isinstance(method_params, Mapping):
            resolved_method_params = FelzenszwalbParams(**method_params)
        else:
            raise TypeError(
                f"`method_params` for 'felzenszwalb' must be a FelzenszwalbParams or mapping, "
                f"got {type(method_params).__name__}.",
            )
    elif method == DetectTissueMethod.WEKA:
        if method_params is None:
            resolved_method_params = WekaParams()
        elif isinstance(method_params, WekaParams):
            resolved_method_params = method_params
        elif isinstance(method_params, Mapping):
            resolved_method_params = WekaParams(**method_params)
        else:
            raise TypeError(
                f"`method_params` for 'weka' must be a WekaParams or mapping, got {type(method_params).__name__}.",
            )
    else:
        raise ValueError(f"Unsupported method: {method}")

    # Background params
    bgp = background_detection_params or BackgroundDetectionParams(
        ymin_xmin_is_bg=corners_are_background,
        ymax_xmin_is_bg=corners_are_background,
        ymin_xmax_is_bg=corners_are_background,
        ymax_xmax_is_bg=corners_are_background,
    )

    manual_scale = scale.lower() != "auto"
    normalized_margins_target = (0, 0, 0, 0)  # set after image load for shape-aware validation

    # Load smallest available or explicit scale
    img_node = sdata.images[image_key]
    img_da = _get_element_data(img_node, scale if manual_scale else "auto", "image", image_key)
    img_src = _ensure_dim_order(img_da, "yxc")
    src_h = int(img_src.sizes["y"])
    src_w = int(img_src.sizes["x"])
    n_src_px = src_h * src_w
    base_margin_px = border_margin_px
    if method == DetectTissueMethod.WEKA and _is_zero_margin(base_margin_px):
        wp_local = cast(WekaParams, resolved_method_params)
        base_margin_px = getattr(wp_local, "border_margin_px", 0)
    target_shape = _get_target_upscale_shape(sdata, image_key)
    normalized_margins_target = _normalize_margins(base_margin_px, target_shape)

    # Decide working resolution
    need_downscale = (not manual_scale) and (n_src_px > auto_max_pixels)

    # Channel flattening (greyscale) for threshold-based methods
    img_grey = None
    if method != DetectTissueMethod.WEKA:
        img_grey_da: xr.DataArray = _flatten_channels(img=img_src, channel_format=channel_format)
        if need_downscale:
            logger.info("Downscaling for faster computation.")
            img_grey = _downscale_with_dask(img_grey=img_grey_da, target_pixels=auto_max_pixels)
        else:
            img_grey = img_grey_da.values  # may compute

    # Prepare color image for WEKA (keeps channels)
    if method == DetectTissueMethod.WEKA:
        if need_downscale:
            logger.info("Downscaling for faster computation.")
            img_weka = _downscale_with_dask_multichannel(img_rgb=img_src, target_pixels=auto_max_pixels)
        else:
            img_weka = np.asarray(_dask_compute(_ensure_dask(img_src)))
        working_shape = img_weka.shape[:2]
    else:
        working_shape = img_grey.shape if img_grey is not None else (src_h, src_w)

    normalized_margins = _rescale_margins(
        normalized_margins_target,
        from_shape=target_shape,
        to_shape=working_shape,
    )

    # First-pass foreground
    if method == DetectTissueMethod.OTSU:
        img_fg_mask_bool = _segment_otsu(img_grey=img_grey, params=bgp)
        img_fg_mask_bool = _apply_border_margin(img_fg_mask_bool, normalized_margins)
    elif method == DetectTissueMethod.WEKA:
        wp = cast(WekaParams, resolved_method_params)
        img_fg_mask_bool = _segment_weka(
            img=img_weka,
            params=bgp,
            weka_params=wp,
            border_margins_px=normalized_margins,
        )
    else:
        p = cast(FelzenszwalbParams, resolved_method_params)
        labels_sp = _segment_felzenszwalb(img_grey=img_grey, params=p)
        img_fg_mask_bool = _mask_from_labels_via_corners(img_grey=img_grey, labels=labels_sp, params=bgp)
        img_fg_mask_bool = _apply_border_margin(img_fg_mask_bool, normalized_margins)

    logger.info("Finished segmentation.")

    # Solidify holes
    if close_holes_smaller_than_frac > 0:
        img_fg_mask_bool = _make_solid(img_fg_mask_bool, close_holes_smaller_than_frac)

    # Keep specimen-sized components → integer labels
    img_fg_labels = _filter_by_area(
        mask=img_fg_mask_bool,
        min_specimen_area_frac=min_specimen_area_frac,
        n_samples=n_samples,
    )

    img_fg_labels = _smooth_mask(img_fg_labels, mask_smoothing_cycles)
    img_fg_labels = _apply_border_margin(img_fg_labels, normalized_margins)

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
        logger.info(f"Saved tissue mask to `sdata.labels['{lk}']`.")
        return None

    # If dask-backed, return a NumPy array to honor the signature
    if is_dask_collection(img_fg_labels_up):
        return np.asarray(img_fg_labels_up.compute())

    return np.asarray(img_fg_labels_up)


# Line 182 - convert dask array to numpy
def _affine_upscale_nearest(labels: np.ndarray, scale_matrix: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Nearest-neighbor affine upscaling using dask-image. Returns dask array if available, else NumPy.
    """
    logger.info("Upscaling mask back to original size.")
    try:
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

    da_small = _ensure_dask(img_grey).coarsen(y=fy, x=fx, boundary="trim").mean()
    return np.asarray(_dask_compute(da_small))


def _downscale_with_dask_multichannel(img_rgb: xr.DataArray, target_pixels: int) -> np.ndarray:
    """
    Downscale multichannel (y, x, c) with xarray.coarsen(mean) until H*W <= target_pixels. Returns NumPy array.
    """
    h = int(img_rgb.sizes["y"]) if hasattr(img_rgb, "sizes") and "y" in img_rgb.sizes else img_rgb.shape[0]
    w = int(img_rgb.sizes["x"]) if hasattr(img_rgb, "sizes") and "x" in img_rgb.sizes else img_rgb.shape[1]
    n = h * w
    if n <= target_pixels:
        return np.asarray(_dask_compute(_ensure_dask(img_rgb)))

    scale = float(np.sqrt(target_pixels / float(n)))  # 0 < scale < 1
    target_h = max(1, int(h * scale))
    target_w = max(1, int(w * scale))

    fy = max(1, int(np.ceil(h / target_h)))
    fx = max(1, int(np.ceil(w / target_w)))

    da_small = _ensure_dask(img_rgb).coarsen(y=fy, x=fx, boundary="trim").mean()
    return np.asarray(_dask_compute(da_small))


def _ensure_dask(arr: xr.DataArray) -> xr.DataArray:
    """
    Ensure DataArray is dask-backed. If not, chunk to reasonable tiles.
    """
    try:
        if hasattr(arr, "data") and isinstance(arr.data, da.Array):
            return arr
        return arr.chunk({"y": 2048, "x": 2048})
    except (ImportError, AttributeError):
        return arr


def _dask_compute(img_da: xr.DataArray) -> np.ndarray:
    """
    Compute an xarray DataArray (possibly dask-backed) to a NumPy array with a ProgressBar if available.
    """
    try:
        from dask.diagnostics import ProgressBar

        if hasattr(img_da, "data") and isinstance(img_da.data, da.Array):
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


def _segment_weka(
    img: np.ndarray,
    params: BackgroundDetectionParams,
    weka_params: WekaParams,
    border_margins_px: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> np.ndarray:
    """
    Trainable segmentation using multiscale features and a RandomForest classifier.

    Logic:
        - Use corners as reliable background seeds (label 1).
        - Compute a distance-from-background "zmap" over the image.
        - Mark the top `pseudo_tissue_percentile` of non-corner pixels as tissue seeds (label 2),
          enforcing at least `pseudo_min_pixels` seeds.
        - All remaining pixels stay 0 (unlabeled) and are ignored during training.
        - Train a RandomForest on these seeds in multiscale feature space and predict tissue (class 2).
        - Optionally refine the resulting mask with a second-stage classifier that
          sees inside-mask pixels as candidate tissue and outside as background.
    """
    img_f = img_as_float(img)

    if img_f.ndim == 2:
        channel_axis = None
        img_mono = img_f
    elif img_f.ndim == 3 and img_f.shape[2] <= 4:
        channel_axis = -1
        img_mono = img_f.mean(axis=2)
    else:  # pragma: no cover - defensive
        raise ValueError("WEKA segmentation expects 2D or 3D (RGB/RGBA) input.")

    H, W = img_mono.shape
    inner_mask = _build_inner_mask((H, W), border_margins_px)

    # Multiscale features (WEKA-like)
    feats = feature.multiscale_basic_features(
        img_f,
        intensity=True,
        edges=weka_params.edges,
        texture=True,
        sigma_min=weka_params.sigma_min,
        sigma_max=weka_params.sigma_max,
        channel_axis=channel_axis,
    )

    # Label image: 0 = unlabeled, 1 = background, 2 = tissue
    training_labels = np.zeros((H, W), dtype=np.uint8)

    # Background seeds from corners
    corner_mask = _corner_mask((H, W), params)
    if not corner_mask.any():
        # Fallback: small block in top-left if corners disabled
        h_block = max(1, H // 50)
        w_block = max(1, W // 50)
        corner_mask[:h_block, :w_block] = True
    training_labels[corner_mask] = 1
    if any(border_margins_px):
        # Treat excluded border as background seeds to down-weight fiducial rings.
        training_labels[~inner_mask] = 1

    non_bg = (~corner_mask) & inner_mask

    # Background prototype from corners
    if img_f.ndim == 2:
        # Grayscale: scalar mean/std, simple z-score magnitude
        bg_mean = float(img_f[corner_mask].mean())
        bg_std = float(img_f[corner_mask].std())
        bg_std = max(bg_std, 1e-6)
        zmap = np.abs((img_f - bg_mean) / bg_std)
    else:
        # Multichannel: per-channel z-score, then L2 norm across channels
        bg_mean = img_f[corner_mask].mean(axis=0)
        bg_std = img_f[corner_mask].std(axis=0)
        bg_std = np.maximum(bg_std, 1e-6)
        zmap = np.linalg.norm((img_f - bg_mean) / bg_std, axis=-1)

    # Pseudo tissue seeds from most non-background-like pixels
    if np.any(non_bg):
        perc = float(np.clip(weka_params.pseudo_tissue_percentile, 0.0, 100.0))
        thr = np.percentile(zmap[non_bg], perc)
        tissue_mask = (zmap >= thr) & non_bg

        # Ensure minimum number of tissue seeds
        if tissue_mask.sum() < weka_params.pseudo_min_pixels:
            flat_non_bg = np.flatnonzero(non_bg)
            if flat_non_bg.size > 0:
                z_flat = zmap.ravel()[flat_non_bg]
                order = np.argsort(z_flat)[::-1]
                n_take = min(flat_non_bg.size, weka_params.pseudo_min_pixels)
                chosen = flat_non_bg[order[:n_take]]

                seed_mask_flat = np.zeros_like(non_bg.ravel(), dtype=bool)
                seed_mask_flat[chosen] = True
                tissue_mask = seed_mask_flat.reshape(non_bg.shape)

        training_labels[tissue_mask] = 2

    # Ensure both classes exist for training
    if not (training_labels == 1).any():
        # Minimal case: force a tiny background block
        h_block = max(1, H // 50)
        w_block = max(1, W // 50)
        training_labels[:h_block, :w_block] = 1

    if not (training_labels == 2).any():
        # No tissue seeds found: pick the most different non-corner pixel if possible
        flat_non_bg = np.flatnonzero(non_bg)
        training_labels_flat = training_labels.ravel()

        if flat_non_bg.size > 0:
            z_flat = zmap.ravel()[flat_non_bg]
            idx_rel = int(np.argmax(z_flat))
            idx = int(flat_non_bg[idx_rel])
        else:
            # Full-corner image: arbitrary fallback (center pixel)
            idx = int((H * W) // 2)

        training_labels_flat[idx] = 2
        training_labels = training_labels_flat.reshape(H, W)

    clf = RandomForestClassifier(
        n_estimators=weka_params.rf_estimators,
        n_jobs=-1,
        max_depth=weka_params.rf_max_depth,
        max_samples=weka_params.rf_max_samples,
        random_state=weka_params.random_state,
    )
    clf = future.fit_segmenter(training_labels, feats, clf)
    result = future.predict_segmenter(feats, clf)

    prior_mask = np.asarray(result == 2) & inner_mask

    # Optional second-stage refinement: inside-vs-outside mask classification
    if weka_params.refine_with_classifier:
        prior_mask = _refine_with_background_classifier(
            feats=feats,
            prior_mask=prior_mask,
            n_samples_per_class=weka_params.refine_n_samples_per_class,
            bg_prob_threshold=weka_params.refine_bg_prob_threshold,
            random_state=weka_params.random_state,
        )

    return prior_mask


def _refine_with_background_classifier(
    feats: np.ndarray,
    prior_mask: np.ndarray,
    n_samples_per_class: int,
    bg_prob_threshold: float,
    random_state: int | None = 0,
) -> np.ndarray:
    """
    Refine a prior tissue mask using a simple classifier on multiscale features.

    We:
        - treat pixels inside the prior mask as candidate tissue, outside as background,
        - sample up to `n_samples_per_class` pixels from each class,
        - train a logistic regression to discriminate background(1) vs tissue(0),
        - compute p(background | x) for inside-mask pixels,
        - turn to background those prior-mask pixels with high background probability.

    Parameters
    ----------
    feats
        Feature array of shape (H, W, F) from `multiscale_basic_features`.
    prior_mask
        Boolean array (H, W) with the initial tissue mask.
    n_samples_per_class
        Maximum number of training samples to draw from each class.
    bg_prob_threshold
        Background probability threshold above which an inside-mask pixel
        is reclassified as background.
    random_state
        Seed for subsampling; use None for non-deterministic.

    Returns
    -------
    refined_mask : bool ndarray, same shape as prior_mask.
    """
    logger.info("Refining mask with background classifier.")

    H, W, F = feats.shape
    prior_flat = prior_mask.ravel()

    # Define classes based on prior mask
    idx_tissue = np.flatnonzero(prior_flat)
    idx_bg = np.flatnonzero(~prior_flat)

    if idx_tissue.size == 0 or idx_bg.size == 0:
        # Nothing to refine
        return prior_mask

    rng = np.random.default_rng(random_state)

    # Subsample for training
    n_tissue = min(n_samples_per_class, idx_tissue.size)
    n_bg = min(n_samples_per_class, idx_bg.size)

    idx_tissue_sample = rng.choice(idx_tissue, size=n_tissue, replace=False)
    idx_bg_sample = rng.choice(idx_bg, size=n_bg, replace=False)

    feats_flat = feats.reshape(-1, F)

    X_tissue = feats_flat[idx_tissue_sample]
    X_bg = feats_flat[idx_bg_sample]

    # y=1 for background, y=0 for tissue
    X_train = np.vstack([X_bg, X_tissue])
    y_train = np.concatenate(
        [
            np.ones(n_bg, dtype=np.int32),
            np.zeros(n_tissue, dtype=np.int32),
        ]
    )

    # Standardize features for better optimizer behavior
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.maximum(std, 1e-6)

    def _z(x: np.ndarray) -> np.ndarray:
        return (x - mean) / std

    X_train_z = _z(X_train)

    # Simple logistic regression classifier
    clf = LogisticRegression(
        max_iter=10_000,
        n_jobs=-1,
        solver="lbfgs",
    )
    clf.fit(X_train_z, y_train)

    # Predict p(background) only for inside-mask pixels
    p_bg_tissue = clf.predict_proba(_z(feats_flat[idx_tissue]))[:, 1]

    refined_flat = prior_flat.copy()
    # Demote high-confidence background inside the prior mask
    refined_flat[idx_tissue[p_bg_tissue >= bg_prob_threshold]] = False

    return refined_flat.reshape(H, W)


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


def _corner_mask(shape: tuple[int, int], params: BackgroundDetectionParams) -> np.ndarray:
    """
    Build a boolean mask for selected corners.
    """
    H, W = shape
    ch = max(1, int(params.corner_size_pct * H))
    cw = max(1, int(params.corner_size_pct * W))

    mask = np.zeros((H, W), dtype=bool)
    if params.ymin_xmin_is_bg:
        mask[:ch, :cw] = True
    if params.ymin_xmax_is_bg:
        mask[:ch, -cw:] = True
    if params.ymax_xmin_is_bg:
        mask[-ch:, :cw] = True
    if params.ymax_xmax_is_bg:
        mask[-ch:, -cw:] = True
    return mask


def _background_is_bright(img_grey: np.ndarray, params: BackgroundDetectionParams) -> bool:
    """
    Decide if background is bright using flagged corners.
    If none are flagged or mask ends up empty, return True.
    """
    if not params.any_corner:
        return True

    corner_mask = _corner_mask(img_grey.shape, params)
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

    logger.info(f"Smoothing mask for {cycles} cycle(s).")
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
