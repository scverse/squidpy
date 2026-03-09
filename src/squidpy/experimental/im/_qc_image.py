from __future__ import annotations

from collections import defaultdict
from typing import Literal

import dask
import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from dask.diagnostics import ProgressBar
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import TableModel

from squidpy._utils import _ensure_dim_order
from squidpy.experimental.im._qc_metrics import _HNE_METRICS, InputKind, QCMetric, get_metric_info
from squidpy.experimental.im._utils import (
    TileGrid,
    _ensure_tissue_mask,
    _get_element_data,
    _get_mask_dask,
    _save_tile_grid_to_shapes,
)

_DEFAULT_HNE_METRICS: list[QCMetric] = [
    QCMetric.TENENGRAD,
    QCMetric.VAR_OF_LAPLACIAN,
    QCMetric.ENTROPY,
    QCMetric.BRIGHTNESS_MEAN,
    QCMetric.HEMATOXYLIN_MEAN,
    QCMetric.EOSIN_MEAN,
]

_DEFAULT_GENERIC_METRICS: list[QCMetric] = [
    QCMetric.TENENGRAD,
    QCMetric.VAR_OF_LAPLACIAN,
    QCMetric.ENTROPY,
    QCMetric.BRIGHTNESS_MEAN,
]


def qc_image(
    sdata: SpatialData,
    image_key: str,
    *,
    scale: str = "scale0",
    metrics: QCMetric | list[QCMetric] | None = None,
    tile_size: Literal["auto"] | tuple[int, int] = "auto",
    is_hne: bool = True,
    detect_outliers: bool = True,
    detect_tissue: bool = True,
    outlier_threshold: float = 0.1,
    progress: bool = True,
    tissue_mask_key: str | None = None,
    preview: bool = True,
) -> None:
    """
    Perform quality control analysis on an image.

    Computes tile-based QC metrics including sharpness, intensity, staining
    (H&E only), artifact detection, and tissue coverage.

    Parameters
    ----------
    sdata
        SpatialData object containing the image.
    image_key
        Key of the image in ``sdata.images`` to analyze.
    scale
        Scale level to use for processing. Defaults to ``"scale0"``.
    metrics
        QC metrics to compute. Can be a single metric or list of metrics.
        If ``None``, uses sensible defaults based on ``is_hne``.
    tile_size
        Size of tiles for analysis. If ``"auto"``, automatically determines size.
    is_hne
        Whether the image is H&E stained. Controls which metrics are available
        and which defaults are used. If ``False`` and H&E-specific metrics are
        explicitly requested, raises ``ValueError``.
    detect_outliers
        Whether to detect outlier tiles based on QC scores.
    detect_tissue
        Whether to detect tissue regions for context-aware outlier detection.
    outlier_threshold
        Percentile threshold (0-1) for outlier detection. A tile is flagged as
        an outlier if its within-tissue percentile rank falls below this value.
        For example, ``0.1`` flags the bottom 10% of tissue tiles (by their
        worst sharpness metric) as outliers. Default is ``0.1``.
    progress
        Whether to show progress bar during computation.
    tissue_mask_key
        Key of the tissue mask in ``sdata.labels`` to use. If ``None``, the function will
        check if ``"{image_key}_tissue"`` already exists in ``sdata.labels`` and reuse it.
        If it doesn't exist, tissue detection will be performed and the mask will be added
        to ``sdata.labels`` with key ``"{image_key}_tissue"``. If provided, the existing
        mask at this key will be used.
    preview
        If ``True``, render a preview showing the image overlaid with outlier tiles
        highlighted in red. Only shown when ``detect_outliers=True``.

    Returns
    -------
    None
        Results are stored in the following locations:

        - ``sdata.tables[f"qc_img_{image_key}"]``: AnnData object with QC scores
        - ``sdata.shapes[f"qc_img_{image_key}_grid"]``: GeoDataFrame with tile geometries
        - ``sdata.tables[...].uns["qc_image"]``: Metadata about the analysis
    """
    # Parameter validation
    if image_key not in sdata.images:
        raise KeyError(f"Image key '{image_key}' not found in sdata.images")

    if metrics is None:
        metrics = list(_DEFAULT_HNE_METRICS if is_hne else _DEFAULT_GENERIC_METRICS)
    elif isinstance(metrics, QCMetric):
        metrics = [metrics]
    else:
        metrics = list(metrics)

    if not isinstance(metrics, list) or not all(isinstance(m, QCMetric) for m in metrics):
        available = ", ".join(m.value for m in QCMetric)
        raise TypeError(f"metrics must be QCMetric or list of QCMetric. Available: {available}")

    # Validate H&E constraint
    if not is_hne:
        hne_requested = _HNE_METRICS & set(metrics)
        if hne_requested:
            names = ", ".join(m.value for m in hne_requested)
            raise ValueError(
                f"H&E-specific metrics ({names}) cannot be used when is_hne=False. "
                f"Set is_hne=True or remove these metrics."
            )

    if not 0 < outlier_threshold < 1:
        raise ValueError(f"outlier_threshold must be in (0, 1), got {outlier_threshold}")

    # Compute QC metrics
    img_node = sdata.images[image_key]
    img_da = _get_element_data(img_node, scale, "image", image_key)
    img_yxc = _ensure_dim_order(img_da, "yxc")
    gray = _to_gray_dask_yx(img_yxc)
    H, W = int(gray.shape[0]), int(gray.shape[1])

    tg = TileGrid(H, W, tile_size)
    tile_indices = tg.indices()
    obs_names = tg.names()

    logger.info("Quantifying image quality.")
    logger.info(f"- Input image (x, y): ({W}, {H})")
    logger.info(f"- Tile size (x, y): ({tg.tx}, {tg.ty})")
    logger.info(f"- Number of tiles (n_x, n_y): ({tg.tiles_x}, {tg.tiles_y})")

    # Group metrics by InputKind
    groups: dict[InputKind, list[QCMetric]] = defaultdict(list)
    for m in metrics:
        kind, _fn = get_metric_info(m)
        groups[kind].append(m)

    out_chunks = ((1,) * tg.tiles_y, (1,) * tg.tiles_x)

    # Prepare inputs lazily (only for needed kinds)
    prepared_inputs: dict[InputKind, da.Array] = {}

    # Resolve tissue mask once if needed by MASK metrics or tissue detection
    _tissue_mask_da: da.Array | None = None
    if InputKind.MASK in groups or detect_tissue:
        mask_key_resolved = _ensure_tissue_mask(sdata, image_key, scale, tissue_mask_key)
        _tissue_mask_da = _get_mask_dask(sdata, mask_key_resolved, scale)

    if InputKind.GRAYSCALE in groups:
        prepared_inputs[InputKind.GRAYSCALE] = gray.rechunk((tg.ty, tg.tx))

    if InputKind.RGB in groups:
        src_dtype = img_yxc.data.dtype
        rgb_arr = img_yxc.data[..., :3].astype(np.float32, copy=False)
        # Normalize integer images to [0, 1]
        if np.issubdtype(src_dtype, np.integer):
            rgb_arr = rgb_arr / float(np.iinfo(src_dtype).max)
        prepared_inputs[InputKind.RGB] = rgb_arr.rechunk((tg.ty, tg.tx, 3))

    if InputKind.MASK in groups and _tissue_mask_da is not None:
        binary = (_tissue_mask_da > 0).astype(np.float32).rechunk((tg.ty, tg.tx))
        prepared_inputs[InputKind.MASK] = binary

    # Build all dask graphs lazily
    delayed_scores: dict[str, da.Array] = {}
    for m in metrics:
        kind, metric_func = get_metric_info(m)
        source = prepared_inputs[kind]

        if kind == InputKind.RGB:
            # RGB tiles are (ty, tx, 3) -> metric returns (1, 1)
            # drop_axis=2 removes the channel dim from the output
            delayed_scores[m.value] = da.map_blocks(
                metric_func, source, dtype=np.float32, chunks=out_chunks, drop_axis=2
            )
        else:
            delayed_scores[m.value] = da.map_blocks(metric_func, source, dtype=np.float32, chunks=out_chunks)
        logger.info(f"- Calculating metric: '{m.value}'")

    # Single dask.compute() across all metric types
    if progress:
        with ProgressBar():
            results = dask.compute(*delayed_scores.values())
    else:
        results = dask.compute(*delayed_scores.values())
    all_scores: dict[str, np.ndarray] = dict(zip(delayed_scores.keys(), results, strict=True))

    # Build AnnData
    metric_names = [m.value for m in metrics]
    first = next(iter(all_scores.values()))
    cents, _ = tg.centroids_and_polygons()
    n_tiles = first.size
    X = np.column_stack([all_scores[n].ravel() for n in metric_names])
    var_names = [f"qc_{n}" for n in metric_names]

    adata = AnnData(X=X)
    adata.var_names = var_names
    adata.obs_names = obs_names
    adata.obs["centroid_y"] = cents[:, 0]
    adata.obs["centroid_x"] = cents[:, 1]
    adata.obsm["spatial"] = cents

    # Defaults to avoid NameError when skipping tissue/outliers
    tissue = np.zeros(n_tiles, dtype=bool)
    back = ~tissue
    t_sim = np.zeros(n_tiles, np.float32)
    b_sim = np.zeros(n_tiles, np.float32)
    outlier_labels = np.ones(n_tiles, dtype=int)
    unfocus_scores: np.ndarray | None = None

    if detect_outliers:
        if detect_tissue:
            tissue, back, t_sim, b_sim = _detect_tissue_from_mask(
                sdata,
                image_key,
                tile_indices,
                tg.ty,
                tg.tx,
                scale,
                tissue_mask_key=tissue_mask_key,
                mask_da=_tissue_mask_da,
            )
            logger.info(f"- Classified tiles: background: {back.sum()}, tissue: {tissue.sum()}.")

        if detect_tissue and tissue.any():
            tissue_scores = _compute_unfocus_scores(X[tissue], var_names)
            unfocus_scores = np.full(n_tiles, np.nan, dtype=np.float32)
            unfocus_scores[tissue] = tissue_scores
            outlier_labels = np.ones(n_tiles, dtype=int)
            outlier_labels[tissue] = np.where(tissue_scores >= 1 - outlier_threshold, -1, 1)
        else:
            # No tissue mask: score all tiles
            all_scores_arr = _compute_unfocus_scores(X, var_names)
            unfocus_scores = all_scores_arr
            outlier_labels = np.where(all_scores_arr >= 1 - outlier_threshold, -1, 1)

        adata.obs["qc_outlier"] = pd.Categorical((outlier_labels == -1).astype(str), categories=["False", "True"])
        if detect_tissue:
            adata.obs["is_tissue"] = pd.Categorical(tissue.astype(str), categories=["False", "True"])
            adata.obs["is_background"] = pd.Categorical(back.astype(str), categories=["False", "True"])
            adata.obs["tissue_similarity"] = t_sim
            adata.obs["background_similarity"] = b_sim
        adata.obs["unfocus_score"] = unfocus_scores

        logger.info(f"- Detected {int((outlier_labels == -1).sum())} outlier tiles.")

    adata.uns["qc_image"] = {
        "metrics": list(all_scores.keys()),
        "tile_size_y": tg.ty,
        "tile_size_x": tg.tx,
        "image_height": H,
        "image_width": W,
        "n_tiles_y": tg.tiles_y,
        "n_tiles_x": tg.tiles_x,
        "image_key": image_key,
        "scale": scale,
        "is_hne": is_hne,
        "detect_tissue": detect_tissue,
        "outlier_threshold": outlier_threshold,
        "n_tissue_tiles": int(tissue.sum()),
        "n_background_tiles": int(back.sum()),
        "n_outlier_tiles": int((outlier_labels == -1).sum()),
    }

    table_key = f"qc_img_{image_key}"
    shapes_key = f"qc_img_{image_key}_grid"

    # Build shapes first (need the index for tile_id linkage)
    _save_tile_grid_to_shapes(sdata, tg, shapes_key, copy_transforms_from_key=image_key)

    # Set spatialdata linkage on adata BEFORE TableModel.parse
    adata.obs["grid_name"] = pd.Categorical([shapes_key] * len(adata))
    adata.obs["tile_id"] = sdata.shapes[shapes_key].index
    adata.uns["spatialdata_attrs"] = {
        "region": shapes_key,
        "region_key": "grid_name",
        "instance_key": "tile_id",
    }

    sdata.tables[table_key] = TableModel.parse(adata)
    logger.info(f"- Saved QC scores as 'sdata.tables[\"{table_key}\"]'")

    if preview and detect_outliers and "qc_outlier" in adata.obs.columns:
        try:
            (
                sdata.pl.render_images(image_key)
                .pl.render_shapes(
                    shapes_key,
                    color="qc_outlier",
                    groups="True",
                    palette="red",
                    fill_alpha=0.5,
                    table_name=table_key,
                )
                .pl.show()
            )
        except (AttributeError, KeyError, ValueError) as e:
            logger.warning(f"Could not generate preview plot: {e}")


def _to_gray_dask_yx(
    img_yxc: xr.DataArray,
    weights: tuple[float, float, float] = (0.2126, 0.7152, 0.0722),
) -> da.Array:
    """Convert multi-channel image to grayscale using luminance weights."""
    arr = img_yxc.data
    if arr.ndim != 3:
        raise ValueError(f"Expected image with shape `(y, x, c)`, found `{arr.shape}`.")
    c = arr.shape[2]
    if c == 1:
        return arr[..., 0].astype(np.float32, copy=False)
    rgb = arr[..., :3].astype(np.float32, copy=False)
    w = da.from_array(np.asarray(weights, dtype=np.float32), chunks=(3,))
    gray = da.tensordot(rgb, w, axes=([2], [0]))
    return gray.astype(np.float32, copy=False)


def _detect_tissue_from_mask(
    sdata: SpatialData,
    image_key: str,
    tile_indices: np.ndarray,
    ty: int,
    tx: int,
    scale: str = "scale0",
    tissue_mask_key: str | None = None,
    mask_da: da.Array | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect tissue regions from mask and classify tiles."""
    n_tiles = len(tile_indices)

    if mask_da is None:
        mask_key_resolved = _ensure_tissue_mask(sdata, image_key, scale, tissue_mask_key)
        mask_da = _get_mask_dask(sdata, mask_key_resolved, scale)
    H, W = mask_da.shape
    tiles_y = (H + ty - 1) // ty
    tiles_x = (W + tx - 1) // tx

    binary = (mask_da > 0).astype(np.float32).rechunk((ty, tx))

    def _mean_block(block: np.ndarray) -> np.ndarray:
        return np.array([[float(block.mean())]], dtype=np.float32)

    frac_da = da.map_blocks(
        _mean_block,
        binary,
        dtype=np.float32,
        chunks=((1,) * tiles_y, (1,) * tiles_x),
    )
    frac_grid = frac_da.compute()
    frac = frac_grid.ravel()[:n_tiles]

    tissue = frac > 0.5
    back = ~tissue
    t_sim = tissue.astype(np.float32)
    b_sim = back.astype(np.float32)

    return tissue, back, t_sim, b_sim


def _compute_unfocus_scores(X: np.ndarray, var_names: list[str]) -> np.ndarray:
    """Compute per-tile unfocus scores using within-group percentile ranks.

    For each sharpness metric, every tile is ranked among its peers (the tiles
    in ``X``) and assigned a percentile in [0, 1] where 0 = lowest value =
    worst quality and 1 = highest.  The final ``unfocus_score`` for a tile is
    ``1 - min_rank`` across all *sharpness* metrics, so a tile that scores
    poorly on *any* sharpness axis gets a high unfocus score.

    Only gradient-based sharpness metrics (tenengrad, var_of_laplacian)
    contribute to the score.  Other metrics like entropy, wavelet energy,
    or FFT energy correlate more with tissue structure than with actual
    optical focus and are therefore excluded.

    Parameters
    ----------
    X
        Score matrix of shape ``(n_tiles, n_metrics)``.
    var_names
        Column names matching ``X`` (``qc_`` prefix expected).

    Returns
    -------
    np.ndarray of shape ``(n_tiles,)`` with values in [0, 1].
    """
    from scipy.stats import rankdata

    _SHARPNESS_KEYWORDS = {"tenengrad", "laplacian"}

    sharpness_cols = [i for i, name in enumerate(var_names) if any(kw in name.lower() for kw in _SHARPNESS_KEYWORDS)]
    if not sharpness_cols:
        sharpness_cols = list(range(X.shape[1]))

    n = X.shape[0]
    if n <= 1:
        return np.zeros(n, dtype=np.float32)

    # Percentile rank per metric: 0 = worst, 1 = best
    ranks = np.column_stack([(rankdata(X[:, col], method="average") - 1) / (n - 1) for col in sharpness_cols])

    # Worst (minimum) rank across sharpness metrics governs the score
    min_rank = ranks.min(axis=1)

    # Invert: high unfocus_score = low quality
    return (1.0 - min_rank).astype(np.float32)
