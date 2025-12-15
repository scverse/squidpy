from __future__ import annotations

from typing import Literal

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from dask.diagnostics import ProgressBar
from sklearn.preprocessing import StandardScaler
from spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata.models import TableModel

from squidpy._utils import _ensure_dim_order

from ._detect_tissue import detect_tissue
from ._make_tiles import _TileGrid, _save_tiles_to_shapes
from ._sharpness_metrics import SharpnessMetric, _get_sharpness_metric_function
from ._utils import _flatten_channels, _get_element_data


def qc_sharpness(
    sdata: SpatialData,
    image_key: str,
    *,
    scale: str = "scale0",
    metrics: SharpnessMetric | list[SharpnessMetric] | None = None,
    tile_size: Literal["auto"] | tuple[int, int] = "auto",
    detect_outliers: bool = True,
    detect_tissue: bool = True,
    outlier_method: Literal["pvalue", "iqr", "zscore", "tenengrad_tissue"] = "pvalue",
    outlier_cutoff: float = 0.1,
    progress: bool = True,
    tissue_mask_key: str | None = None,
) -> None:
    """
    Perform tile-based quality control analysis on image sharpness.
    """
    if image_key not in sdata.images:
        raise KeyError(f"Image key '{image_key}' not found in sdata.images")

    metrics = _resolve_metrics(metrics)
    if outlier_method not in ["pvalue", "iqr", "zscore", "tenengrad_tissue"]:
        raise ValueError(
            f"Unknown outlier_method '{outlier_method}'. Must be one of: pvalue, iqr, zscore, tenengrad_tissue"
        )

    img_node = sdata.images[image_key]
    img_da = _get_element_data(img_node, scale, "image", image_key)
    img_cyx = _ensure_dim_order(img_da, "cyx")
    gray = _to_gray_dask_yx(img_cyx)
    H, W = _yx_from_array(gray)

    tg = _TileGrid(H, W, tile_size=tile_size)
    tile_indices = tg.indices()
    obs_names = tg.names()

    logger.info("Quantifying image sharpness.")
    logger.info("- Input image (x, y): (%s, %s)", W, H)
    logger.info("- Tile size (x, y): (%s, %s)", tg.tx, tg.ty)
    logger.info("- Number of tiles (n_x, n_y): (%s, %s)", tg.tiles_x, tg.tiles_y)

    metric_names = [(m.value if isinstance(m, SharpnessMetric) else str(m)) for m in metrics]
    all_scores: dict[str, np.ndarray] = {}
    for name in metric_names:
        metric_func = _get_sharpness_metric_function(name)
        gray_re = gray.rechunk((tg.ty, tg.tx))
        field = da.map_overlap(metric_func, gray_re, depth=0, boundary="reflect", dtype=np.float32)
        padded = tg.rechunk_and_pad(field)
        if name == SharpnessMetric.TENENGRAD.value:
            tiles_da = tg.coarsen(padded, "sum") / float(tg.ty * tg.tx)
        else:
            tiles_da = tg.coarsen(padded, "mean")

        logger.info("- Calculating metric: '%s'", name)
        if progress:
            with ProgressBar():
                all_scores[name] = tiles_da.compute()
        else:
            all_scores[name] = tiles_da.compute()

    first = next(iter(all_scores.values()))
    cents, _ = tg.centroids_and_polygons()
    n_tiles = first.size
    X = np.column_stack([all_scores[n].ravel() for n in metric_names])
    var_names = [f"sharpness_{n}" for n in metric_names]

    adata = AnnData(X=X)
    adata.var_names = var_names
    adata.obs_names = obs_names
    adata.obs["centroid_y"] = cents[:, 0]
    adata.obs["centroid_x"] = cents[:, 1]
    adata.obsm["spatial"] = cents

    tissue = np.zeros(n_tiles, dtype=bool)
    back = ~tissue
    t_sim = np.zeros(n_tiles, np.float32)
    b_sim = np.zeros(n_tiles, np.float32)
    outlier_labels = np.ones(n_tiles, dtype=int)
    unfocus_scores: np.ndarray | None = None

    if detect_outliers:
        if detect_tissue:
            tissue, back, t_sim, b_sim = _detect_tissue_from_mask(
                sdata, image_key, tile_indices, tg.ty, tg.tx, scale, tissue_mask_key=tissue_mask_key
            )
            logger.info("- Classified tiles: background: %s, tissue: %s.", back.sum(), tissue.sum())

        if detect_tissue and tissue.any():
            if outlier_method == "pvalue":
                labels, pvals = _detect_outliers_pvalue(X, tissue_mask=tissue, var_names=var_names)
                scores = 1.0 - pvals
                lo, hi = float(np.min(scores)), float(np.max(scores))
                scores = (scores - lo) / (hi - lo) if hi > lo else np.zeros_like(scores)
                outlier_labels = np.where(scores >= outlier_cutoff, -1, 1)
                unfocus_scores = scores
            elif outlier_method == "tenengrad_tissue":
                scores = _detect_tenengrad_tissue_outliers(X, tissue_mask=tissue, var_names=var_names)
                outlier_labels = np.where(scores >= outlier_cutoff, -1, 1)
                unfocus_scores = scores
            else:
                tX = X[tissue]
                t_labels = _detect_sharpness_outliers(tX, method=outlier_method)
                outlier_labels = np.ones(n_tiles, dtype=int)
                outlier_labels[tissue] = t_labels
        else:
            method = "zscore" if outlier_method == "pvalue" else outlier_method
            outlier_labels = _detect_sharpness_outliers(X, method=method)

        adata.obs["sharpness_outlier"] = pd.Categorical(
            (outlier_labels == -1).astype(str), categories=["False", "True"]
        )
        if detect_tissue:
            adata.obs["is_tissue"] = pd.Categorical(tissue.astype(str), categories=["False", "True"])
            adata.obs["is_background"] = pd.Categorical(back.astype(str), categories=["False", "True"])
            adata.obs["tissue_similarity"] = t_sim
            adata.obs["background_similarity"] = b_sim
        if unfocus_scores is not None:
            adata.obs["unfocus_score"] = unfocus_scores

        logger.info("- Detected %s outlier tiles.", int((outlier_labels == -1).sum()))

    adata.uns["qc_sharpness"] = {
        "metrics": list(all_scores.keys()),
        "tile_size_y": tg.ty,
        "tile_size_x": tg.tx,
        "image_height": H,
        "image_width": W,
        "n_tiles_y": tg.tiles_y,
        "n_tiles_x": tg.tiles_x,
        "image_key": image_key,
        "scale": scale,
        "detect_tissue": detect_tissue,
        "outlier_method": outlier_method,
        "n_tissue_tiles": int(tissue.sum()),
        "n_background_tiles": int(back.sum()),
        "n_outlier_tiles": int((outlier_labels == -1).sum()),
    }

    table_key = f"qc_img_{image_key}_sharpness"
    shapes_key = f"qc_img_{image_key}_sharpness_grid"

    # Add SpatialData attrs/columns before parsing so downstream validation succeeds.
    adata.uns["spatialdata_attrs"] = {
        "region": shapes_key,
        "region_key": "grid_name",
        "instance_key": "tile_id",
    }
    adata.obs["grid_name"] = pd.Series([shapes_key] * len(adata), dtype=str)
    adata.obs["tile_id"] = pd.Series(obs_names, dtype=str)

    sdata.tables[table_key] = TableModel.parse(adata)
    logger.info("- Saved sharpness scores as 'sdata.tables[\"%s\"]'", table_key)

    _save_tiles_to_shapes(sdata, tg, image_key, shapes_key)
    logger.info("- Saved tile grid as 'sdata.shapes[\"%s\"]'", shapes_key)


def _to_gray_dask_yx(img_cyx: xr.DataArray) -> da.Array:
    arr = _flatten_channels(img_cyx, channel_format="infer").data
    darr = da.from_array(arr, chunks="auto") if not isinstance(arr, da.Array) else arr
    return darr.astype(np.float32, copy=False)


def _yx_from_array(arr: da.Array) -> tuple[int, int]:
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale array (y, x), found shape {arr.shape}")
    return int(arr.shape[0]), int(arr.shape[1])


def _get_mask_from_labels(sdata: SpatialData, mask_key: str, scale: str) -> np.ndarray:
    label_node = sdata.labels[mask_key]
    mask_da = _get_element_data(label_node, scale, "label", mask_key)
    if hasattr(mask_da, "compute"):
        mask = np.asarray(mask_da.compute())
    elif hasattr(mask_da, "values"):
        mask = np.asarray(mask_da.values)
    else:
        mask = np.asarray(mask_da)
    if mask.ndim > 2:
        mask = mask.squeeze()
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask with shape (y, x), got shape {mask.shape}")
    return mask


def _detect_tissue_from_mask(
    sdata: SpatialData,
    image_key: str,
    tile_indices: np.ndarray,
    ty: int,
    tx: int,
    scale: str = "scale0",
    tissue_mask_key: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_tiles = len(tile_indices)
    if tissue_mask_key is None:
        mask_key = f"{image_key}_tissue"
        if mask_key not in sdata.labels:
            detect_tissue(sdata=sdata, image_key=image_key, scale=scale, inplace=True, new_labels_key=mask_key)
            logger.info("- Saved tissue mask as 'sdata.labels[\"%s\"]'", mask_key)
        mask = _get_mask_from_labels(sdata, mask_key, scale)
    elif tissue_mask_key not in sdata.labels:
        raise KeyError(f"Tissue mask key '{tissue_mask_key}' not found in sdata.labels")
    else:
        mask = _get_mask_from_labels(sdata, tissue_mask_key, scale)
    if mask is None:
        logger.warning("Tissue mask missing. Marking all tiles as tissue.")
        t = np.ones(n_tiles, dtype=bool)
        b = ~t
        return t, b, np.ones(n_tiles, np.float32), np.zeros(n_tiles, np.float32)

    H, W = mask.shape
    tissue = np.zeros(n_tiles, dtype=bool)
    back = np.zeros(n_tiles, dtype=bool)
    t_sim = np.zeros(n_tiles, dtype=np.float32)
    b_sim = np.zeros(n_tiles, dtype=np.float32)
    for i, (iy, ix) in enumerate(tile_indices):
        y0, y1 = iy * ty, min((iy + 1) * ty, H)
        x0, x1 = ix * tx, min((ix + 1) * tx, W)
        frac = float(np.mean(mask[y0:y1, x0:x1] > 0.0)) if (y1 > y0 and x1 > x0) else 0.0
        is_t = frac > 0.5
        tissue[i] = is_t
        back[i] = not is_t
        t_sim[i] = 1.0 if is_t else 0.0
        b_sim[i] = 0.0 if is_t else 1.0

    return tissue, back, t_sim, b_sim


def _clean_sharpness_data(X: np.ndarray) -> np.ndarray:
    Xc: np.ndarray = X.copy()
    Xc[np.isinf(Xc)] = np.nan
    for i in range(Xc.shape[1]):
        col = Xc[:, i]
        if np.any(np.isnan(col)):
            med = np.nanmedian(col)
            Xc[np.isnan(col), i] = med
        lo, hi = np.percentile(Xc[:, i], [0.1, 99.9])
        clipped = np.clip(Xc[:, i], lo, hi)
        Xc[:, i] = clipped
    return Xc


def _detect_outliers_iqr(X_scaled: np.ndarray) -> np.ndarray:
    Q1 = np.percentile(X_scaled, 25, axis=0)
    Q3 = np.percentile(X_scaled, 75, axis=0)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    mask = np.any(X_scaled < lower, axis=1)
    return np.where(mask, -1, 1)


def _detect_outliers_zscore(X_scaled: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    mask = np.any(X_scaled < -threshold, axis=1)
    return np.where(mask, -1, 1)


def _detect_sharpness_outliers(
    X: np.ndarray, method: str = "iqr", tissue_mask: np.ndarray | None = None, var_names: list[str] | None = None
) -> np.ndarray:
    Xc = _clean_sharpness_data(X)
    if method == "tenengrad_tissue":
        return _detect_tenengrad_tissue_outliers(Xc, tissue_mask, var_names)
    if method == "pvalue":
        return _detect_outliers_pvalue(Xc, tissue_mask, var_names)[0]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xc)
    if method == "iqr":
        return _detect_outliers_iqr(Xs)
    if method == "zscore":
        return _detect_outliers_zscore(Xs)
    raise ValueError(f"Unknown method '{method}'. Use 'iqr', 'zscore', 'tenengrad_tissue', or 'pvalue'.")


def _detect_tenengrad_tissue_outliers(
    X: np.ndarray, tissue_mask: np.ndarray | None = None, var_names: list[str] | None = None
) -> np.ndarray:
    if tissue_mask is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(_clean_sharpness_data(X))
        return _detect_outliers_zscore(Xs)

    bg_mask = ~tissue_mask
    if bg_mask.sum() == 0:
        return np.zeros(len(X))
    ten_idx = None
    if var_names is not None:
        for i, n in enumerate(var_names):
            if "tenengrad" in n.lower():
                ten_idx = i
                break
    if ten_idx is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(_clean_sharpness_data(X))
        return _detect_outliers_zscore(Xs)

    t = X[tissue_mask, ten_idx]
    b = X[bg_mask, ten_idx]
    bmin, bmax = float(np.min(b)), float(np.max(b))
    rng = bmax - bmin
    if rng <= 0:
        mu_t, sd_t = float(np.mean(t)), float(np.std(t)) + 1e-10
        scores = np.clip((mu_t - t) / sd_t, 0, 1)
    else:
        norm = np.clip((t - bmin) / rng, 0, 1)
        scores = 1.0 - norm
    out = np.zeros(len(X))
    out[np.where(tissue_mask)[0]] = scores
    return out


def _detect_outliers_pvalue(
    X: np.ndarray, tissue_mask: np.ndarray | None = None, var_names: list[str] | None = None, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    from scipy import stats

    if tissue_mask is None:
        tissue_mask = np.ones(len(X), dtype=bool)
    if var_names is None:
        var_names = [f"metric_{i}" for i in range(X.shape[1])]
    tX = X[tissue_mask]
    bX = X[~tissue_mask]
    if len(bX) < 10:
        return np.ones(len(X), dtype=int), np.ones(len(X))
    P = np.ones((len(tX), len(var_names)))
    for i in range(len(var_names)):
        bg = bX[:, i]
        mu, sd = float(np.mean(bg)), float(np.std(bg))
        if sd < 1e-10:
            continue
        P[:, i] = stats.norm.cdf(tX[:, i], loc=mu, scale=sd)
    minP = np.min(P, axis=1)
    fullP = np.ones(len(X))
    fullP[np.where(tissue_mask)[0]] = minP
    out = np.where(fullP < alpha, -1, 1)
    return out, fullP


def _resolve_metrics(metrics: SharpnessMetric | list[SharpnessMetric] | None) -> list[SharpnessMetric]:
    if metrics is None:
        return [SharpnessMetric.TENENGRAD, SharpnessMetric.VAR_OF_LAPLACIAN]
    if isinstance(metrics, SharpnessMetric):
        return [metrics]
    if not isinstance(metrics, list) or not all(isinstance(m, SharpnessMetric) for m in metrics):
        available = ", ".join(m.value for m in SharpnessMetric)
        raise TypeError(f"metrics must be SharpnessMetric or list of SharpnessMetric. Available: {available}")
    return metrics
