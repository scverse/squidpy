"""AnnData-facing wrappers for experimental STalign."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from squidpy.experimental.tl._stalign_helpers import extract_landmarks, extract_points
from squidpy.experimental.tl.stalign_tools import STalignConfig, STalignResult, stalign_points

__all__ = ["stalign"]


def stalign(
    adata_src: AnnData,
    adata_tgt: AnnData,
    *,
    src_key: str = "spatial",
    tgt_key: str = "spatial",
    src_landmarks_key: str | None = None,
    tgt_landmarks_key: str | None = None,
    config: STalignConfig | None = None,
    inplace: bool = True,
) -> STalignResult:
    """
    Align point coordinates stored on two AnnData objects.

    This is the high-level experimental wrapper around
    :func:`squidpy.experimental.tl.stalign_tools.stalign_points`.
    It reads source and target coordinates from ``adata_src.obsm[src_key]``
    and ``adata_tgt.obsm[tgt_key]``, optionally reads landmark coordinates
    from ``.obsm`` or ``.uns`` via ``src_landmarks_key`` and
    ``tgt_landmarks_key``, and runs point-cloud registration.

    Parameters
    ----------
    adata_src
        Source AnnData containing the point cloud to transform.
    adata_tgt
        Target AnnData containing the reference point cloud.
    src_key
        Key in ``adata_src.obsm`` holding source coordinates in ``(x, y)``
        order.
    tgt_key
        Key in ``adata_tgt.obsm`` holding target coordinates in ``(x, y)``
        order.
    src_landmarks_key
        Optional key in ``adata_src.obsm`` or ``adata_src.uns`` containing
        source landmark coordinates in ``(x, y)`` order.
    tgt_landmarks_key
        Optional key in ``adata_tgt.obsm`` or ``adata_tgt.uns`` containing
        target landmark coordinates in ``(x, y)`` order.
    config
        Optional STalign hyperparameter bundle. ``config.preprocess`` controls
        rasterization and ``config.registration`` controls LDDMM fitting.
    inplace
        If ``True``, store a serializable summary of the fitted result under
        ``adata_src.uns["stalign"]``. The fitted result object is returned in
        all cases.

    Returns
    -------
    STalignResult
        Fitted registration result. The returned object exposes
        ``transform_points(...)`` and ``transform_adata(...)`` helpers.
    """
    source_points_xy = extract_points(adata_src, key=src_key)
    target_points_xy = extract_points(adata_tgt, key=tgt_key)
    source_points = source_points_xy[:, [1, 0]]
    target_points = target_points_xy[:, [1, 0]]

    if (src_landmarks_key is None) != (tgt_landmarks_key is None):
        raise ValueError("Expected both landmark keys to be provided together.")

    if src_landmarks_key is None:
        landmarks_source = None
        landmarks_target = None
    else:
        landmarks_source = extract_landmarks(adata_src, key=src_landmarks_key)[:, [1, 0]]
        landmarks_target = extract_landmarks(adata_tgt, key=tgt_landmarks_key)[:, [1, 0]]

    result = stalign_points(
        source_points,
        target_points,
        config=config,
        landmarks_source=landmarks_source,
        landmarks_target=landmarks_target,
    )
    result.point_order = "xy"
    result.aligned_points = np.asarray(result.aligned_points)[:, [1, 0]]

    if inplace:
        adata_src.uns["stalign"] = {
            "result": _result_to_uns(result),
            "src_key": src_key,
            "tgt_key": tgt_key,
            "src_landmarks_key": src_landmarks_key,
            "tgt_landmarks_key": tgt_landmarks_key,
        }

    return result


def _result_to_uns(result: STalignResult) -> dict[str, object]:
    return {
        "affine": np.asarray(result.affine),
        "velocity": np.asarray(result.velocity),
        "velocity_grid": {
            "row": np.asarray(result.velocity_grid[0]),
            "col": np.asarray(result.velocity_grid[1]),
        },
        "aligned_points": np.asarray(result.aligned_points),
        "point_order": result.point_order,
    }
