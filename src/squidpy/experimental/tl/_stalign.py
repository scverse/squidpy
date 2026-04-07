"""AnnData-facing wrappers for experimental STalign."""

from __future__ import annotations

import numpy as np
from anndata import AnnData

from squidpy.experimental.tl._stalign_helpers import extract_landmarks, extract_points
from squidpy.experimental.tl.stalign_tools import STalignResult, stalign_points

__all__ = ["stalign"]


def stalign(
    adata_src: AnnData,
    adata_tgt: AnnData,
    *,
    src_key: str = "spatial",
    tgt_key: str = "spatial",
    src_landmarks_key: str | None = None,
    tgt_landmarks_key: str | None = None,
    dx: float = 30.0,
    blur: float | list[float] = (2.0, 1.0, 0.5),
    a: float = 500.0,
    p: float = 2.0,
    expand: float = 2.0,
    nt: int = 3,
    niter: int = 5000,
    diffeo_start: int = 0,
    epL: float = 2e-8,
    epT: float = 2e-1,
    epV: float = 2e3,
    sigmaM: float = 1.0,
    sigmaB: float = 2.0,
    sigmaA: float = 5.0,
    sigmaR: float = 5e5,
    sigmaP: float = 2e1,
    copy: bool = False,
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
    dx
        Rasterization grid spacing.
    blur
        Rasterization blur scale or scales.
    a
        Velocity field smoothness scale.
    p
        Power of the differential operator used in regularization.
    expand
        Expansion factor for the velocity field domain.
    nt
        Number of time steps used for integrating the velocity field.
    niter
        Number of optimization iterations.
    diffeo_start
        Iteration at which nonlinear velocity updates start.
    epL
        Gradient step size for the affine linear term.
    epT
        Gradient step size for the affine translation term.
    epV
        Gradient step size for the velocity field.
    sigmaM
        Matching term scale.
    sigmaB
        Background term scale.
    sigmaA
        Artifact term scale.
    sigmaR
        Velocity regularization scale.
    sigmaP
        Landmark matching scale.
    copy
        If ``False``, store a serializable summary of the fitted result under
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
        dx=dx,
        blur=blur,
        landmarks_source=landmarks_source,
        landmarks_target=landmarks_target,
        a=a,
        p=p,
        expand=expand,
        nt=nt,
        niter=niter,
        diffeo_start=diffeo_start,
        epL=epL,
        epT=epT,
        epV=epV,
        sigmaM=sigmaM,
        sigmaB=sigmaB,
        sigmaA=sigmaA,
        sigmaR=sigmaR,
        sigmaP=sigmaP,
    )
    result.point_order = "xy"
    result.aligned_points = np.asarray(result.aligned_points)[:, [1, 0]]

    if not copy:
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
