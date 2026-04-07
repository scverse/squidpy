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
) -> STalignResult:
    """Extract coordinates from AnnData and align them with STalign."""
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
    return result
