from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

import squidpy as sq

jax = pytest.importorskip("jax")

_ = jax


def _make_row_col_points() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 2.0],
        ]
    )


def _make_xy_adata() -> AnnData:
    points_xy = np.array(
        [
            [10.0, 1.0],
            [12.0, 1.0],
            [11.0, 2.0],
            [10.0, 3.0],
            [12.0, 3.0],
        ]
    )
    adata = AnnData(np.zeros((points_xy.shape[0], 1)))
    adata.obsm["spatial"] = points_xy
    adata.uns["landmarks"] = points_xy[:3]
    return adata


def test_stalign_preprocess_returns_channels_first():
    points = _make_row_col_points()

    result = sq.experimental.tl.stalign_tools.stalign_preprocess(points, points, dx=0.5, blur=1.0)

    assert result.source_image.ndim == 3
    assert result.source_image.shape[0] == 1
    assert result.source_grid[0].ndim == 1
    assert result.source_grid[1].ndim == 1


def test_stalign_points_returns_result_and_transform_points():
    points = _make_row_col_points()

    result = sq.experimental.tl.stalign_tools.stalign_points(
        points,
        points,
        dx=0.5,
        blur=1.0,
        landmarks_source=points[:3],
        landmarks_target=points[:3],
        a=1.0,
        expand=1.0,
        nt=1,
        niter=1,
        epV=1.0,
    )

    transformed = np.asarray(result.transform_points(points, direction="forward"))
    backward = np.asarray(result.transform_points(points, direction="backward"))

    assert result.aligned_points.shape == points.shape
    assert transformed.shape == points.shape
    assert backward.shape == points.shape
    assert np.all(np.isfinite(transformed))
    assert np.all(np.isfinite(backward))


def test_stalign_wrapper_and_transform_adata_method():
    adata_src = _make_xy_adata()
    adata_tgt = _make_xy_adata()

    result = sq.experimental.tl.stalign(
        adata_src,
        adata_tgt,
        src_key="spatial",
        tgt_key="spatial",
        src_landmarks_key="landmarks",
        tgt_landmarks_key="landmarks",
        dx=0.5,
        blur=1.0,
        a=1.0,
        expand=1.0,
        nt=1,
        niter=1,
        epV=1.0,
    )

    transformed = result.transform_adata(adata_src)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == adata_src.obsm["spatial"].shape

    result.transform_adata(adata_src, key_added="stalign")
    assert "stalign" in adata_src.obsm

    copied = result.transform_adata(adata_src, key_added="stalign_copy", copy=True)
    assert isinstance(copied, AnnData)
    assert "stalign_copy" in copied.obsm
