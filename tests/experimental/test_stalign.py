from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from anndata import AnnData

import squidpy as sq
from squidpy.experimental.tl._stalign_core import transform_points_row_col

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


def test_transform_points_backward_reverses_nonstationary_flow():
    xv = (jnp.linspace(0.0, 2.0, 3), jnp.linspace(0.0, 2.0, 3))
    affine = jnp.eye(3)
    velocity = np.zeros((2, 3, 3, 2), dtype=float)

    for i in range(3):
        velocity[0, i, :, 1] = i
    for j in range(3):
        velocity[1, :, j, 0] = j

    points = jnp.asarray([[1.0, 1.0]])
    transformed = transform_points_row_col(xv, jnp.asarray(velocity), affine, points, direction="forward")
    restored = transform_points_row_col(xv, jnp.asarray(velocity), affine, transformed, direction="backward")

    np.testing.assert_allclose(np.asarray(restored), np.asarray(points))


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

    assert "stalign" in adata_src.uns
    assert "result" in adata_src.uns["stalign"]
    assert adata_src.uns["stalign"]["result"]["aligned_points"].shape == adata_src.obsm["spatial"].shape
    assert set(adata_src.uns["stalign"]["result"]["velocity_grid"]) == {"row", "col"}

    transformed = result.transform_adata(adata_src)
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape == adata_src.obsm["spatial"].shape

    result.transform_adata(adata_src, key_added="stalign")
    assert "stalign" in adata_src.obsm

    copied = result.transform_adata(adata_src, key_added="stalign_copy", copy=True)
    assert isinstance(copied, AnnData)
    assert "stalign_copy" in copied.obsm


def test_stalign_copy_true_does_not_write_uns():
    adata_src = _make_xy_adata()
    adata_tgt = _make_xy_adata()

    sq.experimental.tl.stalign(
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
        copy=True,
    )

    assert "stalign" not in adata_src.uns


def test_stalign_uns_payload_is_h5ad_serializable(tmp_path):
    adata_src = _make_xy_adata()
    adata_tgt = _make_xy_adata()

    sq.experimental.tl.stalign(
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

    adata_src.write_h5ad(tmp_path / "stalign.h5ad")
