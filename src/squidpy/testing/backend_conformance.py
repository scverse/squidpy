"""Backend conformance helpers for Squidpy-compatible accelerators."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData
from scverse_backends.testing import run_conformance

import squidpy as sq
from squidpy._backends import get_backend

if TYPE_CHECKING:
    from collections.abc import Sequence


def _conformance_adata(n_obs: int = 144, n_vars: int = 6) -> AnnData:
    """Create deterministic data with a planted spatial signal."""
    rng = np.random.default_rng(42)
    side = int(np.ceil(np.sqrt(n_obs)))
    yy, xx = np.mgrid[:side, :side]
    coords = np.c_[xx.ravel(), yy.ravel()][:n_obs].astype(np.float32)
    scale = float(max(side - 1, 1))
    x = coords[:, 0] / scale
    y = coords[:, 1] / scale

    planted = np.column_stack(
        [
            x + y,
            x - y,
            np.sin(np.pi * x) * np.cos(np.pi * y),
            rng.normal(0.0, 0.01, size=n_obs),
        ]
    )
    if n_vars > planted.shape[1]:
        noise = rng.normal(0.0, 0.05, size=(n_obs, n_vars - planted.shape[1]))
        planted = np.column_stack([planted, noise])

    adata = AnnData(X=planted[:, :n_vars].astype(np.float32))
    adata.obsm["spatial"] = coords
    adata.obs["cell_type"] = np.select(
        [coords[:, 0] < side / 3, coords[:, 0] >= 2 * side / 3],
        ["A", "C"],
        default="B",
    )
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    sq.gr.spatial_neighbors_knn(adata)
    return adata


def _test_spatial_autocorr(backend_name: str) -> None:
    expected = _conformance_adata()
    actual = expected.copy()

    expected_result = sq.gr.spatial_autocorr(expected, mode="moran", copy=True, backend="cpu")
    actual_result = sq.gr.spatial_autocorr(actual, mode="moran", copy=True, backend=backend_name)

    np.testing.assert_array_equal(actual_result.index.to_numpy(), expected_result.index.to_numpy())
    np.testing.assert_allclose(
        actual_result.select_dtypes(include=np.number).to_numpy(),
        expected_result.select_dtypes(include=np.number).to_numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


def _test_co_occurrence(backend_name: str) -> None:
    expected = _conformance_adata()
    actual = expected.copy()

    expected_result = sq.gr.co_occurrence(expected, cluster_key="cell_type", copy=True, backend="cpu")
    actual_result = sq.gr.co_occurrence(actual, cluster_key="cell_type", copy=True, backend=backend_name)

    for actual_value, expected_value in zip(actual_result, expected_result, strict=True):
        np.testing.assert_allclose(actual_value, expected_value, rtol=1e-6, atol=1e-6)


_TESTS = {
    "spatial_autocorr": _test_spatial_autocorr,
    "co_occurrence": _test_co_occurrence,
}


def validate_backend(
    backend_name: str,
    *,
    functions: Sequence[str] | None = None,
    raise_on_failure: bool = True,
) -> dict[str, str]:
    """Run Squidpy's backend conformance checks against an installed backend."""
    return run_conformance(
        backend_name=backend_name,
        tests=_TESTS,
        get_backend=get_backend,
        functions=functions,
        raise_on_failure=raise_on_failure,
    )


__all__ = ["validate_backend"]
