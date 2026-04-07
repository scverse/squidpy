"""Conformance test suite for squidpy backends.

Usage in backend CI::

    from squidpy.testing.backend_conformance import validate_backend
    validate_backend("rapids_singlecell")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData

if TYPE_CHECKING:
    from collections.abc import Sequence

# Tolerance registry per function
TOLERANCES: dict[str, dict[str, float]] = {
    "spatial_autocorr": {"atol": 1e-5, "rtol": 1e-3},
    "co_occurrence": {"atol": 1e-5, "rtol": 1e-2},
}


def _make_test_adata(n_obs: int = 500, n_vars: int = 100) -> AnnData:
    """Create a minimal AnnData for testing."""
    rng = np.random.default_rng(42)
    adata = AnnData(X=rng.random((n_obs, n_vars)).astype(np.float32))
    adata.obsm["spatial"] = rng.random((n_obs, 2)).astype(np.float32) * 1000
    adata.obs["cell_type"] = rng.choice(["A", "B", "C"], size=n_obs)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype("category")
    return adata


def validate_backend(
    backend_name: str,
    functions: Sequence[str] | None = None,
) -> dict[str, str]:
    """Run conformance tests against a backend.

    Parameters
    ----------
    backend_name
        Name or alias of the backend to test.
    functions
        Specific functions to test. None = test all known functions.

    Returns
    -------
    Dict mapping function name to result string.

    Raises
    ------
    AssertionError
        If any test fails.
    """
    import squidpy as sq
    from squidpy._backends._registry import get_backend

    backend = get_backend(backend_name)
    assert backend is not None, f"Backend {backend_name!r} not found"

    adata = _make_test_adata()

    # Build spatial graph (required for spatial_autocorr)
    sq.gr.spatial_neighbors(adata)

    all_tests = {
        "spatial_autocorr": _test_spatial_autocorr,
        "co_occurrence": _test_co_occurrence,
    }

    to_test = {k: v for k, v in all_tests.items() if functions is None or k in functions}

    results: dict[str, str] = {}
    for name, test_fn in to_test.items():
        method = getattr(backend, name, None)
        if method is None:
            results[name] = "SKIPPED (not implemented)"
            continue
        try:
            test_fn(adata, backend_name)
            results[name] = "PASSED"
        except Exception as e:
            results[name] = f"FAILED: {e}"
            raise

    return results


def _test_spatial_autocorr(adata: AnnData, backend_name: str) -> None:
    import squidpy as sq

    # CPU reference
    cpu_result = sq.gr.spatial_autocorr(adata.copy(), mode="moran", copy=True)

    # Backend result
    with sq.settings.use_backend(backend_name):
        backend_result = sq.gr.spatial_autocorr(adata.copy(), mode="moran", copy=True)

    tol = TOLERANCES["spatial_autocorr"]
    np.testing.assert_allclose(
        cpu_result["I"].values,
        backend_result["I"].values,
        **tol,
        err_msg="spatial_autocorr Moran's I mismatch",
    )


def _test_co_occurrence(adata: AnnData, backend_name: str) -> None:
    import squidpy as sq

    cpu_result = sq.gr.co_occurrence(adata.copy(), cluster_key="cell_type", copy=True)
    with sq.settings.use_backend(backend_name):
        backend_result = sq.gr.co_occurrence(adata.copy(), cluster_key="cell_type", copy=True)

    tol = TOLERANCES["co_occurrence"]
    np.testing.assert_allclose(
        cpu_result[0],
        backend_result[0],
        **tol,
        err_msg="co_occurrence probability mismatch",
    )
