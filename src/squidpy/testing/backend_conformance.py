"""Conformance test suite for squidpy backends.

Usage in backend CI::

    from squidpy.testing.backend_conformance import validate_backend

    validate_backend("rapids_singlecell")
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from anndata import AnnData

# Tolerance registry per function
TOLERANCES: dict[str, dict[str, float]] = {
    "spatial_autocorr": {"atol": 1e-5, "rtol": 1e-5},
    "co_occurrence": {"atol": 1e-6, "rtol": 1e-6},
    "nhood_enrichment": {"atol": 1e-6, "rtol": 1e-5},
}
_CONFORMANCE_ERRORS = (AssertionError, AttributeError, ImportError, KeyError, RuntimeError, TypeError, ValueError)


def _make_test_adata(n_obs: int = 144, n_vars: int = 6) -> AnnData:
    """Create deterministic data with planted spatial signal."""
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
    from squidpy._backends import get_backend

    backend = get_backend(backend_name)
    assert backend is not None, f"Backend {backend_name!r} not found"

    adata = _make_test_adata()

    # Build spatial graph (required for spatial_autocorr)
    sq.gr.spatial_neighbors(adata)

    all_tests = {
        "spatial_autocorr": _test_spatial_autocorr,
        "co_occurrence": _test_co_occurrence,
        "nhood_enrichment": _test_nhood_enrichment,
    }

    to_test = {k: v for k, v in all_tests.items() if functions is None or k in functions}

    results: dict[str, str] = {}
    failures: dict[str, BaseException] = {}
    for name, test_fn in to_test.items():
        method = getattr(backend, name, None)
        if method is None:
            results[name] = "SKIPPED (not implemented)"
            continue
        try:
            test_fn(adata, backend_name)
            results[name] = "PASSED"
        except _CONFORMANCE_ERRORS as e:
            results[name] = f"FAILED: {e}"
            failures[name] = e

    if failures:
        summary = "; ".join(f"{name}: {results[name]}" for name in failures)
        first = next(iter(failures.values()))
        raise AssertionError(f"Backend conformance failed: {summary}") from first

    return results


def _assert_numeric_frame_close(cpu_result, backend_result, *, name: str) -> None:
    np.testing.assert_array_equal(
        cpu_result.index.to_numpy(),
        backend_result.index.to_numpy(),
        err_msg=f"{name} index mismatch",
    )
    numeric_cols = cpu_result.select_dtypes(include=np.number).columns
    np.testing.assert_array_equal(
        numeric_cols.to_numpy(),
        backend_result.select_dtypes(include=np.number).columns.to_numpy(),
        err_msg=f"{name} numeric columns mismatch",
    )
    np.testing.assert_allclose(
        cpu_result[numeric_cols].to_numpy(),
        backend_result[numeric_cols].to_numpy(),
        **TOLERANCES[name],
        err_msg=f"{name} numeric result mismatch",
    )


def _test_spatial_autocorr(adata: AnnData, backend_name: str) -> None:
    import squidpy as sq

    # CPU reference
    cpu_result = sq.gr.spatial_autocorr(adata.copy(), mode="moran", copy=True)

    # Backend result
    with sq.settings.use_backend(backend_name):
        backend_result = sq.gr.spatial_autocorr(adata.copy(), mode="moran", copy=True)

    _assert_numeric_frame_close(cpu_result, backend_result, name="spatial_autocorr")


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
    np.testing.assert_allclose(
        cpu_result[1],
        backend_result[1],
        **tol,
        err_msg="co_occurrence interval mismatch",
    )


def _test_nhood_enrichment(adata: AnnData, backend_name: str) -> None:
    import squidpy as sq

    kwargs = {
        "cluster_key": "cell_type",
        "copy": True,
        "n_jobs": 1,
        "n_perms": 32,
        "seed": 42,
        "show_progress_bar": False,
    }
    cpu_result = sq.gr.nhood_enrichment(adata.copy(), **kwargs)
    with sq.settings.use_backend(backend_name):
        backend_result = sq.gr.nhood_enrichment(adata.copy(), **kwargs)

    tol = TOLERANCES["nhood_enrichment"]
    np.testing.assert_array_equal(
        cpu_result.counts,
        backend_result.counts,
        err_msg="nhood_enrichment counts mismatch",
    )
    np.testing.assert_allclose(
        cpu_result.zscore,
        backend_result.zscore,
        equal_nan=True,
        **tol,
        err_msg="nhood_enrichment z-score mismatch",
    )
