"""Benchmark fixtures and configuration for squidpy performance tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from squidpy._constants._pkg_constants import Key

if TYPE_CHECKING:
    from collections.abc import Callable


def pytest_configure(config: pytest.Config) -> None:
    """Register benchmark markers."""
    config.addinivalue_line("markers", "benchmark: mark test as a benchmark test")


@pytest.fixture(scope="session")
def make_adata() -> Callable[[int, int], AnnData]:
    """
    Factory fixture to create synthetic AnnData objects for benchmarking.

    Returns a function that generates AnnData with specified n_obs and n_clusters.
    """

    def _make_adata(n_obs: int, n_clusters: int = 10) -> AnnData:
        """
        Create a synthetic AnnData object for benchmarking.

        Parameters
        ----------
        n_obs
            Number of observations (cells).
        n_clusters
            Number of cluster categories.

        Returns
        -------
        AnnData object with spatial coordinates and cluster labels.
        """
        rng = np.random.default_rng(42)

        # Create random spatial coordinates
        spatial_coords = rng.uniform(0, 1000, size=(n_obs, 2))

        # Create random cluster assignments
        cluster_labels = pd.Categorical(
            rng.choice([f"cluster_{i}" for i in range(n_clusters)], size=n_obs)
        )

        # Create minimal expression matrix
        X = rng.random((n_obs, 50))

        # Build AnnData
        adata = AnnData(X=X)
        adata.obsm[Key.obsm.spatial] = spatial_coords
        adata.obs["cluster"] = cluster_labels

        return adata

    return _make_adata


# Pre-defined dataset sizes for parameterized benchmarks
# Adjust these values to match your benchmarking needs
BENCHMARK_SIZES = {
    "1k": 1_000,
    "5k": 5_000,
    "10k": 10_000,
    "50k": 50_000,
    "100k": 100_000,
}


@pytest.fixture(params=list(BENCHMARK_SIZES.keys()), ids=list(BENCHMARK_SIZES.keys()))
def adata_scaling(
    request: pytest.FixtureRequest, make_adata: Callable[[int, int], AnnData]
) -> AnnData:
    """
    Parameterized fixture that provides AnnData objects of varying sizes.

    The fixture name makes it clear that the dataset size varies across test runs.
    Sizes are defined in BENCHMARK_SIZES dict - modify that to change scale points.
    """
    size_name = request.param
    n_obs = BENCHMARK_SIZES[size_name]
    return make_adata(n_obs)


# Default size for non-scaling benchmarks (uses first size in BENCHMARK_SIZES)
DEFAULT_BENCHMARK_SIZE = next(iter(BENCHMARK_SIZES.values()))


@pytest.fixture
def adata_default(make_adata: Callable[[int, int], AnnData]) -> AnnData:
    """Fixed dataset for non-scaling benchmarks. Size defined by DEFAULT_BENCHMARK_SIZE."""
    return make_adata(DEFAULT_BENCHMARK_SIZE)
