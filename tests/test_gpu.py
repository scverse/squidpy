"""Tests for GPU functionality (skipped in CI without GPU).

These tests verify GPU results match CPU results. Structure/correctness
of CPU outputs is tested elsewhere, so we only test equivalence here.
"""

from __future__ import annotations

import numpy as np
import pytest

import squidpy as sq
from squidpy._settings import settings

pytestmark = pytest.mark.skipif(
    not settings.gpu_available,
    reason="GPU tests require rapids-singlecell to be installed",
)


@pytest.fixture
def adata_filtered(adata):
    """Filter adata to genes with non-zero variance (avoids NaN in GPU spatial_autocorr)."""
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_var = np.var(X, axis=0)
    return adata[:, gene_var > 1e-6].copy()


class TestGPUvsCPU:
    """Test that GPU and CPU produce equivalent results."""

    def test_co_occurrence(self, adata_filtered):
        """Test co_occurrence GPU vs CPU equivalence."""
        with settings.use_device("cpu"):
            cpu_arr, cpu_interval = sq.gr.co_occurrence(adata_filtered, cluster_key="leiden", copy=True)
        with settings.use_device("gpu"):
            gpu_arr, gpu_interval = sq.gr.co_occurrence(adata_filtered, cluster_key="leiden", copy=True)

        np.testing.assert_allclose(cpu_interval, gpu_interval, rtol=1e-5)
        np.testing.assert_allclose(cpu_arr, gpu_arr, rtol=1e-5)

    def test_spatial_autocorr(self, adata_filtered):
        """Test spatial_autocorr GPU vs CPU equivalence."""
        sq.gr.spatial_neighbors(adata_filtered)

        with settings.use_device("cpu"):
            cpu_result = sq.gr.spatial_autocorr(adata_filtered, mode="moran", copy=True)
        with settings.use_device("gpu"):
            gpu_result = sq.gr.spatial_autocorr(adata_filtered, mode="moran", copy=True)

        np.testing.assert_allclose(cpu_result["I"].values, gpu_result["I"].values, rtol=1e-3, equal_nan=True)

    def test_ligrec(self, adata_filtered):
        """Test ligrec GPU vs CPU equivalence."""
        with settings.use_device("cpu"):
            cpu_result = sq.gr.ligrec(adata_filtered, cluster_key="leiden", copy=True, n_perms=5)
        with settings.use_device("gpu"):
            gpu_result = sq.gr.ligrec(adata_filtered, cluster_key="leiden", copy=True, n_perms=5)

        # Compare means (deterministic)
        np.testing.assert_allclose(
            cpu_result["means"].values, gpu_result["means"].values, rtol=1e-5, equal_nan=True
        )
