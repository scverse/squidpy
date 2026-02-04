"""Tests for GPU functionality (skipped in CI without GPU)."""

from __future__ import annotations

import numpy as np
import pytest

import squidpy as sq
from squidpy._settings import settings

# Skip all tests in this module if GPU is not available
pytestmark = pytest.mark.skipif(
    not settings.gpu_available,
    reason="GPU tests require rapids-singlecell to be installed",
)


@pytest.fixture
def adata_filtered(adata):
    """Filter adata to only include genes with non-zero variance.

    This avoids NaN values in GPU spatial_autocorr due to constant genes.
    """
    # Calculate variance per gene and filter out zero/low variance genes
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    gene_var = np.var(X, axis=0)
    # Keep genes with variance > small threshold to avoid numerical issues
    valid_genes = gene_var > 1e-6
    return adata[:, valid_genes].copy()


class TestGPUCoOccurrence:
    """Test GPU-accelerated co_occurrence function."""

    def test_co_occurrence_gpu(self, adata_filtered):
        """Test co_occurrence with GPU device."""
        with settings.use_device("gpu"):
            result = sq.gr.co_occurrence(adata_filtered, cluster_key="leiden", copy=True)

        assert result is not None
        arr, interval = result
        assert arr.ndim == 3
        assert arr.shape[1] == arr.shape[0] == adata_filtered.obs["leiden"].nunique()

    def test_co_occurrence_gpu_vs_cpu(self, adata_filtered):
        """Test that GPU and CPU results are approximately equal."""
        with settings.use_device("cpu"):
            cpu_arr, cpu_interval = sq.gr.co_occurrence(adata_filtered, cluster_key="leiden", copy=True)
        with settings.use_device("gpu"):
            gpu_arr, gpu_interval = sq.gr.co_occurrence(adata_filtered, cluster_key="leiden", copy=True)

        np.testing.assert_allclose(cpu_interval, gpu_interval, rtol=1e-5)
        np.testing.assert_allclose(cpu_arr, gpu_arr, rtol=1e-5)


class TestGPUSpatialAutocorr:
    """Test GPU-accelerated spatial_autocorr function."""

    def test_spatial_autocorr_gpu(self, adata_filtered):
        """Test spatial_autocorr with GPU device."""
        sq.gr.spatial_neighbors(adata_filtered)
        with settings.use_device("gpu"):
            result = sq.gr.spatial_autocorr(adata_filtered, mode="moran", copy=True)

        assert result is not None
        assert "I" in result.columns
        assert "pval_norm" in result.columns

    def test_spatial_autocorr_gpu_vs_cpu(self, adata_filtered):
        """Test that GPU and CPU results are approximately equal."""
        sq.gr.spatial_neighbors(adata_filtered)
        with settings.use_device("cpu"):
            cpu_result = sq.gr.spatial_autocorr(adata_filtered, mode="moran", copy=True)
        with settings.use_device("gpu"):
            gpu_result = sq.gr.spatial_autocorr(adata_filtered, mode="moran", copy=True)

        np.testing.assert_allclose(cpu_result["I"].values, gpu_result["I"].values, rtol=1e-3, equal_nan=True)


class TestGPULigrec:
    """Test GPU-accelerated ligrec function."""

    def test_ligrec_gpu(self, adata_filtered):
        """Test ligrec with GPU device."""
        with settings.use_device("gpu"):
            result = sq.gr.ligrec(adata_filtered, cluster_key="leiden", copy=True)

        assert result is not None
        assert "means" in result
        assert "pvalues" in result
