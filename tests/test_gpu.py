"""Tests for GPU functionality (skipped in CI without GPU)."""

from __future__ import annotations

import numpy as np
import pytest

import squidpy as sq
from squidpy._settings import settings

# Skip all tests in this module if GPU is not available
pytestmark = pytest.mark.skipif(
    not settings.gpu_available(),
    reason="GPU tests require rapids-singlecell to be installed",
)


class TestGPUCoOccurrence:
    """Test GPU-accelerated co_occurrence function."""

    def test_co_occurrence_gpu(self, adata):
        """Test co_occurrence with GPU device."""
        result = sq.gr.co_occurrence(adata, cluster_key="leiden", copy=True, device="gpu")

        assert result is not None
        arr, interval = result
        assert arr.ndim == 3
        assert arr.shape[1] == arr.shape[0] == adata.obs["leiden"].nunique()

    def test_co_occurrence_gpu_vs_cpu(self, adata):
        """Test that GPU and CPU results are approximately equal."""
        cpu_arr, cpu_interval = sq.gr.co_occurrence(adata, cluster_key="leiden", copy=True, device="cpu")
        gpu_arr, gpu_interval = sq.gr.co_occurrence(adata, cluster_key="leiden", copy=True, device="gpu")

        np.testing.assert_allclose(cpu_interval, gpu_interval, rtol=1e-5)
        np.testing.assert_allclose(cpu_arr, gpu_arr, rtol=1e-5)


class TestGPUSpatialAutocorr:
    """Test GPU-accelerated spatial_autocorr function."""

    def test_spatial_autocorr_gpu(self, adata):
        """Test spatial_autocorr with GPU device."""
        sq.gr.spatial_neighbors(adata)
        result = sq.gr.spatial_autocorr(adata, mode="moran", copy=True, device="gpu")

        assert result is not None
        assert "I" in result.columns
        assert "pval_norm" in result.columns

    def test_spatial_autocorr_gpu_vs_cpu(self, adata):
        """Test that GPU and CPU results are approximately equal."""
        sq.gr.spatial_neighbors(adata)
        cpu_result = sq.gr.spatial_autocorr(adata, mode="moran", copy=True, device="cpu")
        gpu_result = sq.gr.spatial_autocorr(adata, mode="moran", copy=True, device="gpu")

        np.testing.assert_allclose(cpu_result["I"].values, gpu_result["I"].values, rtol=1e-3, equal_nan=True)


class TestGPULigrec:
    """Test GPU-accelerated ligrec function."""

    def test_ligrec_gpu(self, adata):
        """Test ligrec with GPU device."""
        result = sq.gr.ligrec(adata, cluster_key="leiden", copy=True, device="gpu")

        assert result is not None
        assert "means" in result
        assert "pvalues" in result
