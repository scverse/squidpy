"""Tests for GPU functionality (skipped in CI without GPU)."""

from __future__ import annotations

import pytest

from squidpy.settings import settings

# Skip all tests in this module if GPU is not available
pytestmark = pytest.mark.skipif(
    not settings.gpu_available(),
    reason="GPU tests require rapids-singlecell to be installed",
)


class TestGPUCoOccurrence:
    """Test GPU-accelerated co_occurrence function."""

    def test_co_occurrence_gpu(self, adata):
        """Test co_occurrence with GPU device."""
        import squidpy as sq

        # Run with explicit GPU device
        result = sq.gr.co_occurrence(
            adata,
            cluster_key="leiden",
            copy=True,
            device="gpu",
        )

        assert result is not None
        arr, interval = result
        assert arr.ndim == 3
        assert arr.shape[1] == arr.shape[0] == adata.obs["leiden"].unique().shape[0]

    def test_co_occurrence_gpu_vs_cpu(self, adata):
        """Test that GPU and CPU results are approximately equal."""
        import numpy as np

        import squidpy as sq

        # Run on CPU
        cpu_result = sq.gr.co_occurrence(
            adata,
            cluster_key="leiden",
            copy=True,
            device="cpu",
        )

        # Run on GPU
        gpu_result = sq.gr.co_occurrence(
            adata,
            cluster_key="leiden",
            copy=True,
            device="gpu",
        )

        cpu_arr, cpu_interval = cpu_result
        gpu_arr, gpu_interval = gpu_result

        # Results should be close (allow for floating point differences)
        np.testing.assert_allclose(cpu_interval, gpu_interval, rtol=1e-5)
        np.testing.assert_allclose(cpu_arr, gpu_arr, rtol=1e-5)


class TestGPUSpatialAutocorr:
    """Test GPU-accelerated spatial_autocorr function."""

    def test_spatial_autocorr_gpu(self, adata):
        """Test spatial_autocorr with GPU device."""
        import squidpy as sq

        # Ensure spatial neighbors are computed
        sq.gr.spatial_neighbors(adata)

        # Run with explicit GPU device
        result = sq.gr.spatial_autocorr(
            adata,
            mode="moran",
            copy=True,
            device="gpu",
        )

        assert result is not None
        assert "I" in result.columns
        assert "pval_norm" in result.columns

    def test_spatial_autocorr_gpu_vs_cpu(self, adata):
        """Test that GPU and CPU results are approximately equal."""
        import numpy as np

        import squidpy as sq

        # Ensure spatial neighbors are computed
        sq.gr.spatial_neighbors(adata)

        # Run on CPU
        cpu_result = sq.gr.spatial_autocorr(
            adata,
            mode="moran",
            copy=True,
            device="cpu",
        )

        # Run on GPU
        gpu_result = sq.gr.spatial_autocorr(
            adata,
            mode="moran",
            copy=True,
            device="gpu",
        )

        # Results should be close (allow for floating point differences)
        # Use equal_nan=True since some genes may have NaN values, and relax rtol for float32/float64 differences
        np.testing.assert_allclose(cpu_result["I"].values, gpu_result["I"].values, rtol=1e-3, equal_nan=True)


class TestGPULigrec:
    """Test GPU-accelerated ligrec function."""

    def test_ligrec_gpu(self, adata):
        """Test ligrec with GPU device."""
        import squidpy as sq

        # Run with explicit GPU device
        result = sq.gr.ligrec(
            adata,
            cluster_key="leiden",
            copy=True,
            device="gpu",
        )

        assert result is not None
        assert "means" in result
        assert "pvalues" in result


class TestGPUSettingsOptIn:
    """Test settings-based GPU opt-in functionality."""

    def test_settings_device_gpu(self, adata):
        """Test that setting device='gpu' globally uses GPU for all functions."""
        import squidpy as sq
        from squidpy.settings import settings

        # Save original setting
        original_device = settings.device

        try:
            # Opt-in to GPU globally
            settings.device = "gpu"

            # Run without explicit device - should use GPU
            result = sq.gr.co_occurrence(
                adata,
                cluster_key="leiden",
                copy=True,
            )

            assert result is not None
        finally:
            # Restore original setting
            settings.device = original_device

    def test_settings_device_auto(self, adata):
        """Test that device='auto' uses GPU when available."""
        import squidpy as sq
        from squidpy.settings import settings

        # auto is the default, and should use GPU when available
        assert settings.device == "auto"

        # Run without explicit device - should automatically use GPU
        result = sq.gr.co_occurrence(
            adata,
            cluster_key="leiden",
            copy=True,
        )

        assert result is not None

    def test_explicit_device_overrides_settings(self, adata):
        """Test that explicit device parameter overrides global settings."""
        import squidpy as sq
        from squidpy.settings import settings

        # Save original setting
        original_device = settings.device

        try:
            # Set global to CPU
            settings.device = "cpu"

            # But explicitly request GPU - should use GPU
            result = sq.gr.co_occurrence(
                adata,
                cluster_key="leiden",
                copy=True,
                device="gpu",
            )

            assert result is not None
        finally:
            # Restore original setting
            settings.device = original_device
