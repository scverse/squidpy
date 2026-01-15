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
