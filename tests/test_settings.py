"""Tests for squidpy.settings module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squidpy.settings import gpu_dispatch, settings


class TestSettings:
    """Test the settings module."""

    def test_default_device(self):
        """Test that default device is 'auto'."""
        # Reset to default
        settings.device = "auto"
        assert settings.device == "auto"

    def test_set_device_cpu(self):
        """Test setting device to 'cpu'."""
        settings.device = "cpu"
        assert settings.device == "cpu"
        settings.device = "auto"  # reset

    def test_set_device_invalid(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            settings.device = "invalid"

    def test_set_device_gpu_without_rsc(self):
        """Test that setting device to 'gpu' without rapids-singlecell raises RuntimeError."""
        # This will fail if rapids-singlecell is not installed
        if not settings.gpu_available():
            with pytest.raises(RuntimeError, match="GPU unavailable"):
                settings.device = "gpu"

class TestGpuDispatch:
    """Test the gpu_dispatch decorator."""

    def test_cpu_path_calls_original(self):
        """Test that CPU device calls the original function."""
        original_called = []

        @gpu_dispatch()
        def my_func(x, y, *, n_jobs=1, device=None):
            original_called.append((x, y, n_jobs))
            return x + y

        result = my_func(1, 2, device="cpu")
        assert result == 3
        assert original_called == [(1, 2, 1)]

    def test_cpu_path_with_auto_device_no_gpu(self):
        """Test that auto device falls back to CPU when GPU unavailable."""
        original_called = []

        @gpu_dispatch()
        def my_func(x, device=None):
            original_called.append(x)
            return x * 2

        # With auto and no GPU available, should call original
        if not settings.gpu_available():
            result = my_func(5, device="auto")
            assert result == 10
            assert original_called == [5]

    def test_gpu_path_calls_adapter(self):
        """Test that GPU dispatch calls the adapter function from _gpu module."""
        mock_adapter = MagicMock(return_value="gpu_result")

        @gpu_dispatch()
        def my_func(adata, cluster_key, *, n_jobs=1, backend="loky", device=None):
            return "cpu_result"

        with patch("squidpy.settings._dispatch._resolve_device", return_value="gpu"):
            with patch("squidpy.gr._gpu.my_func_gpu", mock_adapter, create=True):
                result = my_func("adata_obj", "leiden", n_jobs=4, backend="threading", device="gpu")

        assert result == "gpu_result"
        # Adapter receives all args except device
        mock_adapter.assert_called_once_with(
            adata="adata_obj", cluster_key="leiden", n_jobs=4, backend="threading"
        )

    def test_preserves_function_metadata(self):
        """Test that the decorator preserves function name and docstring."""

        @gpu_dispatch()
        def documented_func(x, device=None):
            """This is the docstring."""
            return x

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == """This is the docstring."""

    def test_custom_gpu_func_name(self):
        """Test using a custom GPU adapter function name."""
        mock_adapter = MagicMock(return_value="gpu_result")

        @gpu_dispatch("custom_adapter_name")
        def my_func(x, device=None):
            return "cpu_result"

        with patch("squidpy.settings._dispatch._resolve_device", return_value="gpu"):
            with patch("squidpy.gr._gpu.custom_adapter_name", mock_adapter, create=True):
                result = my_func(42, device="gpu")

        assert result == "gpu_result"
        mock_adapter.assert_called_once_with(x=42)
