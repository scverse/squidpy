"""Tests for squidpy.settings module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squidpy._utils import gpu_dispatch
from squidpy.settings import settings


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
        with pytest.raises(ValueError, match="Invalid device"):
            settings.device = "invalid"

    def test_set_device_gpu_without_rsc(self):
        """Test that setting device to 'gpu' without rapids-singlecell raises RuntimeError."""
        # This will fail if rapids-singlecell is not installed
        if not settings.gpu_available():
            with pytest.raises(RuntimeError, match="rapids-singlecell not installed"):
                settings.device = "gpu"


class TestGpuDispatch:
    """Test the gpu_dispatch decorator."""

    def test_cpu_path_calls_original(self):
        """Test that CPU device calls the original function."""
        original_called = []

        @gpu_dispatch("fake_rapids_module")
        def my_func(x, y, *, n_jobs=1, device=None):
            original_called.append((x, y, n_jobs))
            return x + y

        result = my_func(1, 2, device="cpu")
        assert result == 3
        assert original_called == [(1, 2, 1)]

    def test_cpu_path_with_auto_device_no_gpu(self):
        """Test that auto device falls back to CPU when GPU unavailable."""
        original_called = []

        @gpu_dispatch("fake_rapids_module")
        def my_func(x, device=None):
            original_called.append(x)
            return x * 2

        # With auto and no GPU available, should call original
        if not settings.gpu_available():
            result = my_func(5, device="auto")
            assert result == 10
            assert original_called == [5]

    def test_gpu_path_filters_parameters(self):
        """Test that GPU dispatch filters out parameters not in rapids signature."""
        mock_rapids_func = MagicMock(return_value="gpu_result")

        # Create a mock module
        mock_module = MagicMock()
        mock_module.my_func = mock_rapids_func

        @gpu_dispatch("mock_rapids")
        def my_func(adata, cluster_key, *, n_jobs=1, backend="loky", device=None):
            return "cpu_result"

        with patch("importlib.import_module", return_value=mock_module):
            with patch("squidpy._utils.resolve_device_arg", return_value="gpu"):
                # Mock the rapids function signature to only accept adata and cluster_key
                import inspect

                mock_sig = inspect.signature(lambda adata, cluster_key: None)
                with patch(
                    "inspect.signature",
                    side_effect=lambda f: mock_sig if f == mock_rapids_func else inspect.signature(f),
                ):
                    result = my_func("adata_obj", "leiden", n_jobs=4, backend="threading", device="gpu")

        assert result == "gpu_result"
        # Should only be called with adata and cluster_key, not n_jobs or backend
        mock_rapids_func.assert_called_once_with(adata="adata_obj", cluster_key="leiden")

    def test_preserves_function_metadata(self):
        """Test that the decorator preserves function name and docstring."""

        @gpu_dispatch("fake_module")
        def documented_func(x, device=None):
            """This is the docstring."""
            return x

        assert documented_func.__name__ == "documented_func"
        assert documented_func.__doc__ == """This is the docstring."""

    def test_custom_rapids_func_name(self):
        """Test using a custom rapids function name."""
        mock_rapids_func = MagicMock(return_value="rapids_result")
        mock_module = MagicMock()
        mock_module.different_name = mock_rapids_func

        @gpu_dispatch("mock_rapids", rapids_func_name="different_name")
        def my_func(x, device=None):
            return "cpu_result"

        with patch("importlib.import_module", return_value=mock_module):
            with patch("squidpy._utils.resolve_device_arg", return_value="gpu"):
                import inspect

                mock_sig = inspect.signature(lambda x: None)
                with patch(
                    "inspect.signature",
                    side_effect=lambda f: mock_sig if f == mock_rapids_func else inspect.signature(f),
                ):
                    result = my_func(42, device="gpu")

        assert result == "rapids_result"
        mock_rapids_func.assert_called_once_with(x=42)
