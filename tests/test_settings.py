"""Tests for squidpy._settings module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squidpy._settings import gpu_dispatch, settings
from squidpy._settings._dispatch import _GPU_FUNC_CACHE


@pytest.fixture(autouse=True)
def clear_gpu_cache():
    """Clear GPU function cache before each test."""
    _GPU_FUNC_CACHE.clear()
    yield
    _GPU_FUNC_CACHE.clear()


class TestSettings:
    """Test the settings module."""

    def test_default_device(self):
        """Test that default device is 'auto'."""
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
        if not settings.gpu_available:
            with pytest.raises(RuntimeError, match="GPU unavailable"):
                settings.device = "gpu"


class TestGpuDispatch:
    """Test the gpu_dispatch decorator."""

    def test_cpu_path(self):
        """Test CPU device calls original function."""
        calls = []

        @gpu_dispatch()
        def my_func(x, y, *, n_jobs=1):
            calls.append((x, y, n_jobs))
            return x + y

        settings.device = "cpu"
        assert my_func(1, 2) == 3
        assert calls == [(1, 2, 1)]
        settings.device = "auto"  # reset

    def test_auto_device_falls_back_to_cpu(self):
        """Test auto device falls back to CPU when GPU unavailable."""
        if settings.gpu_available:
            pytest.skip("GPU is available")

        calls = []

        @gpu_dispatch()
        def my_func(x):
            calls.append(x)
            return x * 2

        settings.device = "auto"
        assert my_func(5) == 10
        assert calls == [5]

    def test_gpu_path(self):
        """Test GPU device dispatches to GPU module."""
        mock_module = MagicMock()

        def gpu_my_func(x):
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._get_effective_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
        ):
            assert my_func(42) == "gpu_result"

    def test_device_kwargs_passed_to_gpu(self):
        """Test device_kwargs are merged and passed to GPU function."""
        mock_module = MagicMock()
        received_kwargs = {}

        def gpu_my_func(x, use_sparse=False):
            received_kwargs.update({"x": x, "use_sparse": use_sparse})
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x, device_kwargs=None):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._get_effective_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
        ):
            result = my_func(42, device_kwargs={"use_sparse": True})
            assert result == "gpu_result"
            assert received_kwargs == {"x": 42, "use_sparse": True}

    def test_device_kwargs_ignored_on_cpu(self):
        """Test device_kwargs are stripped on CPU path."""
        calls = []

        @gpu_dispatch()
        def my_func(x, device_kwargs=None):
            calls.append(x)
            return x * 2

        settings.device = "cpu"
        # device_kwargs should be stripped, not cause an error
        assert my_func(5, device_kwargs={"use_sparse": True}) == 10
        assert calls == [5]
        settings.device = "auto"  # reset

    def test_validate_args_on_gpu(self):
        """Test validate_args runs validators before GPU dispatch."""
        mock_module = MagicMock()
        mock_module.my_func = MagicMock(return_value="gpu_result")

        def validate_attr(value):
            if value != "X":
                raise ValueError(f"attr={value!r} not supported on GPU")

        @gpu_dispatch(gpu_module="test_module", validate_args={"attr": validate_attr})
        def my_func(x, attr="X"):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._get_effective_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
        ):
            # Valid value should work
            assert my_func(42, attr="X") == "gpu_result"

            # Invalid value should raise
            with pytest.raises(ValueError, match="attr='obs' not supported on GPU"):
                my_func(42, attr="obs")

    def test_preserves_function_metadata(self):
        """Test decorator preserves function name and injects GPU note."""

        @gpu_dispatch()
        def documented_func(x):
            """Original docstring.

            Parameters
            ----------
            x
                Input value.
            """
            return x

        assert documented_func.__name__ == "documented_func"
        assert "Original docstring." in documented_func.__doc__
        assert "GPU acceleration" in documented_func.__doc__

    def test_gpu_errors_when_unavailable(self):
        """Test GPU raises error when unavailable."""
        if settings.gpu_available:
            pytest.skip("GPU is available")

        # Should raise error when GPU requested but unavailable
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            settings.device = "gpu"

    def test_docstring_uses_custom_gpu_module(self):
        """Test that docstring GPU note uses the specified gpu_module."""

        @gpu_dispatch(gpu_module="custom.module.path")
        def my_func(x):
            """My function.

            Parameters
            ----------
            x
                Input.
            """
            return x

        assert "custom.module.path.my_func" in my_func.__doc__

