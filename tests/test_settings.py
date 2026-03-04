"""Tests for squidpy._settings module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squidpy._settings import gpu_dispatch, settings
from squidpy._settings._dispatch import _get_gpu_func
from squidpy._settings._settings import _device_var


@pytest.fixture(autouse=True)
def reset_device():
    """Reset device state before and after each test."""
    _device_var.set(None)
    _get_gpu_func.cache_clear()
    yield
    _device_var.set(None)
    _get_gpu_func.cache_clear()


class TestDeviceSettings:
    """Test device property and use_device context manager."""

    def test_invalid_device_raises(self):
        """Test invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            settings.device = "invalid"
        with pytest.raises(ValueError, match="device must be one of"):
            with settings.use_device("invalid"):
                pass

    @pytest.mark.skipif(settings.gpu_available, reason="GPU is available")
    def test_gpu_without_rsc_raises(self):
        """Test setting GPU without rapids-singlecell raises RuntimeError."""
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            settings.device = "gpu"
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            with settings.use_device("gpu"):
                pass


class TestGpuDispatch:
    """Test the gpu_dispatch decorator."""

    def test_cpu_path(self):
        """Test CPU device calls original function."""
        calls = []

        @gpu_dispatch()
        def my_func(x, y):
            calls.append((x, y))
            return x + y

        with settings.use_device("cpu"):
            assert my_func(1, 2) == 3
            assert calls == [(1, 2)]

    def test_gpu_dispatch_and_device_kwargs(self):
        """Test GPU dispatch with device_kwargs."""
        mock_module = MagicMock()
        received = {}

        def gpu_my_func(x, use_sparse=False):
            received.update({"x": x, "use_sparse": use_sparse})
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x, device_kwargs=None):
            return "cpu_result"

        with patch.object(settings, "gpu_available", True):
            with settings.use_device("gpu"):
                with patch("importlib.import_module", return_value=mock_module):
                    # Basic dispatch
                    assert my_func(42) == "gpu_result"
                    # With device_kwargs
                    assert my_func(42, device_kwargs={"use_sparse": True}) == "gpu_result"
                    assert received["use_sparse"] is True

    def test_device_kwargs_error_on_cpu(self):
        """Test device_kwargs raises error on CPU path."""

        @gpu_dispatch()
        def my_func(x, device_kwargs=None):
            return x * 2

        with settings.use_device("cpu"):
            with pytest.raises(ValueError, match="device_kwargs should not be provided"):
                my_func(5, device_kwargs={"use_sparse": True})

    def test_validate_args(self):
        """Test validate_args runs validators before GPU dispatch."""
        mock_module = MagicMock()
        mock_module.my_func = MagicMock(return_value="gpu_result")

        @gpu_dispatch(
            gpu_module="test_module",
            validate_args={
                "attr": lambda v: (_ for _ in ()).throw(ValueError(f"attr={v!r} invalid")) if v != "X" else None
            },
        )
        def my_func(x, attr="X"):
            return "cpu_result"

        with patch.object(settings, "gpu_available", True):
            with settings.use_device("gpu"):
                with patch("importlib.import_module", return_value=mock_module):
                    assert my_func(42, attr="X") == "gpu_result"
                    with pytest.raises(ValueError, match="attr='obs' invalid"):
                        my_func(42, attr="obs")

    def test_preserves_metadata_and_docstring(self):
        """Test decorator preserves function name and injects GPU note."""

        @gpu_dispatch(gpu_module="custom.module")
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
        assert "custom.module.documented_func" in documented_func.__doc__
