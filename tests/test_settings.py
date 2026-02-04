"""Tests for squidpy._settings module."""

from __future__ import annotations

import concurrent.futures
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


class TestSettings:
    """Test the settings module."""

    def test_default_device_cpu_when_gpu_unavailable(self):
        """Test that default device is 'cpu' when GPU unavailable."""
        if settings.gpu_available:
            pytest.skip("GPU is available")
        assert settings.device == "cpu"

    def test_default_device_gpu_when_available(self):
        """Test that default device is 'gpu' when GPU available."""
        if not settings.gpu_available:
            pytest.skip("GPU is not available")
        assert settings.device == "gpu"

    def test_set_device_cpu(self):
        """Test setting device to 'cpu'."""
        settings.device = "cpu"
        assert settings.device == "cpu"

    def test_set_device_invalid(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="device must be one of"):
            settings.device = "invalid"


    def test_set_device_gpu_without_rsc(self):
        """Test that setting device to 'gpu' without rapids-singlecell raises RuntimeError."""
        if settings.gpu_available:
            pytest.skip("GPU is available")
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            settings.device = "gpu"


class TestUseDeviceContextManager:
    """Test the use_device context manager."""

    def test_use_device_temporarily_sets_cpu(self):
        """Test that use_device temporarily sets the device."""
        if settings.gpu_available:
            pytest.skip("GPU is available - can't test CPU default")

        original = settings.device
        with settings.use_device("cpu"):
            assert settings.device == "cpu"
        assert settings.device == original

    def test_use_device_restores_on_exception(self):
        """Test that use_device restores device even on exception."""
        original = settings.device
        with pytest.raises(ValueError, match="test error"):
            with settings.use_device("cpu"):
                assert settings.device == "cpu"
                raise ValueError("test error")
        assert settings.device == original

    def test_use_device_invalid_raises(self):
        """Test that use_device raises on invalid device."""
        with pytest.raises(ValueError, match="device must be one of"):
            with settings.use_device("invalid"):
                pass

    def test_use_device_gpu_without_rsc_raises(self):
        """Test that use_device('gpu') raises when GPU unavailable."""
        if settings.gpu_available:
            pytest.skip("GPU is available")
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            with settings.use_device("gpu"):
                pass

    def test_nested_use_device(self):
        """Test nested use_device contexts restore correctly."""
        if settings.gpu_available:
            pytest.skip("GPU is available")

        original = settings.device
        settings.device = "cpu"

        with settings.use_device("cpu"):
            assert settings.device == "cpu"
            with settings.use_device("cpu"):
                assert settings.device == "cpu"
            assert settings.device == "cpu"
        assert settings.device == "cpu"

    def test_use_device_thread_isolation(self):
        """Test that use_device is thread-safe with isolated contexts."""
        results = {}

        def thread_func(thread_id: int, device: str):
            with settings.use_device(device):
                # Small delay to increase chance of interleaving
                import time

                time.sleep(0.01)
                results[thread_id] = settings.device

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # Both threads use CPU since GPU may not be available
            f1 = executor.submit(thread_func, 1, "cpu")
            f2 = executor.submit(thread_func, 2, "cpu")
            f1.result()
            f2.result()

        assert results[1] == "cpu"
        assert results[2] == "cpu"


class TestGpuFuncCache:
    """Test the GPU function caching."""

    def test_cache_info_available(self):
        """Test that cache_info is accessible."""
        info = _get_gpu_func.cache_info()
        assert hasattr(info, "hits")
        assert hasattr(info, "misses")

    def test_cache_clear_works(self):
        """Test that cache_clear works."""
        _get_gpu_func.cache_clear()
        info = _get_gpu_func.cache_info()
        assert info.hits == 0
        assert info.misses == 0


class TestGpuDispatch:
    """Test the gpu_dispatch decorator."""

    def test_cpu_path(self):
        """Test CPU device calls original function."""
        calls = []

        @gpu_dispatch()
        def my_func(x, y, *, n_jobs=1):
            calls.append((x, y, n_jobs))
            return x + y

        with settings.use_device("cpu"):
            assert my_func(1, 2) == 3
            assert calls == [(1, 2, 1)]

    def test_gpu_path(self):
        """Test GPU device dispatches to GPU module."""
        mock_module = MagicMock()

        def gpu_my_func(x):
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x):
            return "cpu_result"

        with patch.object(settings, "gpu_available", True):
            with settings.use_device("gpu"):
                with patch("importlib.import_module", return_value=mock_module):
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

        with patch.object(settings, "gpu_available", True):
            with settings.use_device("gpu"):
                with patch("importlib.import_module", return_value=mock_module):
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

        with settings.use_device("cpu"):
            # device_kwargs should be stripped, not cause an error
            assert my_func(5, device_kwargs={"use_sparse": True}) == 10
            assert calls == [5]

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

        with patch.object(settings, "gpu_available", True):
            with settings.use_device("gpu"):
                with patch("importlib.import_module", return_value=mock_module):
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

    def test_gpu_import_error_propagates(self):
        """Test ImportError propagates when GPU module not found."""

        @gpu_dispatch(gpu_module="nonexistent_module")
        def my_func(x):
            return "cpu_result"

        with patch.object(settings, "gpu_available", True):
            with settings.use_device("gpu"):
                with pytest.raises(ImportError):
                    my_func(42)

    def test_gpu_attribute_error_propagates(self):
        """Test AttributeError propagates when function not in GPU module."""
        mock_module = MagicMock(spec=[])  # Empty spec, no attributes

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x):
            return "cpu_result"

        with patch.object(settings, "gpu_available", True):
            with settings.use_device("gpu"):
                with patch("importlib.import_module", return_value=mock_module):
                    with pytest.raises(AttributeError):
                        my_func(42)

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
