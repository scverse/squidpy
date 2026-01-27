"""Tests for squidpy._settings module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squidpy._settings import gpu_dispatch, settings
from squidpy.gr._gpu import GpuParamSpec


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
        if not settings.gpu_available():
            with pytest.raises(RuntimeError, match="GPU unavailable"):
                settings.device = "gpu"


class TestGpuDispatch:
    """Test the gpu_dispatch decorator."""

    @pytest.fixture
    def mock_gpu_module(self):
        """Create a mock GPU module with adapter function."""
        mock_adapter = MagicMock(return_value="gpu_result")
        mock_module = MagicMock()
        mock_module.my_func = mock_adapter
        return mock_module, mock_adapter

    def test_cpu_path(self):
        """Test CPU device calls original function."""
        calls = []

        @gpu_dispatch()
        def my_func(x, y, *, n_jobs=1, device=None):
            calls.append((x, y, n_jobs))
            return x + y

        assert my_func(1, 2, device="cpu") == 3
        assert calls == [(1, 2, 1)]

    def test_auto_device_falls_back_to_cpu(self):
        """Test auto device falls back to CPU when GPU unavailable."""
        if settings.gpu_available():
            pytest.skip("GPU is available")

        calls = []

        @gpu_dispatch()
        def my_func(x, device=None):
            calls.append(x)
            return x * 2

        assert my_func(5, device="auto") == 10
        assert calls == [5]

    def test_gpu_path(self, mock_gpu_module):
        """Test GPU device dispatches to GPU module."""
        mock_module, mock_adapter = mock_gpu_module

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x, device=None):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
            patch("squidpy.gr._gpu.GPU_PARAM_REGISTRY", {"my_func": {"cpu_only": {}, "gpu_only": {}}}),
        ):
            assert my_func(42, device="gpu") == "gpu_result"

        mock_adapter.assert_called_once_with(x=42)

    def test_custom_gpu_func_name(self, mock_gpu_module):
        """Test custom GPU function name."""
        mock_module, mock_adapter = mock_gpu_module
        mock_module.custom_name = mock_adapter

        @gpu_dispatch(gpu_module="test_module", gpu_func_name="custom_name")
        def my_func(x, device=None):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
            patch("squidpy.gr._gpu.GPU_PARAM_REGISTRY", {"my_func": {"cpu_only": {}, "gpu_only": {}}}),
        ):
            assert my_func(42, device="gpu") == "gpu_result"

        mock_adapter.assert_called_once_with(x=42)

    def test_preserves_function_metadata(self):
        """Test decorator preserves function name and injects GPU note."""

        @gpu_dispatch()
        def documented_func(x, device=None):
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

    def test_cpu_only_params_error_on_gpu_if_provided(self, mock_gpu_module):
        """Test CPU-only params raise error on GPU if user provided a value."""
        mock_module, mock_adapter = mock_gpu_module
        registry = {
            "my_func": {
                "cpu_only": {"n_jobs": GpuParamSpec(1)},
                "gpu_only": {},
            }
        }

        # Need to patch in both modules: dispatch for registry lookup, _gpu for check functions
        with (
            patch("squidpy._settings._dispatch.GPU_PARAM_REGISTRY", registry),
            patch("squidpy.gr._gpu.GPU_PARAM_REGISTRY", registry),
        ):

            @gpu_dispatch(gpu_module="test_module")
            def my_func(x, n_jobs=None, device=None):
                return "cpu_result"

            # Not provided (None) - should work
            with (
                patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
                patch("importlib.import_module", return_value=mock_module),
            ):
                my_func(42, device="gpu")
                mock_adapter.assert_called_once_with(x=42)

            # Provided a value - should error
            with pytest.raises(ValueError, match="n_jobs.*only supported on CPU"):
                with (
                    patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
                    patch("importlib.import_module", return_value=mock_module),
                ):
                    my_func(42, n_jobs=4, device="gpu")

    def test_gpu_only_params_error_on_cpu_if_provided(self):
        """Test GPU-only params raise error on CPU if user provided a value."""
        registry = {
            "my_func": {
                "cpu_only": {},
                "gpu_only": {"use_sparse": GpuParamSpec(True)},
            }
        }

        # Need to patch in both modules: dispatch for registry lookup, _gpu for check functions
        with (
            patch("squidpy._settings._dispatch.GPU_PARAM_REGISTRY", registry),
            patch("squidpy.gr._gpu.GPU_PARAM_REGISTRY", registry),
        ):

            @gpu_dispatch(gpu_module="test_module")
            def my_func(x, use_sparse=None, device=None):
                return "cpu_result"

            # Not provided (None) - should work
            assert my_func(42, device="cpu") == "cpu_result"

            # Provided a value - should error
            with pytest.raises(ValueError, match="use_sparse.*only supported on GPU"):
                my_func(42, use_sparse=True, device="cpu")

    def test_function_not_in_registry_works(self):
        """Test that functions not in registry work transparently."""
        calls = []

        @gpu_dispatch()
        def unregistered_func(x, device=None):
            calls.append(x)
            return x * 3

        # Should work on CPU without issues
        assert unregistered_func(10, device="cpu") == 30
        assert calls == [10]

    def test_gpu_errors_when_unavailable(self):
        """Test GPU raises error when unavailable."""
        if settings.gpu_available():
            pytest.skip("GPU is available")

        @gpu_dispatch()
        def my_func(x, device=None):
            return x + 1

        # Should raise error when GPU requested but unavailable
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            my_func(5, device="gpu")

    def test_custom_validator_error(self, mock_gpu_module):
        """Test custom validator raises appropriate error."""
        mock_module, mock_adapter = mock_gpu_module

        def my_validator(value):
            if value != "allowed":
                return f"value={value!r} is not allowed on GPU"
            return None

        registry = {
            "my_func": {
                "cpu_only": {"custom_param": GpuParamSpec("allowed", validator=my_validator)},
                "gpu_only": {},
            }
        }

        # Need to patch in both modules: dispatch for registry lookup, _gpu for check functions
        with (
            patch("squidpy._settings._dispatch.GPU_PARAM_REGISTRY", registry),
            patch("squidpy.gr._gpu.GPU_PARAM_REGISTRY", registry),
        ):

            @gpu_dispatch(gpu_module="test_module")
            def my_func(x, custom_param=None, device=None):
                return "cpu_result"

            # Allowed value - should work
            with (
                patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
                patch("importlib.import_module", return_value=mock_module),
            ):
                my_func(42, custom_param="allowed", device="gpu")

            # Not allowed value - should error
            with pytest.raises(ValueError, match="value='bad' is not allowed on GPU"):
                with (
                    patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
                    patch("importlib.import_module", return_value=mock_module),
                ):
                    my_func(42, custom_param="bad", device="gpu")

    def test_docstring_uses_custom_gpu_module(self):
        """Test that docstring GPU note uses the specified gpu_module."""

        @gpu_dispatch(gpu_module="custom.module.path")
        def my_func(x, device=None):
            """My function.

            Parameters
            ----------
            x
                Input.
            """
            return x

        assert "custom.module.path.my_func" in my_func.__doc__

    def test_docstring_uses_custom_gpu_func_name(self):
        """Test that docstring GPU note uses the specified gpu_func_name."""

        @gpu_dispatch(gpu_module="some.module", gpu_func_name="different_name")
        def my_func(x, device=None):
            """My function.

            Parameters
            ----------
            x
                Input.
            """
            return x

        assert "some.module.different_name" in my_func.__doc__
