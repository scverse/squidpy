"""Tests for squidpy._settings module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from squidpy._settings import gpu_dispatch, settings
from squidpy.gr._gpu import SPECIAL_PARAM_REGISTRY, GpuParamSpec


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
        def my_func(x, y, *, n_jobs=1, device=None):
            calls.append((x, y, n_jobs))
            return x + y

        assert my_func(1, 2, device="cpu") == 3
        assert calls == [(1, 2, 1)]

    def test_auto_device_falls_back_to_cpu(self):
        """Test auto device falls back to CPU when GPU unavailable."""
        if settings.gpu_available:
            pytest.skip("GPU is available")

        calls = []

        @gpu_dispatch()
        def my_func(x, device=None):
            calls.append(x)
            return x * 2

        assert my_func(5, device="auto") == 10
        assert calls == [5]

    def test_gpu_path(self):
        """Test GPU device dispatches to GPU module."""
        mock_module = MagicMock()

        # Must use real function for signature introspection
        def gpu_my_func(x):
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x, device=None):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
        ):
            assert my_func(42, device="gpu") == "gpu_result"

    def test_custom_gpu_func_name(self):
        """Test custom GPU function name."""
        mock_module = MagicMock()

        # Must use real function for signature introspection
        def custom_name(x):
            return "gpu_result"

        mock_module.custom_name = custom_name

        @gpu_dispatch(gpu_module="test_module", gpu_func_name="custom_name")
        def my_func(x, device=None):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
        ):
            assert my_func(42, device="gpu") == "gpu_result"

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

    def test_cpu_only_params_error_on_gpu_if_provided(self):
        """Test CPU-only params raise error on GPU if user explicitly provided them."""
        mock_module = MagicMock()

        # GPU function without n_jobs param (CPU-only)
        def gpu_my_func(x):
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x, n_jobs=1, device=None):
            return "cpu_result"

        with (
            patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
            patch("importlib.import_module", return_value=mock_module),
        ):
            # Not provided - should work
            assert my_func(42, device="gpu") == "gpu_result"

            # Explicitly provided (even if same as default) - should error
            with pytest.raises(ValueError, match="n_jobs.*only supported on CPU"):
                my_func(42, n_jobs=1, device="gpu")

            # Explicitly provided with different value - should also error
            with pytest.raises(ValueError, match="n_jobs.*only supported on CPU"):
                my_func(42, n_jobs=4, device="gpu")

    def test_gpu_only_params_error_on_cpu_if_provided(self):
        """Test GPU-only params raise error on CPU if user explicitly provided them.

        GPU-only params are those in GPU signature but NOT in CPU signature.
        If user tries to pass a GPU-only param on CPU, Python raises TypeError
        (unexpected keyword argument) unless the CPU func accepts **kwargs.
        """
        mock_module = MagicMock()

        # GPU func has gpu_batch_size (GPU-only, not in CPU sig)
        def gpu_my_func(x, gpu_batch_size=1000):
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        # CPU func does NOT have gpu_batch_size
        @gpu_dispatch(gpu_module="test_module")
        def my_func(x, device=None):
            return "cpu_result"

        with patch("importlib.import_module", return_value=mock_module):
            # Not provided - should work
            assert my_func(42, device="cpu") == "cpu_result"

            # GPU-only param on CPU - Python raises TypeError (not in signature)
            with pytest.raises(TypeError, match="unexpected keyword argument"):
                my_func(42, gpu_batch_size=500, device="cpu")

    def test_function_with_no_exclusive_params(self):
        """Test that functions with matching signatures work transparently."""
        calls = []
        mock_module = MagicMock()

        # GPU func has same signature
        def gpu_func(x):
            return "gpu_result"

        mock_module.my_func = gpu_func

        @gpu_dispatch(gpu_module="test_module")
        def my_func(x, device=None):
            calls.append(x)
            return x * 3

        # Should work on CPU without issues
        assert my_func(10, device="cpu") == 30
        assert calls == [10]

    def test_gpu_errors_when_unavailable(self):
        """Test GPU raises error when unavailable."""
        if settings.gpu_available:
            pytest.skip("GPU is available")

        @gpu_dispatch()
        def my_func(x, device=None):
            return x + 1

        # Should raise error when GPU requested but unavailable
        with pytest.raises(RuntimeError, match="GPU unavailable"):
            my_func(5, device="gpu")

    def test_custom_validator_error(self):
        """Test custom validator raises appropriate error."""
        mock_module = MagicMock()

        def my_validator(value):
            if value != "allowed":
                return f"value={value!r} is not allowed on GPU"
            return None

        # GPU func without custom_param (CPU-only with validator)
        def gpu_my_func(x):
            return "gpu_result"

        mock_module.my_func = gpu_my_func

        registry = {
            "my_func": {
                "cpu_only": {"custom_param": GpuParamSpec(validate_fn=my_validator)},
                "gpu_only": {},
            }
        }

        with patch.dict(SPECIAL_PARAM_REGISTRY, registry):

            @gpu_dispatch(gpu_module="test_module")
            def my_func(x, custom_param="allowed", device=None):
                return "cpu_result"

            with (
                patch("squidpy._settings._dispatch._resolve_device", return_value="gpu"),
                patch("importlib.import_module", return_value=mock_module),
            ):
                # Allowed value - should work
                assert my_func(42, custom_param="allowed", device="gpu") == "gpu_result"

                # Not allowed value - should error
                with pytest.raises(ValueError, match="value='bad' is not allowed on GPU"):
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
