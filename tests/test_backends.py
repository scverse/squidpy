"""Tests for the backend dispatch system."""

from __future__ import annotations

import warnings

import pytest

from squidpy._backends import _registry, dispatch, settings
from squidpy._backends._dispatch import _sig_cache
from squidpy._backends._registry import (
    _TRUSTED_ALIASES,
    TRUSTED_BACKENDS,
    _alias_map,
    _backends,
)


class FakeBackend:
    name = "fake_gpu"
    aliases = ["fake", "test-gpu"]

    def my_func(self, x, gpu_param=None):
        """Run my_func on GPU.

        Parameters
        ----------
        x
            Input value.
        gpu_param
            GPU-specific parameter.
        """
        return f"gpu:{x}:{gpu_param}"


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset all backend state between tests."""
    _backends.clear()
    _alias_map.clear()
    _sig_cache.clear()
    old_discovered = _registry._discovered
    # Mark as discovered to prevent entrypoint loading during tests —
    # tests register their own fake backends explicitly.
    _registry._discovered = True
    settings.backend = "cpu"
    # Temporarily add fake backend to trusted list
    old_trusted = TRUSTED_BACKENDS.copy()
    old_aliases = _TRUSTED_ALIASES.copy()
    TRUSTED_BACKENDS["fake_gpu"] = {
        "aliases": ["fake", "test-gpu"],
        "package": "fake-gpu-pkg",
    }
    _TRUSTED_ALIASES["fake_gpu"] = "fake_gpu"
    _TRUSTED_ALIASES["fake"] = "fake_gpu"
    _TRUSTED_ALIASES["test-gpu"] = "fake_gpu"
    yield
    _backends.clear()
    _alias_map.clear()
    _sig_cache.clear()
    _registry._discovered = old_discovered
    settings.backend = "cpu"
    TRUSTED_BACKENDS.clear()
    TRUSTED_BACKENDS.update(old_trusted)
    _TRUSTED_ALIASES.clear()
    _TRUSTED_ALIASES.update(old_aliases)


def _register_fake():
    backend = FakeBackend()
    _backends["fake_gpu"] = backend
    _alias_map["fake_gpu"] = "fake_gpu"
    _alias_map["fake"] = "fake_gpu"
    _alias_map["test-gpu"] = "fake_gpu"
    return backend


class TestSettings:
    def test_default_is_cpu(self):
        assert settings.backend == "cpu"

    def test_set_unknown_backend_raises_with_suggestion(self):
        _register_fake()
        with pytest.raises(ValueError, match="Unknown backend.*Did you mean"):
            settings.backend = "fak"  # close to "fake"

    def test_set_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            settings.backend = "nonexistent"

    def test_trusted_but_not_installed_raises(self):
        # fake_gpu is trusted but not registered (not installed)
        with pytest.raises(ImportError, match="not installed"):
            settings.backend = "fake"

    def test_context_manager_restores(self):
        _register_fake()

        settings.backend = "fake"
        with settings.use_backend("cpu"):
            assert settings.backend == "cpu"
        assert settings.backend == "fake_gpu"

    def test_set_via_alias(self):
        _register_fake()
        settings.backend = "fake"
        # Aliases resolve to canonical name
        assert settings.backend == "fake_gpu"

    def test_set_via_canonical(self):
        _register_fake()
        settings.backend = "fake_gpu"
        assert settings.backend == "fake_gpu"

    def test_untrusted_backend_warns(self):
        # Remove fake_gpu from trusted list
        TRUSTED_BACKENDS.pop("fake_gpu", None)
        _TRUSTED_ALIASES.pop("fake_gpu", None)
        _TRUSTED_ALIASES.pop("fake", None)
        _TRUSTED_ALIASES.pop("test-gpu", None)

        _register_fake()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            settings.backend = "fake"
            assert len(w) == 1
            assert "not in squidpy's trusted backends list" in str(w[0].message)

    def test_available_backends_empty(self):
        assert settings.available_backends() == []

    def test_available_backends_with_registered(self):
        _register_fake()
        assert "fake_gpu" in settings.available_backends()

    def test_get_backend_returns_instance(self):
        backend = _register_fake()
        result = settings.get_backend("fake_gpu")
        assert result is backend

    def test_get_backend_by_alias(self):
        backend = _register_fake()
        assert settings.get_backend("fake") is backend
        assert settings.get_backend("test-gpu") is backend

    def test_get_backend_unknown_returns_none(self):
        assert settings.get_backend("nonexistent") is None

    def test_get_backend_cpu_returns_none(self):
        assert settings.get_backend("cpu") is None


class TestDispatch:
    def test_cpu_path(self):
        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}:{n_jobs}"

        assert my_func(42) == "cpu:42:None"
        assert my_func(42, n_jobs=4) == "cpu:42:4"

    def test_gpu_dispatch(self):
        _register_fake()

        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with settings.use_backend("fake"):
            assert my_func(42) == "gpu:42:None"

    def test_gpu_specific_kwarg(self):
        _register_fake()

        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with settings.use_backend("fake"):
            assert my_func(42, gpu_param="hello") == "gpu:42:hello"

    def test_cpu_only_kwarg_warns(self):
        _register_fake()

        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with settings.use_backend("fake"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                my_func(42, n_jobs=4)
                assert len(w) == 1
                assert "n_jobs" in str(w[0].message)

    def test_cpu_only_kwarg_default_silent(self):
        _register_fake()

        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with settings.use_backend("fake"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                my_func(42, n_jobs=None)
                assert len(w) == 0

    def test_gpu_kwarg_on_cpu_raises(self):
        _register_fake()

        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with pytest.raises(TypeError, match="gpu_param"):
            my_func(42, gpu_param="hello")

    def test_fallback_when_not_implemented(self):
        _register_fake()

        @dispatch
        def other_func(x):
            return f"cpu:{x}"

        with settings.use_backend("fake"):
            # FakeBackend doesn't have other_func -> CPU fallback
            assert other_func(42) == "cpu:42"

    def test_per_function_override(self):
        _register_fake()

        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        settings.backend = "fake"
        assert my_func(42, backend="cpu") == "cpu:42"

    def test_alias_resolution(self):
        _register_fake()

        @dispatch
        def my_func(x, n_jobs=None):
            return f"cpu:{x}"

        with settings.use_backend("fake_gpu"):
            assert my_func(42) == "gpu:42:None"
        with settings.use_backend("fake"):
            assert my_func(42) == "gpu:42:None"
        with settings.use_backend("test-gpu"):
            assert my_func(42) == "gpu:42:None"

    def test_backend_not_installed_raises(self):
        _register_fake()

        @dispatch
        def my_func(x):
            return f"cpu:{x}"

        with pytest.raises(RuntimeError, match="not installed"):
            my_func(42, backend="nonexistent_backend")


class TestDocstringMerging:
    """Test numpydoc parameter extraction and injection."""

    def test_extract_param_docs(self):
        from squidpy._backends._dispatch import _extract_param_docs

        docstring = """\
Run something.

Parameters
----------
x
    Input value.
gpu_param
    GPU-specific parameter.

Returns
-------
Result.
"""
        result = _extract_param_docs(docstring, {"gpu_param"})
        assert "gpu_param" in result
        assert "GPU-specific" in result["gpu_param"]

    def test_extract_skips_missing_params(self):
        from squidpy._backends._dispatch import _extract_param_docs

        docstring = """\
Parameters
----------
x
    Input.
"""
        result = _extract_param_docs(docstring, {"nonexistent"})
        assert result == {}

    def test_extract_no_params_section(self):
        from squidpy._backends._dispatch import _extract_param_docs

        result = _extract_param_docs("Just a docstring.", {"x"})
        assert result == {}

    def test_inject_param_docs(self):
        from squidpy._backends._dispatch import _inject_param_docs

        docstring = """\
Do something.

Parameters
----------
x
    Input.

Returns
-------
Result.
"""
        result = _inject_param_docs(docstring, {"gpu_param": "gpu_param\n    A GPU param."})
        assert "gpu_param" in result
        assert "backend" in result
        # backend doc should appear before Returns
        lines = result.split("\n")
        backend_idx = next(i for i, l in enumerate(lines) if "backend" in l.lower() and "Backend to use" not in l)
        returns_idx = next(i for i, l in enumerate(lines) if l.strip() == "Returns")
        assert backend_idx < returns_idx

    def test_inject_no_params_section_unchanged(self):
        from squidpy._backends._dispatch import _inject_param_docs

        docstring = "Just a plain docstring."
        assert _inject_param_docs(docstring, {"x": "x\n    Param."}) == docstring

    def test_extract_handles_multiline_descriptions(self):
        from squidpy._backends._dispatch import _extract_param_docs

        docstring = """\
Parameters
----------
multi
    First line of description.
    Second line continues here
    with more detail.
other
    Another param.
"""
        result = _extract_param_docs(docstring, {"multi"})
        assert "multi" in result
        assert "Second line" in result["multi"]
        assert "with more detail" in result["multi"]


class TestLazyDiscovery:
    """Test that backend discovery is lazy."""

    def test_discovery_not_triggered_on_import(self):
        """Verify _discovered stays False until a backend function is used."""
        _registry._discovered = False
        _backends.clear()
        _alias_map.clear()

        assert not _registry._discovered

    def test_discovery_triggered_by_get_backend(self):
        """get_backend triggers lazy discovery."""
        _registry._discovered = False
        from squidpy._backends._registry import get_backend

        get_backend("cpu")
        assert _registry._discovered

    def test_discovery_triggered_by_settings_setter(self):
        """Setting backend triggers lazy discovery."""
        _registry._discovered = False
        _register_fake()
        _registry._discovered = False
        settings.backend = "fake"
        assert _registry._discovered
