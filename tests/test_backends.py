"""Integration tests for squidpy's scverse-backends configuration."""

from __future__ import annotations

import inspect

import pytest

import squidpy as sq
from squidpy._backends import _dispatcher, backend_dispatch, get_backend, settings
from squidpy.testing.backend_conformance import validate_backend


class FakeRapidsBackend:
    name = "rapids_singlecell"
    aliases = ["rapids-singlecell", "rsc", "cuda"]

    def my_func(self, x, backend_param=None):
        """Run my_func on a backend.

        Parameters
        ----------
        x
            Input value.
        backend_param
            Backend-specific parameter.
        """
        return f"backend:{x}:{backend_param}"

    def spatial_autocorr(self, adata, mode="moran", copy=False):
        return sq.gr.spatial_autocorr(adata, mode=mode, copy=copy, backend="cpu")

    def co_occurrence(self, adata, cluster_key, copy=False):
        return sq.gr.co_occurrence(adata, cluster_key=cluster_key, copy=copy, backend="cpu")

    def nhood_enrichment(
        self,
        adata,
        cluster_key,
        copy=False,
        n_jobs=None,
        n_perms=1000,
        seed=None,
        show_progress_bar=True,
    ):
        return sq.gr.nhood_enrichment(
            adata,
            cluster_key=cluster_key,
            backend="cpu",
            copy=copy,
            n_jobs=n_jobs,
            n_perms=n_perms,
            seed=seed,
            show_progress_bar=show_progress_bar,
        )


@pytest.fixture(autouse=True)
def _reset_backend_state():
    registry = _dispatcher._registry
    dispatch_impl = _dispatcher._dispatch_impl

    old_discovered = registry._discovered
    old_backends = registry._backends.copy()
    old_alias_map = registry._alias_map.copy()
    old_load_errors = registry._load_errors.copy()
    old_registration_errors = registry._registration_errors.copy()
    old_warned_untrusted = registry._warned_untrusted.copy()
    old_sig_cache = dispatch_impl._sig_cache.copy()
    old_dispatched_functions = list(dispatch_impl._dispatched_functions)
    backend_token = settings._backend_var.set("cpu")

    registry._discovered = True
    registry._backends.clear()
    registry._alias_map.clear()
    registry._load_errors.clear()
    registry._registration_errors.clear()
    registry._warned_untrusted.clear()
    dispatch_impl._sig_cache.clear()
    dispatch_impl._dispatched_functions = []

    yield

    registry._discovered = old_discovered
    registry._backends.clear()
    registry._backends.update(old_backends)
    registry._alias_map.clear()
    registry._alias_map.update(old_alias_map)
    registry._load_errors.clear()
    registry._load_errors.update(old_load_errors)
    registry._registration_errors.clear()
    registry._registration_errors.update(old_registration_errors)
    registry._warned_untrusted.clear()
    registry._warned_untrusted.update(old_warned_untrusted)
    dispatch_impl._sig_cache.clear()
    dispatch_impl._sig_cache.update(old_sig_cache)
    dispatch_impl._dispatched_functions = old_dispatched_functions
    settings._backend_var.reset(backend_token)


def _register_fake_rsc() -> FakeRapidsBackend:
    backend = FakeRapidsBackend()
    registry = _dispatcher._registry
    registry._backends[backend.name] = backend
    registry._alias_map[backend.name] = backend.name
    for alias in backend.aliases:
        registry._alias_map[alias] = backend.name
    return backend


def test_trusted_rsc_policy_uses_concrete_cuda_alias():
    trusted = _dispatcher._registry.trusted_backends["rapids_singlecell"]

    assert trusted["aliases"] == ["rapids-singlecell", "rsc", "cuda"]
    assert "gpu" not in trusted["aliases"]
    assert _dispatcher._registry.reserved_backends == {"gpu": "Use a concrete backend alias such as 'cuda' or 'rsc'."}


def test_gpu_alias_is_reserved():
    with pytest.raises(ValueError, match="Use a concrete backend alias"):
        settings.backend = "gpu"


def test_trusted_rsc_not_installed_has_install_hint():
    with pytest.raises(ImportError, match="pip install rapids-singlecell"):
        settings.backend = "rsc"


def test_get_backend_resolves_registered_rsc_aliases():
    backend = _register_fake_rsc()

    assert get_backend("rapids_singlecell") is backend
    assert get_backend("rapids-singlecell") is backend
    assert get_backend("rsc") is backend
    assert get_backend("cuda") is backend
    assert get_backend("cpu") is None


def test_backend_conformance_accepts_matching_backend():
    _register_fake_rsc()

    assert validate_backend("cuda") == {
        "spatial_autocorr": "PASSED",
        "co_occurrence": "PASSED",
        "nhood_enrichment": "PASSED",
    }


def test_dispatch_routes_to_registered_rsc_alias():
    _register_fake_rsc()

    @backend_dispatch
    def my_func(x, n_jobs=None):
        return f"cpu:{x}:{n_jobs}"

    with settings.use_backend("cuda"):
        assert my_func(42, backend_param="value") == "backend:42:value"


def test_backend_specific_params_merge_into_signature_and_docstring():
    _register_fake_rsc()

    @backend_dispatch
    def my_func(x, n_jobs=None):
        """Run my_func.

        Parameters
        ----------
        x
            Input value.
        n_jobs
            Host-only parameter.

        Returns
        -------
        Result.
        """
        return f"cpu:{x}:{n_jobs}"

    _dispatcher._dispatch_impl._update_signatures()

    sig = inspect.signature(my_func)
    assert list(sig.parameters) == ["x", "n_jobs", "backend_param", "backend"]
    doc = my_func.__doc__
    assert "backend_param (rapids_singlecell)" in doc
    assert "Other Parameters" in doc
    assert "Backend selector injected by ``scverse-backends``" in doc


def test_untrusted_backend_cannot_claim_reserved_gpu_alias():
    class BadBackend:
        name = "bad_backend"
        aliases = ["gpu"]

    with pytest.warns(UserWarning, match="reserved by squidpy"):
        _dispatcher._registry._register_backend(BadBackend(), entrypoint_name="bad_backend")

    assert get_backend("bad_backend").name == "bad_backend"
    assert get_backend("gpu") is None
    with pytest.raises(ValueError, match="Use a concrete backend alias"):
        settings.backend = "gpu"


def test_backend_dispatch_surface():
    dispatched = [
        sq.gr.spatial_neighbors,
        sq.gr.nhood_enrichment,
        sq.gr.centrality_scores,
        sq.gr.interaction_matrix,
        sq.gr.ripley,
        sq.gr.calculate_niche,
        sq.gr.co_occurrence,
        sq.gr.ligrec,
        sq.gr.spatial_autocorr,
        sq.gr.sepal,
        sq.im.calculate_image_features,
    ]
    for func in dispatched:
        param = inspect.signature(func).parameters["backend"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default is None

    assert "backend" not in inspect.signature(sq.gr.mask_graph).parameters

    parallel_backed = [
        sq.gr.spatial_autocorr,
        sq.gr.nhood_enrichment,
        sq.gr.centrality_scores,
        sq.gr.ligrec,
        sq.gr.sepal,
        sq.im.calculate_image_features,
    ]
    for func in parallel_backed:
        parallel_param = inspect.signature(func).parameters["parallel_backend"]
        assert parallel_param.default == "loky"
