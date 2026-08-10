from __future__ import annotations

from copy import copy
from importlib import metadata
from inspect import Parameter, signature
from typing import TYPE_CHECKING

import numpy as np
import pytest
from anndata import AnnData

import squidpy as sq
from squidpy._backends import dispatched_functions, dispatcher
from squidpy._backends import settings as backend_settings
from squidpy.testing import validate_backend

if TYPE_CHECKING:
    from typing import ClassVar


class FakeRapidsBackend:
    name = "rapids-singlecell"
    aliases = ("cuda", "rapids", "rapids_singlecell")

    def spatial_autocorr(
        self,
        adata: AnnData,
        *,
        mode: str = "moran",
        copy: bool = False,
        fake_param: str | None = None,
    ):
        adata.uns["fake_backend_called"] = {"function": "spatial_autocorr", "fake_param": fake_param}
        return sq.gr.spatial_autocorr(adata, mode=mode, copy=copy, backend="cpu")

    def co_occurrence(self, adata: AnnData, cluster_key: str, *, copy: bool = False):
        return sq.gr.co_occurrence(adata, cluster_key=cluster_key, copy=copy, backend="cpu")


class FakeDistribution:
    metadata: ClassVar = {"Name": "rapids-singlecell"}


class FakeEntryPoint:
    name = "rapids-singlecell"
    value = "rapids_singlecell.backends.squidpy:SquidpyBackend"
    dist = FakeDistribution()

    @staticmethod
    def load():
        """Load the fake backend entry point."""
        return FakeRapidsBackend


@pytest.fixture
def fake_rapids_backend(monkeypatch):
    registry = dispatcher._registry
    dispatch_impl = dispatcher._dispatch_impl
    old_backend = backend_settings.backend
    old_state = {
        "_backends": copy(registry._backends),
        "_alias_map": copy(registry._alias_map),
        "_load_errors": copy(registry._load_errors),
        "_registration_errors": copy(registry._registration_errors),
        "_warned_untrusted": copy(registry._warned_untrusted),
        "_discovered": registry._discovered,
        "_sig_cache": copy(dispatch_impl._sig_cache),
    }

    backend_settings._backend_var.set("cpu")
    registry._backends.clear()
    registry._alias_map.clear()
    registry._load_errors.clear()
    registry._registration_errors.clear()
    registry._warned_untrusted.clear()
    registry._discovered = False
    entry_points = metadata.entry_points

    def _entry_points(*args, **kwargs):
        if kwargs.get("group") == "squidpy.backends":
            return [FakeEntryPoint()]
        return entry_points(*args, **kwargs)

    monkeypatch.setattr("scverse_backends._registry.importlib.metadata.entry_points", _entry_points)
    dispatcher.discover()

    yield

    backend_settings._backend_var.set(old_backend)
    registry._backends.clear()
    registry._backends.update(old_state["_backends"])
    registry._alias_map.clear()
    registry._alias_map.update(old_state["_alias_map"])
    registry._load_errors.clear()
    registry._load_errors.update(old_state["_load_errors"])
    registry._registration_errors.clear()
    registry._registration_errors.update(old_state["_registration_errors"])
    registry._warned_untrusted.clear()
    registry._warned_untrusted.update(old_state["_warned_untrusted"])
    registry._discovered = old_state["_discovered"]
    dispatch_impl._sig_cache.clear()
    dispatch_impl._sig_cache.update(old_state["_sig_cache"])
    dispatch_impl._update_signatures()


def _spatial_adata() -> AnnData:
    adata = AnnData(np.arange(18, dtype=np.float32).reshape(9, 2))
    adata.obsm["spatial"] = np.mgrid[:3, :3].reshape(2, -1).T
    sq.gr.spatial_neighbors_knn(adata)
    return adata


def test_dispatched_functions_have_backend_keyword():
    assert {func.__name__ for func in dispatched_functions} == {
        "calculate_niche",
        "co_occurrence",
        "ligrec",
        "spatial_autocorr",
    }
    for func in dispatched_functions:
        backend = signature(func).parameters["backend"]

        assert backend.kind is Parameter.KEYWORD_ONLY
        assert backend.default is None


def test_settings_resolve_rapids_alias(fake_rapids_backend):
    backend_settings.backend = "cuda"

    assert backend_settings.backend == "rapids-singlecell"
    assert dispatcher.get_backend("rapids") is not None
    assert backend_settings.available_backends() == ["rapids-singlecell"]


def test_settings_context_dispatches_and_restores(fake_rapids_backend):
    adata = _spatial_adata()

    with backend_settings.use_backend("rapids"):
        assert backend_settings.backend == "rapids-singlecell"
        sq.gr.spatial_autocorr(adata, mode="moran", fake_param="from-backend")

    assert backend_settings.backend == "cpu"
    assert adata.uns["fake_backend_called"] == {
        "function": "spatial_autocorr",
        "fake_param": "from-backend",
    }


def test_call_backend_overrides_settings(fake_rapids_backend):
    adata = _spatial_adata()

    with backend_settings.use_backend("cuda"):
        sq.gr.spatial_autocorr(adata, mode="moran", backend="cpu")

    assert "fake_backend_called" not in adata.uns


def test_backend_only_parameters_are_injected(fake_rapids_backend):
    fake_param = signature(sq.gr.spatial_autocorr).parameters["fake_param"]

    assert fake_param.kind is Parameter.KEYWORD_ONLY
    assert fake_param.default is None


def test_backend_conformance_harness(fake_rapids_backend):
    assert validate_backend("cuda") == {
        "spatial_autocorr": "PASSED",
        "co_occurrence": "PASSED",
    }


def test_reserved_gpu_backend_name():
    with pytest.raises(ValueError, match="reserved by squidpy"):
        backend_settings.backend = "gpu"
