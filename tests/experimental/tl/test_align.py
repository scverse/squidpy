"""Integration tests for the public ``align`` (STalign) function.

Tiny synthetic fixtures with ``niter=1`` keep these fast; they verify wiring,
write-back modes, and the key guard -- not solver quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("jax")

from squidpy.experimental.method_registry.align_samples._stalign_impl._tools import StalignResult
from squidpy.experimental.tl import align

# Flat solver kwargs (assembled into the config internally) -- smallest possible solve.
_TINY = {"dx": 0.5, "blur": 1.0, "a": 1.0, "expand": 1.0, "nt": 1, "niter": 1, "epV": 1.0}


def _adata(*, key: str = "spatial") -> AnnData:
    pts = np.array([[10.0, 1.0], [12.0, 1.0], [11.0, 2.0], [10.0, 3.0], [12.0, 3.0]])
    adata = AnnData(np.zeros((pts.shape[0], 1)))
    adata.obsm[key] = pts
    return adata


def test_object_mode_returns_result_and_touches_nothing() -> None:
    ref, query = _adata(), _adata()
    result = align(ref, query, method="stalign", output_mode="object", **_TINY)
    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
    assert "aligned_spatial" not in query.obsm


def test_inplace_writes_explicit_key() -> None:
    ref, query = _adata(), _adata()
    out = align(ref, query, output_mode="inplace", key_added="spatial_aligned", **_TINY)
    assert out is None
    assert query.obsm["spatial_aligned"].shape == query.obsm["spatial"].shape


def test_inplace_default_key() -> None:
    ref, query = _adata(), _adata()
    align(ref, query, output_mode="inplace", **_TINY)
    assert "aligned_spatial" in query.obsm


def test_copy_leaves_original_untouched() -> None:
    ref, query = _adata(), _adata()
    out = align(ref, query, output_mode="copy", key_added="aligned", **_TINY)
    assert isinstance(out, AnnData) and out is not query
    assert "aligned" in out.obsm
    assert "aligned" not in query.obsm


def test_existing_default_key_requires_explicit_key_added() -> None:
    ref, query = _adata(), _adata()
    query.obsm["aligned_spatial"] = np.zeros_like(query.obsm["spatial"])
    with pytest.raises(ValueError, match="aligned_spatial.*already exists"):
        align(ref, query, output_mode="inplace", **_TINY)


def test_image_not_implemented() -> None:
    ref, query = _adata(), _adata()
    with pytest.raises(NotImplementedError, match="on='image'"):
        align(ref, query, on="image", **_TINY)


def test_missing_spatial_key() -> None:
    ref, query = _adata(), _adata()
    with pytest.raises(KeyError, match="missing.*not found"):
        align(ref, query, spatial_key="missing", **_TINY)


def test_public_surface_is_align_result_only() -> None:
    import squidpy.experimental.tl as tl

    # `AlignResult` is the only result type exposed; concretes stay in their home modules.
    assert "AlignResult" in tl.__all__
    assert not hasattr(tl, "StalignResult")
    assert not hasattr(tl, "AffineFitResult")


def test_object_result_satisfies_align_result_protocol() -> None:
    from squidpy.experimental.tl import AlignResult

    result = align(_adata(), _adata(), method="stalign", output_mode="object", **_TINY)
    assert isinstance(result, AlignResult)


def _sdata_tables(**tables: AnnData):
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    return sd.SpatialData(tables={name: TableModel.parse(adata) for name, adata in tables.items()})


def test_sdata_object_mode() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    result = align(sdata, method="stalign", ref_key="ref", query_key="query", output_mode="object", **_TINY)
    assert isinstance(result, StalignResult)
    assert "aligned_spatial" not in sdata.tables["query"].obsm


def test_sdata_copy_leaves_original_untouched() -> None:
    sd = pytest.importorskip("spatialdata")

    sdata = _sdata_tables(ref=_adata(), query=_adata())
    out = align(sdata, ref_key="ref", query_key="query", output_mode="copy", key_added="aligned", **_TINY)
    assert isinstance(out, sd.SpatialData) and out is not sdata
    assert "aligned" in out.tables["query"].obsm
    assert "aligned" not in sdata.tables["query"].obsm


def test_align_with_landmarks() -> None:
    ref, query = _adata(), _adata()
    landmarks = ref.obsm["spatial"][:3]

    result = align(
        ref,
        query,
        method="stalign",
        output_mode="object",
        landmarks_source=landmarks,
        landmarks_target=landmarks,
        **_TINY,
    )

    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
