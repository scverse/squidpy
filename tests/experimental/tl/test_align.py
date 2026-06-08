"""Integration tests for the public ``align`` (STalign) function.

Tiny synthetic fixtures with ``niter=1`` keep these fast; they verify wiring,
write-back modes, and the key guard -- not solver quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("jax")

from squidpy.experimental._methods.align_samples._stalign_impl._tools import (
    STalignConfig,
    STalignPreprocessConfig,
    STalignRegistrationConfig,
    StalignResult,
)
from squidpy.experimental.tl import align


def _adata(*, key: str = "spatial") -> AnnData:
    pts = np.array([[10.0, 1.0], [12.0, 1.0], [11.0, 2.0], [10.0, 3.0], [12.0, 3.0]])
    adata = AnnData(np.zeros((pts.shape[0], 1)))
    adata.obsm[key] = pts
    return adata


def _tiny() -> STalignConfig:
    return STalignConfig(
        preprocess=STalignPreprocessConfig(dx=0.5, blur=1.0),
        registration=STalignRegistrationConfig(a=1.0, expand=1.0, nt=1, niter=1, epV=1.0),
    )


def test_object_mode_returns_result_and_touches_nothing() -> None:
    ref, query = _adata(), _adata()
    result = align(ref, query, method="stalign", output_mode="object", config=_tiny())
    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
    assert "aligned_spatial" not in query.obsm


def test_inplace_writes_explicit_key() -> None:
    ref, query = _adata(), _adata()
    out = align(ref, query, output_mode="inplace", key_added="spatial_aligned", config=_tiny())
    assert out is None
    assert query.obsm["spatial_aligned"].shape == query.obsm["spatial"].shape


def test_inplace_default_key() -> None:
    ref, query = _adata(), _adata()
    align(ref, query, output_mode="inplace", config=_tiny())
    assert "aligned_spatial" in query.obsm


def test_copy_leaves_original_untouched() -> None:
    ref, query = _adata(), _adata()
    out = align(ref, query, output_mode="copy", key_added="aligned", config=_tiny())
    assert isinstance(out, AnnData) and out is not query
    assert "aligned" in out.obsm
    assert "aligned" not in query.obsm


def test_existing_default_key_requires_explicit_key_added() -> None:
    ref, query = _adata(), _adata()
    query.obsm["aligned_spatial"] = np.zeros_like(query.obsm["spatial"])
    with pytest.raises(ValueError, match="aligned_spatial.*already exists"):
        align(ref, query, output_mode="inplace", config=_tiny())


def test_image_not_implemented() -> None:
    ref, query = _adata(), _adata()
    with pytest.raises(NotImplementedError, match="on='image'"):
        align(ref, query, on="image", config=_tiny())


def test_missing_spatial_key() -> None:
    ref, query = _adata(), _adata()
    with pytest.raises(KeyError, match="missing.*not found"):
        align(ref, query, spatial_key="missing", config=_tiny())


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
        config=_tiny(),
    )

    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
