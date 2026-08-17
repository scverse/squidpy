"""Integration tests for the public STalign functions.

Tiny synthetic fixtures with ``niter=1`` keep these fast; they verify wiring,
key resolution, and write-back -- not solver quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("jax")

from squidpy.experimental.tl import StalignResult, align_stalign_image, align_stalign_obs
from tests.experimental.conftest import TINY_SOLVER as _TINY
from tests.experimental.conftest import make_adata as _adata
from tests.experimental.conftest import make_sdata_tables as _sdata_tables

_TINY_IMAGE = {"a": 4.0, "nt": 1, "niter": 1, "epV": 1.0}


def _sdata_images():
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import Image2DModel

    ref = np.zeros((1, 16, 16), dtype=float)
    ref[0, 5:11, 5:11] = 1.0
    query = np.zeros((1, 16, 16), dtype=float)
    query[0, 6:12, 4:10] = 1.0
    return sd.SpatialData(
        images={
            "ref": Image2DModel.parse(ref, dims=("c", "y", "x")),
            "query": Image2DModel.parse(query, dims=("c", "y", "x")),
        }
    )


# --- key_added=None: fit and return, touch nothing -------------------------------------


def test_key_added_none_returns_result_and_writes_nothing() -> None:
    ref, query = _adata(), _adata()
    result = align_stalign_obs(ref, query, **_TINY)
    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
    assert list(query.obsm) == ["spatial"]


def test_result_satisfies_align_result_protocol() -> None:
    from squidpy.experimental.tl import AlignResult

    assert isinstance(align_stalign_obs(_adata(), _adata(), **_TINY), AlignResult)


def test_alignment_types_are_reachable_from_the_public_module() -> None:
    """Everything a caller needs to read a fit lives on `squidpy.experimental.tl`.

    `AlignResult` is the method-agnostic contract; the concrete results and the
    solver-tuning TypedDicts are here because a caller cannot use the returned fit,
    or tune the solver, without them. Nothing is reachable only via a private path.
    """
    import squidpy.experimental.tl as tl

    for name in (
        "AlignResult",
        "StalignResult",
        "AffineFitResult",
        "StalignSolverKwargs",
        "StalignObsSolverKwargs",
    ):
        assert name in tl.__all__, name
        assert hasattr(tl, name), name


# --- writing ---------------------------------------------------------------------------


def test_key_added_writes_in_place() -> None:
    ref, query = _adata(), _adata()
    assert align_stalign_obs(ref, query, key_added="aligned", **_TINY) is None
    assert query.obsm["aligned"].shape == query.obsm["spatial"].shape


def test_not_inplace_leaves_original_untouched() -> None:
    ref, query = _adata(), _adata()
    out = align_stalign_obs(ref, query, key_added="aligned", inplace=False, **_TINY)
    assert isinstance(out, AnnData) and out is not query
    assert "aligned" in out.obsm
    assert "aligned" not in query.obsm


def test_key_added_may_overwrite_spatial_key() -> None:
    """``key_added`` equal to ``spatial_key`` is allowed -- destructive, but explicitly asked for."""
    ref, query = _adata(), _adata()
    original = query.obsm["spatial"].copy()
    align_stalign_obs(ref, query, key_added="spatial", **_TINY)
    assert list(query.obsm) == ["spatial"]
    assert not np.array_equal(query.obsm["spatial"], original)


def test_spatial_key_pair_reads_each_side() -> None:
    ref, query = _adata(key="coords_ref"), _adata(key="coords_query")
    result = align_stalign_obs(ref, query, spatial_key=("coords_ref", "coords_query"), **_TINY)
    assert isinstance(result, StalignResult)


# --- SpatialData tables ----------------------------------------------------------------


def test_sdata_pair_of_tables() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    result = align_stalign_obs(sdata, table_key=("ref", "query"), **_TINY)
    assert isinstance(result, StalignResult)
    assert "aligned" not in sdata.tables["query"].obsm


def test_sdata_writes_into_the_query_table_only() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    align_stalign_obs(sdata, table_key=("ref", "query"), key_added="aligned", **_TINY)
    assert "aligned" in sdata.tables["query"].obsm
    assert "aligned" not in sdata.tables["ref"].obsm


def test_sdata_not_inplace_leaves_original_untouched() -> None:
    sd = pytest.importorskip("spatialdata")

    sdata = _sdata_tables(ref=_adata(), query=_adata())
    out = align_stalign_obs(sdata, table_key=("ref", "query"), key_added="aligned", inplace=False, **_TINY)
    assert isinstance(out, sd.SpatialData) and out is not sdata
    assert "aligned" in out.tables["query"].obsm
    assert "aligned" not in sdata.tables["query"].obsm


def test_two_sdata_objects_share_one_table_key() -> None:
    ref_sdata = _sdata_tables(slice=_adata())
    query_sdata = _sdata_tables(slice=_adata())
    align_stalign_obs(ref_sdata, query_sdata, table_key="slice", key_added="aligned", **_TINY)
    assert "aligned" in query_sdata.tables["slice"].obsm
    assert "aligned" not in ref_sdata.tables["slice"].obsm


def test_mixed_containers_use_a_table_key_pair() -> None:
    ref_sdata = _sdata_tables(slice=_adata())
    query = _adata()
    result = align_stalign_obs(ref_sdata, query, table_key=("slice", None), **_TINY)
    assert isinstance(result, StalignResult)


# --- images ----------------------------------------------------------------------------


def test_images_fit_returns_result() -> None:
    sdata = _sdata_images()
    result = align_stalign_image(sdata, image_key=("ref", "query"), **_TINY_IMAGE)
    assert isinstance(result, StalignResult)
    assert sorted(sdata.images) == ["query", "ref"]


def test_images_key_added_materialises_a_warped_image() -> None:
    """A diffeomorphism has no SpatialData transformation to be registered as."""
    sdata = _sdata_images()
    expected = np.asarray(sdata.images["query"].data).shape

    align_stalign_image(sdata, image_key=("ref", "query"), key_added="query_aligned", **_TINY_IMAGE)

    assert "query_aligned" in sdata.images
    assert np.asarray(sdata.images["query_aligned"].data).shape == expected


def test_images_not_inplace_leaves_original_untouched() -> None:
    sd = pytest.importorskip("spatialdata")

    sdata = _sdata_images()
    out = align_stalign_image(sdata, image_key=("ref", "query"), key_added="aligned", inplace=False, **_TINY_IMAGE)
    assert isinstance(out, sd.SpatialData) and out is not sdata
    assert "aligned" in out.images
    assert "aligned" not in sdata.images


def test_image_alignment_recovers_a_known_shift() -> None:
    """The query square sits one row down and one column left of the reference."""
    sdata = _sdata_images()
    ref = np.asarray(sdata.images["ref"].data)
    query = np.asarray(sdata.images["query"].data)

    align_stalign_image(sdata, image_key=("ref", "query"), key_added="query_aligned", a=4.0, nt=2)
    aligned = np.asarray(sdata.images["query_aligned"].data)

    before = float(np.sum((query - ref) ** 2))
    after = float(np.sum((aligned - ref) ** 2))
    assert after < before / 2.0, f"overlap barely improved: {before:.1f} -> {after:.1f}"


def test_warp_image_rejects_a_point_cloud_fit() -> None:
    result = align_stalign_obs(_adata(), _adata(), **_TINY)
    with pytest.raises(ValueError, match="fitted on point clouds"):
        result.warp_image(np.zeros((1, 4, 4)))


# --- key errors ------------------------------------------------------------------------


def test_missing_spatial_key_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="no `obsm\\['missing'\\]`.*spatial"):
        align_stalign_obs(_adata(), _adata(), spatial_key="missing", **_TINY)


def test_missing_table_lists_what_is_available() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    with pytest.raises(KeyError, match="table_key='nope'.*query.*ref"):
        align_stalign_obs(sdata, table_key=("nope", "query"), **_TINY)


def test_table_key_on_anndata_is_rejected() -> None:
    with pytest.raises(ValueError, match="is an AnnData, which has no tables"):
        align_stalign_obs(_adata(), _adata(), table_key="t", **_TINY)


def test_sdata_without_table_key_is_ambiguous() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    with pytest.raises(ValueError, match="may hold several tables"):
        align_stalign_obs(sdata, _adata(), **_TINY)


def test_single_sdata_needs_a_table_key_pair() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    with pytest.raises(ValueError, match=r"pass `table_key=\(ref, query\)`"):
        align_stalign_obs(sdata, table_key="ref", **_TINY)


def test_single_sdata_needs_an_image_key_pair() -> None:
    sdata = _sdata_images()
    with pytest.raises(ValueError, match=r"pass `image_key=\(ref, query\)`"):
        align_stalign_image(sdata, image_key="ref", **_TINY_IMAGE)


def test_missing_image_lists_what_is_available() -> None:
    sdata = _sdata_images()
    with pytest.raises(KeyError, match="image_key='nope'.*query.*ref"):
        align_stalign_image(sdata, image_key=("ref", "nope"), **_TINY_IMAGE)


def test_bad_key_pair_is_rejected() -> None:
    with pytest.raises(ValueError, match="single key or a `\\(ref, query\\)` pair"):
        align_stalign_obs(_adata(), _adata(), spatial_key=("a", "b", "c"), **_TINY)


def test_query_required_for_anndata() -> None:
    with pytest.raises(ValueError, match="`data_query` is required"):
        align_stalign_obs(_adata(), **_TINY)


def test_solver_rejects_unknown_kwarg() -> None:
    ref, query = _adata(), _adata()
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        align_stalign_obs(ref, query, not_a_real_param=1.0, **_TINY)


def test_align_with_landmark_initialisation() -> None:
    ref, query = _adata(), _adata()
    landmarks = ref.obsm["spatial"][:3]

    result = align_stalign_obs(ref, query, landmarks_ref=landmarks, landmarks_query=landmarks, **_TINY)

    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
