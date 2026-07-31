"""Integration tests for the public ``align`` (STalign) function.

Tiny synthetic fixtures with ``niter=1`` keep these fast; they verify wiring,
path resolution, and write-back -- not solver quality.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

pytest.importorskip("jax")

from squidpy.experimental.methods.align_samples import StalignResult
from squidpy.experimental.tl import align

# Flat solver kwargs (assembled into the config internally) -- smallest possible solve.
_TINY = {"dx": 0.5, "blur": 1.0, "a": 1.0, "expand": 1.0, "nt": 1, "niter": 1, "epV": 1.0}
_TINY_IMAGE = {"a": 4.0, "nt": 1, "niter": 1, "epV": 1.0}


def _adata(*, key: str = "spatial") -> AnnData:
    pts = np.array([[10.0, 1.0], [12.0, 1.0], [11.0, 2.0], [10.0, 3.0], [12.0, 3.0]])
    adata = AnnData(np.zeros((pts.shape[0], 1)))
    adata.obsm[key] = pts
    return adata


def _sdata_tables(**tables: AnnData):
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    return sd.SpatialData(tables={name: TableModel.parse(adata) for name, adata in tables.items()})


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


# --- out=None: fit and return, touch nothing ------------------------------------------


def test_out_none_returns_result_and_writes_nothing() -> None:
    ref, query = _adata(), _adata()
    result = align(ref, query, in_="obsm/spatial", method="stalign", **_TINY)
    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
    assert list(query.obsm) == ["spatial"]


def test_result_satisfies_align_result_protocol() -> None:
    from squidpy.experimental.tl import AlignResult

    assert isinstance(align(_adata(), _adata(), in_="obsm/spatial", **_TINY), AlignResult)


def test_public_surface_is_align_result_only() -> None:
    import squidpy.experimental.tl as tl

    # `AlignResult` is the only result type exposed; concretes stay in their home modules.
    assert "AlignResult" in tl.__all__
    assert not hasattr(tl, "StalignResult")
    assert not hasattr(tl, "AffineFitResult")


# --- writing ---------------------------------------------------------------------------


def test_out_writes_in_place() -> None:
    ref, query = _adata(), _adata()
    assert align(ref, query, in_="obsm/spatial", out="obsm/aligned", **_TINY) is None
    assert query.obsm["aligned"].shape == query.obsm["spatial"].shape


def test_copy_leaves_original_untouched() -> None:
    ref, query = _adata(), _adata()
    out = align(ref, query, in_="obsm/spatial", out="obsm/aligned", copy=True, **_TINY)
    assert isinstance(out, AnnData) and out is not query
    assert "aligned" in out.obsm
    assert "aligned" not in query.obsm


def test_out_may_overwrite_the_input_path() -> None:
    """``out`` equal to ``in_`` is allowed -- destructive, but explicitly asked for."""
    ref, query = _adata(), _adata()
    original = query.obsm["spatial"].copy()
    align(ref, query, in_="obsm/spatial", out="obsm/spatial", **_TINY)
    assert list(query.obsm) == ["spatial"]
    assert not np.array_equal(query.obsm["spatial"], original)


# --- SpatialData tables ----------------------------------------------------------------


def test_sdata_pair_of_tables() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    result = align(
        sdata,
        in_=("tables/ref/obsm/spatial", "tables/query/obsm/spatial"),
        **_TINY,
    )
    assert isinstance(result, StalignResult)
    assert "aligned" not in sdata.tables["query"].obsm


def test_sdata_writes_into_the_named_table_only() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    align(
        sdata,
        in_=("tables/ref/obsm/spatial", "tables/query/obsm/spatial"),
        out="tables/query/obsm/aligned",
        **_TINY,
    )
    assert "aligned" in sdata.tables["query"].obsm
    assert "aligned" not in sdata.tables["ref"].obsm


def test_sdata_copy_leaves_original_untouched() -> None:
    sd = pytest.importorskip("spatialdata")

    sdata = _sdata_tables(ref=_adata(), query=_adata())
    out = align(
        sdata,
        in_=("tables/ref/obsm/spatial", "tables/query/obsm/spatial"),
        out="tables/query/obsm/aligned",
        copy=True,
        **_TINY,
    )
    assert isinstance(out, sd.SpatialData) and out is not sdata
    assert "aligned" in out.tables["query"].obsm
    assert "aligned" not in sdata.tables["query"].obsm


def test_two_sdata_objects_share_one_path() -> None:
    ref_sdata = _sdata_tables(slice=_adata())
    query_sdata = _sdata_tables(slice=_adata())
    align(ref_sdata, query_sdata, in_="tables/slice/obsm/spatial", out="tables/slice/obsm/aligned", **_TINY)
    assert "aligned" in query_sdata.tables["slice"].obsm
    assert "aligned" not in ref_sdata.tables["slice"].obsm


# --- images ----------------------------------------------------------------------------


def test_images_fit_returns_result() -> None:
    sdata = _sdata_images()
    result = align(sdata, in_=("images/ref", "images/query"), **_TINY_IMAGE)
    assert isinstance(result, StalignResult)
    assert sorted(sdata.images) == ["query", "ref"]


def test_images_out_materialises_a_warped_image() -> None:
    """A diffeomorphism has no SpatialData transformation to be registered as."""
    sdata = _sdata_images()
    expected = np.asarray(sdata.images["query"].data).shape

    align(sdata, in_=("images/ref", "images/query"), out="images/query_aligned", **_TINY_IMAGE)

    assert "query_aligned" in sdata.images
    assert np.asarray(sdata.images["query_aligned"].data).shape == expected


def test_image_alignment_recovers_a_known_shift() -> None:
    """The query square sits one row down and one column left of the reference."""
    sdata = _sdata_images()
    ref = np.asarray(sdata.images["ref"].data)
    query = np.asarray(sdata.images["query"].data)

    align(sdata, in_=("images/ref", "images/query"), out="images/query_aligned", a=4.0, nt=2)
    aligned = np.asarray(sdata.images["query_aligned"].data)

    before = float(np.sum((query - ref) ** 2))
    after = float(np.sum((aligned - ref) ** 2))
    assert after < before / 2.0, f"overlap barely improved: {before:.1f} -> {after:.1f}"


def test_warp_image_rejects_a_point_cloud_fit() -> None:
    result = align(_adata(), _adata(), in_="obsm/spatial", **_TINY)
    with pytest.raises(ValueError, match="fitted on point clouds"):
        result.warp_image(np.zeros((1, 4, 4)))


# --- path errors -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ("nope/x", "does not start with a known collection"),
        ("obsm", "Expected `obsm/<key>`"),
        ("tables/t/spatial", "Expected `tables/<table>/obsm/<key>`"),
        ("images/a/b", "Expected `images/<name>`"),
        ("labels/x", "does not read or write yet"),
        ("", "is empty"),
    ],
)
def test_invalid_paths_explain_the_grammar(path: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        align(_adata(), _adata(), in_=path, **_TINY)


def test_missing_obsm_key_lists_what_is_available() -> None:
    with pytest.raises(KeyError, match="no `obsm\\['missing'\\]`.*spatial"):
        align(_adata(), _adata(), in_="obsm/missing", **_TINY)


def test_missing_table_lists_what_is_available() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    with pytest.raises(KeyError, match="no table 'nope'.*query.*ref"):
        align(sdata, in_="tables/nope/obsm/spatial", **_TINY)


def test_table_path_on_anndata_is_rejected() -> None:
    with pytest.raises(ValueError, match="but the container is an AnnData"):
        align(_adata(), _adata(), in_="tables/t/obsm/spatial", **_TINY)


def test_bare_obsm_path_on_sdata_is_ambiguous() -> None:
    sdata = _sdata_tables(ref=_adata(), query=_adata())
    with pytest.raises(ValueError, match="ambiguous for a SpatialData"):
        align(sdata, in_="obsm/spatial", **_TINY)


def test_mixed_modalities_in_in_are_rejected() -> None:
    sdata = _sdata_images()
    with pytest.raises(ValueError, match="mixes modalities"):
        align(sdata, in_=("images/ref", "tables/t/obsm/spatial"), **_TINY)


def test_out_modality_must_match_in() -> None:
    sdata = _sdata_images()
    with pytest.raises(ValueError, match="does not convert between the two"):
        align(sdata, in_=("images/ref", "images/query"), out="tables/t/obsm/x", **_TINY_IMAGE)


def test_query_required_for_anndata() -> None:
    with pytest.raises(ValueError, match="`data_query` is required"):
        align(_adata(), in_="obsm/spatial", **_TINY)


def test_align_with_landmarks() -> None:
    ref, query = _adata(), _adata()
    landmarks = ref.obsm["spatial"][:3]

    result = align(
        ref,
        query,
        in_="obsm/spatial",
        method="stalign",
        landmarks_source=landmarks,
        landmarks_target=landmarks,
        **_TINY,
    )

    assert isinstance(result, StalignResult)
    assert result.aligned_points.shape == query.obsm["spatial"].shape
