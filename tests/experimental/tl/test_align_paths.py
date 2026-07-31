"""Unit tests for the ``in_`` / ``out`` path layer.

No estimator, no JAX -- these exercise parsing, reading and writing directly, so the
AnnData *and* SpatialData branches and every error guard are covered cheaply.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from squidpy.experimental.tl._align._paths import parse_path, read_path, write_path

_PTS = np.array([[10.0, 1.0], [12.0, 1.0], [11.0, 2.0], [10.0, 3.0], [12.0, 3.0]])


def _adata(coords: np.ndarray = _PTS, *, key: str = "spatial") -> AnnData:
    adata = AnnData(np.zeros((coords.shape[0], 1)))
    adata.obsm[key] = coords.copy()
    return adata


def _sdata_tables(**tables: AnnData):
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    return sd.SpatialData(tables={name: TableModel.parse(adata) for name, adata in tables.items()})


def _sdata_image(name: str = "he", shape: tuple[int, int, int] = (2, 8, 6)):
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import Image2DModel

    data = np.arange(int(np.prod(shape)), dtype=float).reshape(shape)
    return sd.SpatialData(images={name: Image2DModel.parse(data, dims=("c", "y", "x"))})


# --- parsing ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "modality", "element", "key"),
    [
        ("obsm/spatial", "points", None, "spatial"),
        ("/obsm/spatial/", "points", None, "spatial"),
        ("tables/s1/obsm/xy", "points", "s1", "xy"),
        ("images/he", "image", "he", None),
    ],
)
def test_parse_accepts_the_documented_forms(path, modality, element, key) -> None:
    parsed = parse_path(path, name="in_")
    assert (parsed.modality, parsed.element, parsed.key) == (modality, element, key)
    assert parsed.raw == path


@pytest.mark.parametrize(
    ("path", "match"),
    [
        ("", "is empty"),
        ("///", "is empty"),
        ("obsm", "Expected `obsm/<key>`"),
        ("obsm/a/b", "Expected `obsm/<key>`"),
        ("tables/s1/obs/x", "Expected `tables/<table>/obsm/<key>`"),
        ("tables/s1", "Expected `tables/<table>/obsm/<key>`"),
        ("images", "Expected `images/<name>`"),
        ("shapes/x", "does not read or write yet"),
        ("varm/x", "does not start with a known collection"),
    ],
)
def test_parse_rejects_malformed_paths(path, match) -> None:
    with pytest.raises(ValueError, match=match):
        parse_path(path, name="in_")


def test_parse_rejects_non_strings() -> None:
    with pytest.raises(TypeError, match="must be a string path"):
        parse_path(("obsm/spatial",), name="in_")  # type: ignore[arg-type]


# --- reading ---------------------------------------------------------------------------


def test_read_anndata_obsm() -> None:
    got = read_path(_adata(), parse_path("obsm/spatial", name="in_"), name="in_")
    np.testing.assert_array_equal(got, _PTS)


def test_read_sdata_table_obsm() -> None:
    sdata = _sdata_tables(s1=_adata(), s2=_adata(_PTS + 5))
    got = read_path(sdata, parse_path("tables/s2/obsm/spatial", name="in_"), name="in_")
    np.testing.assert_array_equal(got, _PTS + 5)


def test_read_image_is_channels_first() -> None:
    got = read_path(_sdata_image(), parse_path("images/he", name="in_"), name="in_")
    assert got.shape == (2, 8, 6)


def test_read_rejects_non_2d_coordinates() -> None:
    adata = _adata()
    adata.obsm["bad"] = np.zeros((5, 3))
    with pytest.raises(ValueError, match=r"must be an \(N, 2\) array"):
        read_path(adata, parse_path("obsm/bad", name="in_"), name="in_")


def test_missing_obsm_key_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="no `obsm\\['nope'\\]`.*spatial"):
        read_path(_adata(), parse_path("obsm/nope", name="in_"), name="in_")


def test_missing_table_lists_alternatives() -> None:
    sdata = _sdata_tables(s1=_adata(), s2=_adata())
    with pytest.raises(KeyError, match="no table 'nope'.*s1.*s2"):
        read_path(sdata, parse_path("tables/nope/obsm/spatial", name="in_"), name="in_")


def test_missing_image_lists_alternatives() -> None:
    with pytest.raises(KeyError, match="no image 'nope'.*he"):
        read_path(_sdata_image(), parse_path("images/nope", name="in_"), name="in_")


def test_table_path_against_anndata_is_rejected() -> None:
    with pytest.raises(ValueError, match="but the container is an AnnData"):
        read_path(_adata(), parse_path("tables/s1/obsm/spatial", name="in_"), name="in_")


def test_bare_obsm_path_against_sdata_is_ambiguous() -> None:
    sdata = _sdata_tables(s1=_adata(), s2=_adata())
    with pytest.raises(ValueError, match="ambiguous for a SpatialData"):
        read_path(sdata, parse_path("obsm/spatial", name="in_"), name="in_")


def test_image_path_against_anndata_is_rejected() -> None:
    with pytest.raises(TypeError, match="only a SpatialData holds"):
        read_path(_adata(), parse_path("images/he", name="in_"), name="in_")


# --- writing ---------------------------------------------------------------------------


def test_write_anndata_obsm() -> None:
    adata = _adata()
    write_path(adata, parse_path("obsm/aligned", name="out"), _PTS + 1)
    np.testing.assert_array_equal(adata.obsm["aligned"], _PTS + 1)
    np.testing.assert_array_equal(adata.obsm["spatial"], _PTS)


def test_write_sdata_table_obsm() -> None:
    sdata = _sdata_tables(s1=_adata(), s2=_adata())
    write_path(sdata, parse_path("tables/s2/obsm/aligned", name="out"), _PTS + 1)
    assert "aligned" in sdata.tables["s2"].obsm
    assert "aligned" not in sdata.tables["s1"].obsm


def test_write_image_creates_a_new_element() -> None:
    sdata = _sdata_image()
    write_path(sdata, parse_path("images/warped", name="out"), np.zeros((2, 8, 6)))
    assert sorted(sdata.images) == ["he", "warped"]
    assert np.asarray(sdata.images["warped"].data).shape == (2, 8, 6)


def test_write_promotes_a_2d_image() -> None:
    sdata = _sdata_image()
    write_path(sdata, parse_path("images/warped", name="out"), np.zeros((8, 6)))
    assert np.asarray(sdata.images["warped"].data).shape == (1, 8, 6)
