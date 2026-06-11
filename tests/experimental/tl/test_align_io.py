"""Unit tests for the I/O layer of the public ``align`` functions.

These exercise input resolution and write-back directly -- no estimator, no JAX --
so they cover the ``AnnData`` *and* ``SpatialData`` branches and every error guard
in :mod:`squidpy.experimental.tl._align._io` cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from anndata import AnnData

from squidpy.experimental._methods.align_landmarks import AffineFitResult
from squidpy.experimental.tl._align._io import (
    get_coords,
    resolve_obs_pair,
    shallow_copy_sdata,
    writeback_affine_sdata,
    writeback_obs,
)

_PTS = np.array([[10.0, 1.0], [12.0, 1.0], [11.0, 2.0], [10.0, 3.0], [12.0, 3.0]])


@dataclass
class _ShiftResult:
    """Minimal :class:`AlignResult`: a constant offset baked into ``transform``."""

    delta: float = 100.0

    def transform(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=float) + self.delta


def _adata(coords: np.ndarray = _PTS, *, key: str = "spatial") -> AnnData:
    adata = AnnData(np.zeros((coords.shape[0], 1)))
    adata.obsm[key] = coords.copy()
    return adata


def _sdata_tables(**tables: AnnData):
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    return sd.SpatialData(tables={name: TableModel.parse(adata) for name, adata in tables.items()})


def _sdata_points(cs: str = "qcs"):
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Identity

    pts = PointsModel.parse(_PTS, transformations={cs: Identity()})
    return sd.SpatialData(points={"pts": pts})


# ---------------------------------------------------------------------------
# resolve_obs_pair
# ---------------------------------------------------------------------------


def test_resolve_anndata_pair() -> None:
    ref, query = _adata(), _adata()
    r_adata, q_adata, container, element_key = resolve_obs_pair(ref, query, None, None)
    assert r_adata is ref and q_adata is query
    assert container is None and element_key is None


def test_resolve_anndata_requires_query() -> None:
    with pytest.raises(ValueError, match="`data_query` is required when `data_ref` is an AnnData"):
        resolve_obs_pair(_adata(), None, None, None)


def test_resolve_anndata_rejects_mixed_query() -> None:
    with pytest.raises(TypeError, match="Mixed AnnData/SpatialData"):
        resolve_obs_pair(_adata(), _sdata_points(), None, None)


def test_resolve_anndata_rejects_keys() -> None:
    with pytest.raises(ValueError, match="only valid for SpatialData"):
        resolve_obs_pair(_adata(), _adata(), "ref", None)


def test_resolve_bad_ref_type() -> None:
    with pytest.raises(TypeError, match="must be AnnData or SpatialData"):
        resolve_obs_pair(object(), _adata(), None, None)  # type: ignore[arg-type]


def test_resolve_sdata_pair() -> None:
    pytest.importorskip("spatialdata")
    ref_sd = _sdata_tables(ref=_adata())
    query_sd = _sdata_tables(query=_adata(_PTS + 5))
    r_adata, q_adata, container, element_key = resolve_obs_pair(ref_sd, query_sd, "ref", "query")
    assert r_adata is ref_sd.tables["ref"]
    assert q_adata is query_sd.tables["query"]
    assert container is query_sd
    assert element_key == "query"


def test_resolve_sdata_single_two_tables() -> None:
    pytest.importorskip("spatialdata")
    both = _sdata_tables(ref=_adata(), query=_adata(_PTS + 5))
    r_adata, q_adata, container, element_key = resolve_obs_pair(both, None, "ref", "query")
    assert r_adata is both.tables["ref"]
    assert q_adata is both.tables["query"]
    assert container is both
    assert element_key == "query"


def test_resolve_sdata_requires_keys() -> None:
    pytest.importorskip("spatialdata")
    both = _sdata_tables(ref=_adata(), query=_adata())
    with pytest.raises(ValueError, match="`ref_key` and `query_key` are required"):
        resolve_obs_pair(both, None, None, None)


def test_resolve_sdata_rejects_mixed_query() -> None:
    pytest.importorskip("spatialdata")
    with pytest.raises(TypeError, match="Mixed AnnData/SpatialData"):
        resolve_obs_pair(_sdata_tables(ref=_adata()), _adata(), "ref", "query")


def test_resolve_sdata_missing_key() -> None:
    pytest.importorskip("spatialdata")
    both = _sdata_tables(ref=_adata())
    with pytest.raises(KeyError, match="nope"):
        resolve_obs_pair(both, None, "nope", "ref")


# ---------------------------------------------------------------------------
# get_coords
# ---------------------------------------------------------------------------


def test_get_coords_missing_key() -> None:
    with pytest.raises(KeyError, match="missing.*not found"):
        get_coords(_adata(), "missing")


def test_get_coords_rejects_non_2d() -> None:
    adata = _adata()
    adata.obsm["bad"] = np.zeros((_PTS.shape[0], 3))
    with pytest.raises(ValueError, match=r"must be an \(N, 2\) array"):
        get_coords(adata, "bad")


# ---------------------------------------------------------------------------
# writeback_obs -- AnnData
# ---------------------------------------------------------------------------


def test_writeback_obs_object_mode_returns_result() -> None:
    result = _ShiftResult()
    out = writeback_obs(
        result,
        output_mode="object",
        query_adata=_adata(),
        container=None,
        element_key=None,
        spatial_key="spatial",
        key_added=None,
    )
    assert out is result


def test_writeback_obs_anndata_inplace() -> None:
    query = _adata()
    out = writeback_obs(
        _ShiftResult(),
        output_mode="inplace",
        query_adata=query,
        container=None,
        element_key=None,
        spatial_key="spatial",
        key_added="aligned",
    )
    assert out is None
    np.testing.assert_allclose(query.obsm["aligned"], _PTS + 100.0)


def test_writeback_obs_anndata_copy_leaves_original_untouched() -> None:
    query = _adata()
    out = writeback_obs(
        _ShiftResult(),
        output_mode="copy",
        query_adata=query,
        container=None,
        element_key=None,
        spatial_key="spatial",
        key_added="aligned",
    )
    assert isinstance(out, AnnData) and out is not query
    assert "aligned" in out.obsm
    assert "aligned" not in query.obsm


# ---------------------------------------------------------------------------
# writeback_obs -- SpatialData
# ---------------------------------------------------------------------------


def test_writeback_obs_sdata_inplace() -> None:
    pytest.importorskip("spatialdata")
    sdata = _sdata_tables(query=_adata())
    out = writeback_obs(
        _ShiftResult(),
        output_mode="inplace",
        query_adata=sdata.tables["query"],
        container=sdata,
        element_key="query",
        spatial_key="spatial",
        key_added="aligned",
    )
    assert out is None
    np.testing.assert_allclose(sdata.tables["query"].obsm["aligned"], _PTS + 100.0)


def test_writeback_obs_sdata_copy_leaves_original_untouched() -> None:
    pytest.importorskip("spatialdata")
    sdata = _sdata_tables(query=_adata())
    out = writeback_obs(
        _ShiftResult(),
        output_mode="copy",
        query_adata=sdata.tables["query"],
        container=sdata,
        element_key="query",
        spatial_key="spatial",
        key_added="aligned",
    )
    assert out is not sdata
    assert "aligned" in out.tables["query"].obsm
    # regression: copy must not leak the new key back into the input container
    assert "aligned" not in sdata.tables["query"].obsm


# ---------------------------------------------------------------------------
# writeback_affine_sdata
# ---------------------------------------------------------------------------


def test_writeback_affine_inplace_registers_transform() -> None:
    pytest.importorskip("spatialdata")
    from spatialdata.transformations import get_transformation

    sdata = _sdata_points()
    out = writeback_affine_sdata(
        AffineFitResult(matrix=np.eye(3)), sdata, output_mode="inplace", moving_cs="qcs", target_cs="tcs"
    )
    assert out is None
    assert "tcs" in get_transformation(sdata.points["pts"], get_all=True)


def test_writeback_affine_copy_leaves_original_untouched() -> None:
    pytest.importorskip("spatialdata")
    from spatialdata.transformations import get_transformation

    sdata = _sdata_points()
    out = writeback_affine_sdata(
        AffineFitResult(matrix=np.eye(3)), sdata, output_mode="copy", moving_cs="qcs", target_cs="tcs"
    )
    assert out is not sdata
    assert "tcs" in get_transformation(out.points["pts"], get_all=True)
    # regression: copy must not register the transform on the input container
    assert "tcs" not in get_transformation(sdata.points["pts"], get_all=True)


def test_writeback_affine_requires_cs_names() -> None:
    pytest.importorskip("spatialdata")
    with pytest.raises(ValueError, match="`cs_name_query` and `cs_name_ref` are required"):
        writeback_affine_sdata(
            AffineFitResult(matrix=np.eye(3)), _sdata_points(), output_mode="inplace", moving_cs=None, target_cs="tcs"
        )


def test_writeback_affine_no_matching_cs() -> None:
    pytest.importorskip("spatialdata")
    sdata = _sdata_points(cs="qcs")
    with pytest.raises(KeyError, match="No elements .* registered to coordinate system 'other'"):
        writeback_affine_sdata(
            AffineFitResult(matrix=np.eye(3)), sdata, output_mode="inplace", moving_cs="other", target_cs="tcs"
        )


# ---------------------------------------------------------------------------
# shallow_copy_sdata
# ---------------------------------------------------------------------------


def test_shallow_copy_sdata_preserves_elements() -> None:
    pytest.importorskip("spatialdata")
    sdata = _sdata_tables(ref=_adata(), query=_adata(_PTS + 5))
    copy = shallow_copy_sdata(sdata)
    assert copy is not sdata
    assert set(copy.tables) == {"ref", "query"}
