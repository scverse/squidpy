"""Unit tests for the SpatialData transformation write-back.

No estimator, no JAX -- these exercise registering a fitted affine on a coordinate
system directly, including the copy semantics, which are easy to get subtly wrong
because :func:`shallow_copy_sdata` shares element objects with the original.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from squidpy.experimental.methods.align_landmarks import AffineFitResult
from squidpy.experimental.tl._align._io import shallow_copy_sdata, writeback_affine_sdata

_PTS = np.array([[10.0, 1.0], [12.0, 1.0], [11.0, 2.0], [10.0, 3.0], [12.0, 3.0]])


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
    with pytest.raises(ValueError, match="`moving_cs` and `target_cs` are required"):
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
