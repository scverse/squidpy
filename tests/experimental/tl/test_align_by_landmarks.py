from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from squidpy.experimental._methods.align_landmarks import AffineFitResult
from squidpy.experimental.tl import align_by_landmarks

# square corners; query = ref shifted by (5, 7) -> a pure translation both models recover
_REF = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
_SHIFT = np.array([5.0, 7.0])
_QUERY = _REF + _SHIFT


def _adata(coords: np.ndarray = _QUERY, *, key: str = "spatial") -> AnnData:
    adata = AnnData(np.zeros((coords.shape[0], 1)))
    adata.obsm[key] = coords.copy()
    return adata


@pytest.mark.parametrize("method", ["similarity", "affine"])
def test_object_mode_returns_affine_result(method: str) -> None:
    result = align_by_landmarks(_REF, _QUERY, method=method, output_mode="object")
    assert isinstance(result, AffineFitResult)
    assert result.matrix.shape == (3, 3)
    # affine maps query -> ref
    np.testing.assert_allclose(result.transform(_QUERY), _REF, atol=1e-6)
    assert result.metadata["method"] == method


def test_object_is_default() -> None:
    assert isinstance(align_by_landmarks(_REF, _QUERY), AffineFitResult)


def test_anndata_inplace_writes_default_key() -> None:
    adata = _adata()
    out = align_by_landmarks(_REF, _QUERY, method="affine", data=adata, output_mode="inplace")
    assert out is None
    assert "aligned_spatial" in adata.obsm
    np.testing.assert_allclose(adata.obsm["aligned_spatial"], _REF, atol=1e-6)
    # source coords untouched
    np.testing.assert_allclose(adata.obsm["spatial"], _QUERY)


def test_anndata_copy_leaves_original_untouched() -> None:
    adata = _adata()
    out = align_by_landmarks(_REF, _QUERY, method="affine", data=adata, output_mode="copy", key_added="xy_aligned")
    assert isinstance(out, AnnData) and out is not adata
    assert "xy_aligned" in out.obsm
    assert "xy_aligned" not in adata.obsm


def test_custom_spatial_key() -> None:
    adata = _adata(key="loc")
    align_by_landmarks(_REF, _QUERY, method="affine", data=adata, spatial_key="loc", output_mode="inplace")
    assert "aligned_loc" in adata.obsm


def test_existing_default_key_requires_explicit_key_added() -> None:
    adata = _adata()
    adata.obsm["aligned_spatial"] = np.zeros_like(_QUERY)
    with pytest.raises(ValueError, match="aligned_spatial.*already exists"):
        align_by_landmarks(_REF, _QUERY, method="affine", data=adata, output_mode="inplace")
    # explicit key_added overwrites without error
    align_by_landmarks(_REF, _QUERY, method="affine", data=adata, output_mode="inplace", key_added="aligned_spatial")
    np.testing.assert_allclose(adata.obsm["aligned_spatial"], _REF, atol=1e-6)


def test_write_mode_requires_data() -> None:
    with pytest.raises(ValueError, match="`data` is required"):
        align_by_landmarks(_REF, _QUERY, method="affine", output_mode="inplace")


def test_too_few_landmarks() -> None:
    with pytest.raises(ValueError, match="at least 3 landmark pairs"):
        align_by_landmarks(_REF[:2], _QUERY[:2], method="affine", output_mode="object")


def test_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same shape"):
        align_by_landmarks(_REF, _QUERY[:3], method="affine", output_mode="object")


def test_unknown_method_lists_available() -> None:
    with pytest.raises(ValueError, match=r"Unknown align_landmarks method 'nope'"):
        align_by_landmarks(_REF, _QUERY, method="nope", output_mode="object")


def test_non_finite_landmarks_rejected() -> None:
    bad = _QUERY.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="must contain only finite values"):
        align_by_landmarks(_REF, bad, method="affine", output_mode="object")


def test_bad_data_type_raises() -> None:
    with pytest.raises(TypeError, match="must be AnnData or SpatialData"):
        align_by_landmarks(_REF, _QUERY, method="affine", data=object(), output_mode="inplace")  # type: ignore[arg-type]


def test_spatialdata_copy_leaves_original_untouched() -> None:
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Identity, get_transformation

    pts = PointsModel.parse(_QUERY, transformations={"query_cs": Identity()})
    sdata = sd.SpatialData(points={"pts": pts})

    out = align_by_landmarks(
        _REF,
        _QUERY,
        method="affine",
        data=sdata,
        cs_name_query="query_cs",
        cs_name_ref="ref_cs",
        output_mode="copy",
    )
    assert out is not sdata
    assert "ref_cs" in get_transformation(out.points["pts"], get_all=True)
    assert "ref_cs" not in get_transformation(sdata.points["pts"], get_all=True)


def test_spatialdata_registers_transformation() -> None:
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Identity, get_transformation

    pts = PointsModel.parse(_QUERY, transformations={"query_cs": Identity()})
    sdata = sd.SpatialData(points={"pts": pts})

    out = align_by_landmarks(
        _REF,
        _QUERY,
        method="affine",
        data=sdata,
        cs_name_query="query_cs",
        cs_name_ref="ref_cs",
        output_mode="inplace",
    )
    assert out is None
    assert "ref_cs" in get_transformation(sdata.points["pts"], get_all=True)
