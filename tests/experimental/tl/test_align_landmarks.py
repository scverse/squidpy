"""Integration tests for the landmark path of the public ``align``.

Closed-form and JAX-free, so these run everywhere. They cover both write-backs a landmark
fit can have: transforming coordinates into an ``obsm`` key, and registering the affine on
a whole SpatialData coordinate system.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from squidpy.experimental.methods.align_landmarks import AffineFitResult
from squidpy.experimental.tl import align

# square corners; query = ref shifted by (5, 7) -> a pure translation both models recover
_REF = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
_SHIFT = np.array([5.0, 7.0])
_QUERY = _REF + _SHIFT


def _adata(landmarks: np.ndarray, coords: np.ndarray, *, key: str = "spatial") -> AnnData:
    adata = AnnData(np.zeros((coords.shape[0], 1)))
    adata.obsm["landmarks"] = landmarks.copy()
    adata.obsm[key] = coords.copy()
    return adata


def _shapes(points: np.ndarray, cs: str = "global"):
    pytest.importorskip("spatialdata")
    import geopandas
    import shapely
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import Identity

    frame = geopandas.GeoDataFrame(geometry=[shapely.Point(*p) for p in points])
    frame["radius"] = 1.0
    return ShapesModel.parse(frame, transformations={cs: Identity()})


# --- fitting ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["similarity", "affine"])
def test_returns_an_affine_result(method: str) -> None:
    ref, query = _adata(_REF, _REF), _adata(_QUERY, _QUERY)
    result = align(ref, query, in_="obsm/landmarks", by="landmarks", method=method)

    assert isinstance(result, AffineFitResult)
    assert result.matrix.shape == (3, 3)
    np.testing.assert_allclose(result.transform(_QUERY), _REF, atol=1e-6)
    assert result.metadata["method"] == method


def test_landmarks_default_to_similarity() -> None:
    sdata = pytest.importorskip("spatialdata").SpatialData(
        shapes={"lm_ref": _shapes(_REF), "lm_query": _shapes(_QUERY)}
    )
    result = align(sdata, in_=("shapes/lm_ref", "shapes/lm_query"), by="landmarks")
    assert result.metadata["method"] == "similarity"


def test_landmarks_need_not_live_in_a_shapes_element() -> None:
    """``in_`` says where to read; ``by`` says what the arrays mean.

    Requiring a SpatialData just to hold four points would tax AnnData users for nothing.
    """
    ref, query = _adata(_REF, _REF), _adata(_QUERY, _QUERY)
    result = align(ref, query, in_="obsm/landmarks", by="landmarks", method="affine")
    np.testing.assert_allclose(result.transform(_QUERY), _REF, atol=1e-6)


def test_stalign_rejects_a_landmark_path() -> None:
    sdata = pytest.importorskip("spatialdata").SpatialData(
        shapes={"lm_ref": _shapes(_REF), "lm_query": _shapes(_QUERY)}
    )
    with pytest.raises(ValueError, match="does not support landmarks alignment.*obs, images"):
        align(sdata, in_=("shapes/lm_ref", "shapes/lm_query"), by="landmarks", method="stalign")


# --- writing coordinates ----------------------------------------------------------------


def test_apply_to_selects_what_moves() -> None:
    ref, query = _adata(_REF, _REF), _adata(_QUERY, _QUERY)
    out = align(
        ref, query, in_="obsm/landmarks", by="landmarks", method="affine", apply_to="obsm/spatial", out="obsm/aligned"
    )

    assert out is None
    np.testing.assert_allclose(query.obsm["aligned"], _REF, atol=1e-6)
    np.testing.assert_allclose(query.obsm["spatial"], _QUERY)


def test_apply_to_is_required_for_landmarks() -> None:
    """``in_`` holds correspondences, so it cannot also say which array to transform."""
    ref, query = _adata(_REF, _REF), _adata(_QUERY, _QUERY)
    with pytest.raises(ValueError, match="needs `apply_to` when aligning by landmarks"):
        align(ref, query, in_="obsm/landmarks", by="landmarks", method="affine", out="obsm/aligned")


def test_copy_leaves_original_untouched() -> None:
    ref, query = _adata(_REF, _REF), _adata(_QUERY, _QUERY)
    out = align(
        ref,
        query,
        in_="obsm/landmarks",
        by="landmarks",
        method="affine",
        apply_to="obsm/spatial",
        out="obsm/aligned",
        copy=True,
    )
    assert isinstance(out, AnnData) and out is not query
    assert "aligned" in out.obsm
    assert "aligned" not in query.obsm


# --- registering a transformation --------------------------------------------------------


def test_registers_a_transformation_on_the_coordinate_system() -> None:
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Identity, get_transformation

    sdata = sd.SpatialData(
        shapes={"lm_ref": _shapes(_REF, "ref_cs"), "lm_query": _shapes(_QUERY, "query_cs")},
        points={"pts": PointsModel.parse(_QUERY, transformations={"query_cs": Identity()})},
    )
    out = align(sdata, in_=("shapes/lm_ref", "shapes/lm_query"), by="landmarks", method="affine", out="cs/ref_cs")

    assert out is None
    assert "ref_cs" in get_transformation(sdata.points["pts"], get_all=True)


def test_registering_copy_leaves_original_untouched() -> None:
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Identity, get_transformation

    sdata = sd.SpatialData(
        shapes={"lm_ref": _shapes(_REF, "ref_cs"), "lm_query": _shapes(_QUERY, "query_cs")},
        points={"pts": PointsModel.parse(_QUERY, transformations={"query_cs": Identity()})},
    )
    out = align(
        sdata, in_=("shapes/lm_ref", "shapes/lm_query"), by="landmarks", method="affine", out="cs/ref_cs", copy=True
    )

    assert out is not sdata
    assert "ref_cs" in get_transformation(out.points["pts"], get_all=True)
    assert "ref_cs" not in get_transformation(sdata.points["pts"], get_all=True)


def test_registration_composes_with_an_existing_transform() -> None:
    """The fit maps ``query_cs`` coords into the target, not the element's intrinsic frame.

    An element placed into ``query_cs`` by a non-identity transform must keep that
    placement, so the registered transform has to compose the two.
    """
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Translation, get_transformation

    offset = np.array([100.0, 200.0])
    sdata = sd.SpatialData(
        # Landmarks are expressed in `query_cs`: intrinsic coords shifted by `offset`.
        shapes={"lm_ref": _shapes(_REF, "ref_cs"), "lm_query": _shapes(_QUERY + offset, "query_cs")},
        points={"pts": PointsModel.parse(_QUERY, transformations={"query_cs": Translation(offset, axes=("x", "y"))})},
    )
    align(sdata, in_=("shapes/lm_ref", "shapes/lm_query"), by="landmarks", method="affine", out="cs/ref_cs")

    matrix = get_transformation(sdata.points["pts"], to_coordinate_system="ref_cs").to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    mapped = _QUERY @ matrix[:2, :2].T + matrix[:2, 2]
    np.testing.assert_allclose(mapped, _REF, atol=1e-6)


def test_shared_coordinate_system_is_refused() -> None:
    """Registering moves everything in the coordinate system -- including the reference.

    With both samples in one coordinate system the write-back would drag the reference
    along with the query and silently produce a wrong answer, so it has to refuse.
    """
    sdata = pytest.importorskip("spatialdata").SpatialData(
        shapes={"lm_ref": _shapes(_REF, "global"), "lm_query": _shapes(_QUERY, "global")}
    )
    with pytest.raises(ValueError, match="both in coordinate system 'global'.*move the reference too"):
        align(sdata, in_=("shapes/lm_ref", "shapes/lm_query"), by="landmarks", out="cs/aligned")


def test_reference_in_another_object_is_fine() -> None:
    """Only a *shared* coordinate system is a problem; two objects cannot collide."""
    sd = pytest.importorskip("spatialdata")
    from spatialdata.transformations import get_transformation

    ref_sdata = sd.SpatialData(shapes={"lm": _shapes(_REF, "global")})
    query_sdata = sd.SpatialData(shapes={"lm": _shapes(_QUERY, "global")})

    align(ref_sdata, query_sdata, in_="shapes/lm", by="landmarks", out="cs/aligned")

    assert "aligned" in get_transformation(query_sdata["lm"], get_all=True)
    assert "aligned" not in get_transformation(ref_sdata["lm"], get_all=True)


def test_ambiguous_coordinate_system_is_refused() -> None:
    """Which system moves has to be unambiguous, so exactly one is required."""
    sd = pytest.importorskip("spatialdata")
    import geopandas
    import shapely
    from spatialdata.models import ShapesModel
    from spatialdata.transformations import Identity

    frame = geopandas.GeoDataFrame(geometry=[shapely.Point(*p) for p in _QUERY])
    frame["radius"] = 1.0
    both = ShapesModel.parse(frame, transformations={"a": Identity(), "b": Identity()})
    sdata = sd.SpatialData(shapes={"lm_ref": _shapes(_REF, "ref_cs"), "lm_query": both})

    with pytest.raises(ValueError, match="registered to 2 coordinate systems"):
        align(sdata, in_=("shapes/lm_ref", "shapes/lm_query"), by="landmarks", out="cs/aligned")


def test_table_landmarks_cannot_target_a_coordinate_system() -> None:
    """A table annotates elements; it has no coordinate system of its own to move."""
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    def table(points: np.ndarray) -> AnnData:
        adata = AnnData(np.zeros((points.shape[0], 1)))
        adata.obsm["landmarks"] = points
        return TableModel.parse(adata)

    sdata = sd.SpatialData(tables={"r": table(_REF), "q": table(_QUERY)})
    with pytest.raises(ValueError, match="a table has no coordinate system of its own"):
        align(
            sdata,
            in_=("tables/r/obsm/landmarks", "tables/q/obsm/landmarks"),
            by="landmarks",
            out="cs/aligned",
        )


def test_diffeomorphism_cannot_be_registered() -> None:
    """SpatialData's transformations are affine at most, so stalign has to materialise."""
    sd = pytest.importorskip("spatialdata")
    pytest.importorskip("jax")
    from spatialdata.models import TableModel

    def table(points: np.ndarray) -> AnnData:
        adata = AnnData(np.zeros((points.shape[0], 1)))
        adata.obsm["spatial"] = points
        return TableModel.parse(adata)

    pts = np.array([[10.0, 1.0], [12.0, 1.0], [11.0, 2.0], [10.0, 3.0], [12.0, 3.0]])
    sdata = sd.SpatialData(tables={"ref": table(pts), "query": table(pts)})

    with pytest.raises(ValueError, match="no transformation type for"):
        align(
            sdata,
            in_=("tables/ref/obsm/spatial", "tables/query/obsm/spatial"),
            out="cs/aligned",
            dx=0.5,
            blur=1.0,
            a=1.0,
            expand=1.0,
            nt=1,
            niter=1,
            epV=1.0,
        )


# --- input validation --------------------------------------------------------------------


def test_too_few_landmarks() -> None:
    ref, query = _adata(_REF[:2], _REF[:2]), _adata(_QUERY[:2], _QUERY[:2])
    with pytest.raises(ValueError, match="at least 3 landmark pairs"):
        align(ref, query, in_="obsm/landmarks", by="landmarks", method="affine")


def test_length_mismatch() -> None:
    ref, query = _adata(_REF, _REF), _adata(_QUERY[:3], _QUERY[:3])
    with pytest.raises(ValueError, match="same shape"):
        align(ref, query, in_="obsm/landmarks", by="landmarks", method="affine")


def test_unknown_method_lists_available() -> None:
    ref, query = _adata(_REF, _REF), _adata(_QUERY, _QUERY)
    with pytest.raises(ValueError, match="Unknown align method 'nope'.*affine.*similarity.*stalign"):
        align(ref, query, in_="obsm/landmarks", method="nope")


def test_non_finite_landmarks_rejected() -> None:
    bad = _QUERY.copy()
    bad[0, 0] = np.nan
    ref, query = _adata(_REF, _REF), _adata(bad, bad)
    with pytest.raises(ValueError, match="finite"):
        align(ref, query, in_="obsm/landmarks", by="landmarks", method="affine")
