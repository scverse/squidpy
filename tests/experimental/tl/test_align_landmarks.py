"""Integration tests for the public ``align_landmarks`` function.

Closed-form and JAX-free, so these run everywhere. They cover both write-backs a landmark
fit can have: transforming coordinates into an ``obsm`` key, and registering the affine on
a whole SpatialData coordinate system.
"""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from squidpy.experimental.tl import align_landmarks
from tests.experimental.conftest import make_adata

# square corners; query = ref shifted by (5, 7) -> a pure translation both models recover
_REF = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
_SHIFT = np.array([5.0, 7.0])
_QUERY = _REF + _SHIFT


def _adata(points: np.ndarray, *, key: str = "spatial") -> AnnData:
    """The landmarks and the coordinates they move are the same points in these tests."""
    return make_adata(points, key=key, landmarks=points)


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


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """The returned matrix needs no squidpy helper to use -- this is the whole of it."""
    return points @ matrix[:2, :2].T + matrix[:2, 2]


@pytest.mark.parametrize("fit", ["similarity", "affine"])
def test_returns_the_homogeneous_affine(fit: str) -> None:
    ref, query = _adata(_REF), _adata(_QUERY)
    matrix = align_landmarks(ref, query, landmark_key="landmarks", fit=fit)

    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(_apply(matrix, _QUERY), _REF, atol=1e-6)


def test_fit_defaults_to_similarity() -> None:
    """Asserted behaviourally: 4-DOF cannot absorb a non-uniform scale, 6-DOF can.

    A pure translation is recovered identically by both fits, so it cannot tell them
    apart -- a query stretched along one axis only can.
    """
    stretched = _REF * np.array([3.0, 1.0])
    ref, query = _adata(_REF), _adata(stretched)

    default = align_landmarks(ref, query, landmark_key="landmarks")
    affine = align_landmarks(ref, query, landmark_key="landmarks", fit="affine")

    assert np.abs(_apply(default, stretched) - _REF).max() > 0.1, "the default must be the constrained fit"
    np.testing.assert_allclose(_apply(affine, stretched), _REF, atol=1e-6)


def test_landmarks_need_not_live_in_a_shapes_element() -> None:
    """An ``obsm`` key works too -- requiring a SpatialData just to hold four points
    would tax AnnData users for nothing."""
    ref, query = _adata(_REF), _adata(_QUERY)
    result = align_landmarks(ref, query, landmark_key="landmarks", fit="affine")
    np.testing.assert_allclose(_apply(result, _QUERY), _REF, atol=1e-6)


def test_table_key_reads_landmarks_from_a_table() -> None:
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    sdata = sd.SpatialData(tables={"r": TableModel.parse(_adata(_REF)), "q": TableModel.parse(_adata(_QUERY))})
    result = align_landmarks(sdata, landmark_key="landmarks", table_key=("r", "q"), fit="affine")
    np.testing.assert_allclose(_apply(result, _QUERY), _REF, atol=1e-6)


def test_unknown_fit_lists_available() -> None:
    ref, query = _adata(_REF), _adata(_QUERY)
    with pytest.raises(ValueError, match="Unknown `fit='nope'`.*affine, similarity"):
        align_landmarks(ref, query, landmark_key="landmarks", fit="nope")


# --- writing coordinates ----------------------------------------------------------------


def test_spatial_key_selects_what_moves() -> None:
    ref, query = _adata(_REF), _adata(_QUERY)
    out = align_landmarks(
        ref, query, landmark_key="landmarks", fit="affine", spatial_key="spatial", key_added="aligned"
    )

    assert out is None
    np.testing.assert_allclose(query.obsm["aligned"], _REF, atol=1e-6)
    np.testing.assert_allclose(query.obsm["spatial"], _QUERY)


def test_spatial_key_is_required_for_key_added() -> None:
    """The landmarks are correspondences, so they cannot also say which array to transform."""
    ref, query = _adata(_REF), _adata(_QUERY)
    with pytest.raises(ValueError, match="`key_added` needs `spatial_key`"):
        align_landmarks(ref, query, landmark_key="landmarks", fit="affine", key_added="aligned")


def test_spatial_key_without_key_added_is_rejected() -> None:
    ref, query = _adata(_REF), _adata(_QUERY)
    with pytest.raises(ValueError, match="needs `key_added` to be set"):
        align_landmarks(ref, query, landmark_key="landmarks", spatial_key="spatial")


def test_neither_write_target_returns_the_fit() -> None:
    """Nothing to write is the fit-only call, not an error: the matrix is the result."""
    ref, query = _adata(_REF), _adata(_QUERY)
    matrix = align_landmarks(ref, query, landmark_key="landmarks", fit="affine")

    assert matrix.shape == (3, 3)
    assert not any(key.startswith("aligned") for key in query.obsm), "nothing may be written"


def test_key_added_writes_into_the_query_itself() -> None:
    ref, query = _adata(_REF), _adata(_QUERY)
    args = {"landmark_key": "landmarks", "fit": "affine", "spatial_key": "spatial", "key_added": "aligned"}

    assert align_landmarks(ref, query, **args) is None
    np.testing.assert_allclose(query.obsm["aligned"], _REF, atol=1e-6)
    np.testing.assert_allclose(query.obsm["spatial"], _QUERY, err_msg="the source array must survive")


# --- registering a transformation --------------------------------------------------------


def test_registers_a_transformation_on_the_coordinate_system() -> None:
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import PointsModel
    from spatialdata.transformations import Identity, get_transformation

    sdata = sd.SpatialData(
        shapes={"lm_ref": _shapes(_REF, "ref_cs"), "lm_query": _shapes(_QUERY, "query_cs")},
        points={"pts": PointsModel.parse(_QUERY, transformations={"query_cs": Identity()})},
    )
    out = align_landmarks(sdata, landmark_key=("lm_ref", "lm_query"), fit="affine", target_coordinate_system="ref_cs")

    assert out is None
    assert "ref_cs" in get_transformation(sdata.points["pts"], get_all=True)


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
    align_landmarks(sdata, landmark_key=("lm_ref", "lm_query"), fit="affine", target_coordinate_system="ref_cs")

    matrix = get_transformation(sdata.points["pts"], to_coordinate_system="ref_cs").to_affine_matrix(
        input_axes=("x", "y"), output_axes=("x", "y")
    )
    mapped = _QUERY @ matrix[:2, :2].T + matrix[:2, 2]
    np.testing.assert_allclose(mapped, _REF, atol=1e-6)


def test_registered_result_is_stamped_with_the_coordinate_systems() -> None:
    sd = pytest.importorskip("spatialdata")

    sdata = sd.SpatialData(shapes={"lm_ref": _shapes(_REF, "ref_cs"), "lm_query": _shapes(_QUERY, "query_cs")})
    align_landmarks(sdata, landmark_key=("lm_ref", "lm_query"), target_coordinate_system="aligned")
    # write-back is exercised above; here we only care that no error is raised and the
    # transformation landed on the query landmarks element itself
    from spatialdata.transformations import get_transformation

    assert "aligned" in get_transformation(sdata.shapes["lm_query"], get_all=True)


def test_shared_coordinate_system_is_refused() -> None:
    """Registering moves everything in the coordinate system -- including the reference.

    With both samples in one coordinate system the write-back would drag the reference
    along with the query and silently produce a wrong answer, so it has to refuse.
    """
    sdata = pytest.importorskip("spatialdata").SpatialData(
        shapes={"lm_ref": _shapes(_REF, "global"), "lm_query": _shapes(_QUERY, "global")}
    )
    with pytest.raises(ValueError, match="both in coordinate system 'global'.*move the reference too"):
        align_landmarks(sdata, landmark_key=("lm_ref", "lm_query"), target_coordinate_system="aligned")


def test_reference_in_another_object_is_fine() -> None:
    """Only a *shared* coordinate system is a problem; two objects cannot collide."""
    sd = pytest.importorskip("spatialdata")
    from spatialdata.transformations import get_transformation

    ref_sdata = sd.SpatialData(shapes={"lm": _shapes(_REF, "global")})
    query_sdata = sd.SpatialData(shapes={"lm": _shapes(_QUERY, "global")})

    align_landmarks(ref_sdata, query_sdata, landmark_key="lm", target_coordinate_system="aligned")

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
        align_landmarks(sdata, landmark_key=("lm_ref", "lm_query"), target_coordinate_system="aligned")


def test_table_landmarks_cannot_target_a_coordinate_system() -> None:
    """A table annotates elements; it has no coordinate system of its own to move."""
    sd = pytest.importorskip("spatialdata")
    from spatialdata.models import TableModel

    def table(points: np.ndarray) -> AnnData:
        adata = AnnData(np.zeros((points.shape[0], 1)))
        adata.obsm["landmarks"] = points
        return TableModel.parse(adata)

    sdata = sd.SpatialData(tables={"r": table(_REF), "q": table(_QUERY)})
    with pytest.raises(ValueError, match="has no coordinate system of its own"):
        align_landmarks(sdata, landmark_key="landmarks", table_key=("r", "q"), target_coordinate_system="aligned")


def test_key_added_and_target_coordinate_system_are_exclusive() -> None:
    ref, query = _adata(_REF), _adata(_QUERY)
    with pytest.raises(ValueError, match="mutually exclusive"):
        align_landmarks(
            ref,
            query,
            landmark_key="landmarks",
            spatial_key="spatial",
            key_added="aligned",
            target_coordinate_system="aligned",
        )


def test_target_coordinate_system_needs_a_spatialdata() -> None:
    ref, query = _adata(_REF), _adata(_QUERY)
    with pytest.raises(TypeError, match="only a SpatialData has"):
        align_landmarks(ref, query, landmark_key="landmarks", target_coordinate_system="aligned")


def test_missing_shapes_element_lists_what_is_available() -> None:
    sdata = pytest.importorskip("spatialdata").SpatialData(
        shapes={"lm_ref": _shapes(_REF), "lm_query": _shapes(_QUERY)}
    )
    with pytest.raises(KeyError, match="landmark_key='nope'.*lm_query.*lm_ref"):
        align_landmarks(sdata, landmark_key=("lm_ref", "nope"))


# --- input validation --------------------------------------------------------------------


def test_too_few_landmarks() -> None:
    ref, query = _adata(_REF[:2]), _adata(_QUERY[:2])
    with pytest.raises(ValueError, match="at least 3 landmark pairs"):
        align_landmarks(ref, query, landmark_key="landmarks", fit="affine")


def test_length_mismatch() -> None:
    ref, query = _adata(_REF), _adata(_QUERY[:3])
    with pytest.raises(ValueError, match="same shape"):
        align_landmarks(ref, query, landmark_key="landmarks", fit="affine")


def test_non_finite_landmarks_rejected() -> None:
    bad = _QUERY.copy()
    bad[0, 0] = np.nan
    ref, query = _adata(_REF), _adata(bad)
    with pytest.raises(ValueError, match="finite"):
        align_landmarks(ref, query, landmark_key="landmarks", fit="affine")


def test_align_landmarks_accepts_arrays():
    """Landmarks are correspondences, so they need no container to be passed in."""
    query = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0], [0.0, 5.0]])
    expected = np.array([[2.0, 0.0, 7.0], [0.0, 2.0, -3.0], [0.0, 0.0, 1.0]])
    ref = query @ expected[:2, :2].T + expected[:2, 2]

    matrix = align_landmarks(ref, query, fit="affine")
    np.testing.assert_allclose(matrix, expected, atol=1e-9)
    # Same answer either way in: the array path is a shortcut, not a second algorithm.
    np.testing.assert_allclose(
        align_landmarks(
            AnnData(X=np.zeros((len(ref), 1)), obsm={"lm": ref}),
            AnnData(X=np.zeros((len(query), 1)), obsm={"lm": query}),
            landmark_key="lm",
            fit="affine",
        ),
        matrix,
    )


def test_align_landmarks_rejects_container_arguments_for_arrays():
    """A key that addresses nothing is an error, not something to ignore."""
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="addresses a container"):
        align_landmarks(points, points, landmark_key="lm")
    with pytest.raises(ValueError, match="must be the matching"):
        align_landmarks(points)
