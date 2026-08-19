"""Smoke tests for the STalign align functions.

Deliberately tiny: `TINY_SOLVER` runs one or two iterations, so these assert that the
whole path executes and writes where it says it does, not that the fit converges.
"""

from __future__ import annotations

import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Image3DModel

from tests.experimental.conftest import ALIGN_PTS, TINY_SOLVER, make_adata

pytest.importorskip("jax")

from squidpy.experimental.tl import align_stalign_image, align_stalign_obs, align_stalign_volume  # noqa: E402

IMAGE_SOLVER = {"a": 4.0, "nt": 1, "niter": 2, "epV": 1.0}
VOLUME_SOLVER = {"a": 3.0, "nt": 1, "niter": 2, "epV": 1.0}


def _sdata_image(array: np.ndarray, key: str, **kwargs: object) -> SpatialData:
    model = Image3DModel if array.ndim == 4 else Image2DModel
    dims = ("c", "z", "y", "x") if array.ndim == 4 else ("c", "y", "x")
    return SpatialData(images={key: model.parse(array, dims=dims, **kwargs)})


def test_obs_fit_returns_a_result_and_writes_a_copy() -> None:
    ref, query = make_adata(ALIGN_PTS), make_adata(ALIGN_PTS + 0.4)

    result = align_stalign_obs(ref, query, **TINY_SOLVER)
    assert result.aligned_points.shape == ALIGN_PTS.shape

    out = align_stalign_obs(ref, query, key_added="aligned", **TINY_SOLVER)
    assert out.obsm["aligned"].shape == ALIGN_PTS.shape
    assert "aligned" not in query.obsm, "the input must be left untouched"


@pytest.mark.parametrize("scale_factors", [None, [2]], ids=["single_scale", "multiscale"])
def test_image_fit_reads_both_element_layouts(scale_factors: list[int] | None) -> None:
    """A multiscale element is a DataTree, which carries a `.dims` attribute of its own.

    Duck-typing on that attribute silently skipped the scale resolution and crashed on
    every multiscale image -- the layout spatialdata's readers actually produce.
    """
    image = np.random.default_rng(0).random((1, 32, 32))
    kwargs = {} if scale_factors is None else {"scale_factors": scale_factors}
    sdata_ref = _sdata_image(image, "img", **kwargs)
    sdata_query = _sdata_image(np.roll(image, 1, axis=1), "img", **kwargs)

    result = align_stalign_image(sdata_ref, sdata_query, image_key="img", **IMAGE_SOLVER)
    assert result.ref_axes[0].shape == (32,)

    out = align_stalign_image(sdata_ref, sdata_query, image_key="img", key_added="warped", **IMAGE_SOLVER)
    assert out.images["warped"].shape == image.shape
    assert "warped" not in sdata_query.images, "the input must be left untouched"


def test_image_fit_reads_units_and_placement_off_the_elements() -> None:
    """The elements' own scale and translation are the units the fit runs in.

    Nothing is restated as a `*_scale` argument that could contradict the container, and
    the warped element -- which lands on the reference's grid -- inherits the reference's
    placement rather than a reconstructed one.
    """
    from spatialdata.transformations import Scale, Sequence, Translation, get_transformation

    image = np.random.default_rng(0).random((1, 24, 24))
    placement = Sequence([Scale([2.0, 2.0], axes=("y", "x")), Translation([100.0, 50.0], axes=("y", "x"))])
    sdata_ref = _sdata_image(image, "img", transformations={"global": placement})
    sdata_query = _sdata_image(np.roll(image, 1, axis=1), "img", transformations={"global": placement})

    result = align_stalign_image(sdata_ref, sdata_query, image_key="img", a=8.0, nt=1, niter=2, epV=1.0)

    assert float(result.ref_axes[0][1] - result.ref_axes[0][0]) == 2.0, "the element's scale is the unit"
    assert float(result.ref_axes[0][0]) == 100.0, "the element's translation is the origin"
    assert float(result.query_axes[1][0]) == 50.0

    out = align_stalign_image(sdata_ref, sdata_query, image_key="img", key_added="w", a=8.0, nt=1, niter=2, epV=1.0)
    axes = {"input_axes": ("y", "x"), "output_axes": ("y", "x")}
    np.testing.assert_allclose(
        get_transformation(out.images["w"], to_coordinate_system="global").to_affine_matrix(**axes),
        get_transformation(sdata_ref.images["img"], to_coordinate_system="global").to_affine_matrix(**axes),
    )


def test_slice_fit_places_a_section_in_a_volume() -> None:
    volume = np.random.default_rng(0).random((1, 6, 12, 12))
    sdata_ref = _sdata_image(volume, "volume")
    sdata_query = _sdata_image(volume[:, 3], "section")

    result = align_stalign_volume(
        sdata_ref, sdata_query, image_key=("volume", "section"), a=3.0, nt=1, niter=2, epV=1.0
    )

    assert result.affine_xyz.shape == (4, 4)
    assert result.transform(ALIGN_PTS).shape == (len(ALIGN_PTS), 3)


def test_a_2d_reference_is_rejected_with_a_pointer_to_the_2d_path() -> None:
    section = np.random.default_rng(0).random((1, 12, 12))
    sdata = _sdata_image(section, "section")

    with pytest.raises(ValueError, match=r"align_stalign_image"):
        align_stalign_volume(sdata, sdata, image_key=("section", "section"), **IMAGE_SOLVER)


def _sdata_section_with_table(shapes_scale: float | None) -> SpatialData:
    """A section plus a table whose coordinates live in a shapes element's intrinsic frame."""
    import geopandas as gpd
    from anndata import AnnData
    from shapely.geometry import Point
    from spatialdata.models import ShapesModel, TableModel
    from spatialdata.transformations import Scale, set_transformation

    coords = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    shapes = ShapesModel.parse(gpd.GeoDataFrame({"radius": [1.0] * 3}, geometry=[Point(*c) for c in coords]))
    if shapes_scale is not None:
        set_transformation(shapes, Scale([shapes_scale] * 2, axes=("x", "y")), to_coordinate_system="global")
    adata = AnnData(np.zeros((3, 1)), obs={"region": ["cells"] * 3, "id": list(range(3))})
    adata.obs["region"] = adata.obs["region"].astype("category")
    adata.obsm["spatial"] = coords.copy()

    sdata = _sdata_image(np.random.default_rng(0).random((1, 12, 12)), "section")
    sdata.shapes["cells"] = shapes
    sdata.tables["t"] = TableModel.parse(adata, region="cells", region_key="region", instance_key="id")
    return sdata


def test_writing_coords_from_a_different_frame_is_refused() -> None:
    """The fit's units come from the image element, so the coordinates must share that frame.

    A table's ``obsm`` sits in the intrinsic frame of the element it annotates. Applying the
    fit to it under a non-identity transform yields plausible-looking reference coordinates
    that are simply wrong, with nothing to reveal it -- so it raises instead.
    """
    sdata_ref = _sdata_image(np.random.default_rng(0).random((1, 6, 12, 12)), "volume")
    kwargs = {"image_key": ("volume", "section"), "table_key": "t", "spatial_key": "spatial", **VOLUME_SOLVER}

    with pytest.raises(ValueError, match=r"non-identity transformation into 'global'"):
        align_stalign_volume(sdata_ref, _sdata_section_with_table(10.0), key_added="ref_xyz", **kwargs)

    # identity frame writes, and returning the fit is allowed either way
    out = align_stalign_volume(sdata_ref, _sdata_section_with_table(None), key_added="ref_xyz", **kwargs)
    assert out.tables["t"].obsm["ref_xyz"].shape == (3, 3)
    assert align_stalign_volume(sdata_ref, _sdata_section_with_table(10.0), **kwargs) is not None


# --- landmarks on the image path ------------------------------------------------------

_LM_REF = np.array([[2.0, 3.0], [15.0, 4.0], [8.0, 16.0], [4.0, 12.0]])
_LM_QUERY = _LM_REF + np.array([1.5, -0.5])


def _pair() -> tuple[np.ndarray, np.ndarray]:
    image = np.random.default_rng(0).random((1, 20, 20))
    return image, np.roll(image, 1, axis=1)


def test_image_landmarks_reach_the_solver() -> None:
    """The point-matching term the solver weights by ``sigmaP`` must actually change the fit."""
    from squidpy.experimental.tl._align._stalign import fit_stalign_image

    ref, query = _pair()
    solver = {"a": 4.0, "nt": 1, "niter": 3, "epV": 1.0}

    plain = fit_stalign_image(ref, query, **solver)
    with_landmarks = fit_stalign_image(ref, query, landmarks_ref=_LM_REF, landmarks_query=_LM_QUERY, **solver)

    assert not np.allclose(plain.affine, with_landmarks.affine)


def test_landmarks_and_initial_affine_are_not_exclusive() -> None:
    """Landmarks have two roles; only one of them collides with ``initial_affine``.

    They always contribute the matching term, and *also* derive the starting affine when
    ``initial_affine`` is absent. Passing both keeps the term and pins the start -- which
    is what a fit that supplies its own L/T alongside points needs.
    """
    from squidpy.experimental.tl._align._stalign import fit_stalign_image, fit_stalign_obs

    ref, query = _pair()
    solver = {"a": 4.0, "nt": 1, "niter": 3, "epV": 1.0}
    landmarks = {"landmarks_ref": _LM_REF, "landmarks_query": _LM_QUERY}

    pinned = fit_stalign_image(ref, query, initial_affine=np.eye(3), **landmarks, **solver)
    derived = fit_stalign_image(ref, query, **landmarks, **solver)
    assert not np.allclose(pinned.affine, derived.affine), "the given affine must win over the derived one"

    # the point-cloud path shares the resolution, so it accepts the same combination
    assert fit_stalign_obs(ALIGN_PTS, ALIGN_PTS + 0.4, initial_affine=np.eye(3), **landmarks, **TINY_SOLVER)


@pytest.mark.parametrize("missing", ["landmarks_ref", "landmarks_query"])
def test_one_sided_landmarks_are_refused(missing: str) -> None:
    from squidpy.experimental.tl._align._stalign import fit_stalign_image

    ref, query = _pair()
    given = {"landmarks_ref": _LM_REF, "landmarks_query": _LM_QUERY}
    del given[missing]

    with pytest.raises(ValueError, match=r"both landmark arrays"):
        fit_stalign_image(ref, query, **given, a=4.0, nt=1, niter=1, epV=1.0)


def test_align_stalign_image_forwards_landmarks() -> None:
    ref, query = _pair()
    sdata_ref, sdata_query = _sdata_image(ref, "img"), _sdata_image(query, "img")
    solver = {"image_key": "img", **IMAGE_SOLVER}

    plain = align_stalign_image(sdata_ref, sdata_query, **solver)
    with_landmarks = align_stalign_image(
        sdata_ref, sdata_query, landmarks_ref=_LM_REF, landmarks_query=_LM_QUERY, **solver
    )

    assert not np.allclose(plain.affine, with_landmarks.affine)


# --- deformation_grid at rank 3 -------------------------------------------------------


def test_volume_deformation_grid_is_the_transform_the_objective_uses() -> None:
    """Not an approximation for plotting: the same call on the same fitted state.

    Asserted as bit-for-bit equality, since that is what the docstring promises and what a
    comparison against an external 3D transform needs to be able to rely on.
    """
    import jax.numpy as jnp

    from squidpy.experimental.tl._align._stalign_impl._core import jax_dtype, transform_grid_row_col

    volume = np.random.default_rng(0).random((1, 5, 12, 12))
    result = align_stalign_volume(
        _sdata_image(volume, "volume"),
        _sdata_image(volume[:, 2], "section"),
        image_key=("volume", "section"),
        **VOLUME_SOLVER,
    )

    backward = result.deformation_grid()
    assert backward.shape == (3, 1, 12, 12), "the section is lifted onto z = 0, hence the length-1 z"
    assert result.deformation_grid(direction="forward").shape == (3, 5, 12, 12)

    internal = transform_grid_row_col(
        (jnp.zeros(1, dtype=jax_dtype()), *result.query_axes),
        result.velocity_grid,
        result.velocity,
        result.affine,
        direction="backward",
    )
    assert jnp.array_equal(backward, internal)


def test_volume_deformation_grid_rejects_a_bad_direction() -> None:
    volume = np.random.default_rng(0).random((1, 5, 12, 12))
    result = align_stalign_volume(
        _sdata_image(volume, "volume"),
        _sdata_image(volume[:, 2], "section"),
        image_key=("volume", "section"),
        **VOLUME_SOLVER,
    )
    with pytest.raises(ValueError, match=r"'forward' or 'backward'"):
        result.deformation_grid(direction="sideways")
