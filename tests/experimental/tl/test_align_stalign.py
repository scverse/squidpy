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

from squidpy.experimental.tl import (  # noqa: E402
    stalign_affine_xyz,
    stalign_align_image,
    stalign_align_obs,
    stalign_align_volume,
    stalign_apply_transform,
    stalign_apply_warp,
    stalign_deformation_grid,
    stalign_from_uns,
    stalign_to_uns,
    stalign_transform_points,
)

IMAGE_SOLVER = {"a": 4.0, "nt": 1, "niter": 2, "epV": 1.0}
VOLUME_SOLVER = {"a": 3.0, "nt": 1, "niter": 2, "epV": 1.0}


def _sdata_image(array: np.ndarray, key: str, **kwargs: object) -> SpatialData:
    model = Image3DModel if array.ndim == 4 else Image2DModel
    dims = ("c", "z", "y", "x") if array.ndim == 4 else ("c", "y", "x")
    return SpatialData(images={key: model.parse(array, dims=dims, **kwargs)})


def test_obs_fit_returns_a_result_and_applies_to_a_copy() -> None:
    ref, query = make_adata(ALIGN_PTS), make_adata(ALIGN_PTS + 0.4)

    result = stalign_align_obs(ref, query, **TINY_SOLVER)
    assert result["aligned_points"].shape == ALIGN_PTS.shape

    handed_back = stalign_apply_transform(result, query, inplace=False)
    assert handed_back.shape == ALIGN_PTS.shape
    assert "spatial_aligned" not in query.obsm, "`inplace=False` must leave the input untouched"

    # one fit, applied more than once, and writing agrees with what was handed back
    assert stalign_apply_transform(result, query) is None
    np.testing.assert_array_equal(query.obsm["spatial_aligned"], handed_back)
    assert stalign_apply_transform(result, query, key_added="elsewhere") is None
    np.testing.assert_array_equal(query.obsm["elsewhere"], handed_back)


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

    result = stalign_align_image(sdata_ref, sdata_query, image_key="img", **IMAGE_SOLVER)
    assert result["ref_axes"][0].shape == (32,)

    handed_back = stalign_apply_warp(result, sdata_ref, sdata_query, image_key="img", inplace=False)
    assert handed_back.shape == image.shape
    assert not sdata_query.images.keys() - {"img"}, "`inplace=False` must leave the input untouched"

    # the default key derives from the element it warps, the way `sc.tl.dendrogram`'s does
    assert stalign_apply_warp(result, sdata_ref, sdata_query, image_key="img") is None
    np.testing.assert_array_equal(np.asarray(sdata_query.images["img_aligned"]), handed_back)


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

    result = stalign_align_image(sdata_ref, sdata_query, image_key="img", a=8.0, nt=1, niter=2, epV=1.0)

    assert float(result["ref_axes"][0][1] - result["ref_axes"][0][0]) == 2.0, "the element's scale is the unit"
    assert float(result["ref_axes"][0][0]) == 100.0, "the element's translation is the origin"
    assert float(result["query_axes"][1][0]) == 50.0

    assert stalign_apply_warp(result, sdata_ref, sdata_query, image_key="img", key_added="w") is None
    axes = {"input_axes": ("y", "x"), "output_axes": ("y", "x")}
    np.testing.assert_allclose(
        get_transformation(sdata_query.images["w"], to_coordinate_system="global").to_affine_matrix(**axes),
        get_transformation(sdata_ref.images["img"], to_coordinate_system="global").to_affine_matrix(**axes),
    )


def test_slice_fit_places_a_section_in_a_volume() -> None:
    volume = np.random.default_rng(0).random((1, 6, 12, 12))
    sdata_ref = _sdata_image(volume, "volume")
    sdata_query = _sdata_image(volume[:, 3], "section")

    result = stalign_align_volume(
        sdata_ref, sdata_query, image_key=("volume", "section"), a=3.0, nt=1, niter=2, epV=1.0
    )

    assert stalign_affine_xyz(result).shape == (4, 4)
    assert stalign_transform_points(result, ALIGN_PTS).shape == (len(ALIGN_PTS), 3)


def test_a_2d_reference_is_rejected_with_a_pointer_to_the_2d_path() -> None:
    section = np.random.default_rng(0).random((1, 12, 12))
    sdata = _sdata_image(section, "section")

    with pytest.raises(ValueError, match=r"stalign_align_image"):
        stalign_align_volume(sdata, sdata, image_key=("section", "section"), **IMAGE_SOLVER)


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
    kwargs = {"image_key": ("volume", "section"), **VOLUME_SOLVER}
    applied = {"table_key": "t", "spatial_key": "spatial", "key_added": "ref_xyz"}

    # the fit itself is indifferent to the frame; only applying it to `obsm` is not
    skewed = _sdata_section_with_table(10.0)
    fit = stalign_align_volume(sdata_ref, skewed, **kwargs)
    with pytest.raises(ValueError, match=r"non-identity transformation into 'global'"):
        stalign_apply_transform(fit, skewed, **applied)

    identity = _sdata_section_with_table(None)
    out = stalign_apply_transform(stalign_align_volume(sdata_ref, identity, **kwargs), identity, **applied)
    assert out is None, "`inplace` defaults to True"
    assert identity.tables["t"].obsm["ref_xyz"].shape == (3, 3)


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

    assert not np.allclose(plain["affine"], with_landmarks["affine"])


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
    assert not np.allclose(pinned["affine"], derived["affine"]), "the given affine must win over the derived one"

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


def test_stalign_align_image_forwards_landmarks() -> None:
    ref, query = _pair()
    sdata_ref, sdata_query = _sdata_image(ref, "img"), _sdata_image(query, "img")
    solver = {"image_key": "img", **IMAGE_SOLVER}

    plain = stalign_align_image(sdata_ref, sdata_query, **solver)
    with_landmarks = stalign_align_image(
        sdata_ref, sdata_query, landmarks_ref=_LM_REF, landmarks_query=_LM_QUERY, **solver
    )

    assert not np.allclose(plain["affine"], with_landmarks["affine"])


# --- storing a fit --------------------------------------------------------------------


def test_a_stored_fit_survives_a_zarr_round_trip_and_still_transforms(tmp_path) -> None:
    """The point of storing it: `uns` is the only place either container has for a fit.

    Asserted through an actual write, because the trap is on that path and not in memory:
    anndata has no writer for a tuple, and a *list* of the axes only survives when they
    happen to be equal length -- a non-square raster fails. Hence the mapping, and hence a
    12x9 raster here rather than a square one.
    """
    import anndata as ad

    volume = np.random.default_rng(0).random((1, 5, 12, 9))
    fit = stalign_align_volume(
        _sdata_image(volume, "volume"),
        _sdata_image(volume[:, 2], "section"),
        image_key=("volume", "section"),
        **VOLUME_SOLVER,
    )
    adata = make_adata(ALIGN_PTS)
    assert stalign_to_uns(fit, adata) is None

    path = tmp_path / "a.zarr"
    adata.write_zarr(path)
    restored = stalign_from_uns(ad.read_zarr(path))

    assert restored["rank"] == 3
    assert len(restored["velocity_grid"]) == 3
    assert [a.shape for a in restored["ref_axes"]] == [a.shape for a in fit["ref_axes"]]
    # the ragged pair is what a list would have destroyed
    assert restored["query_axes"][0].shape != restored["query_axes"][1].shape
    np.testing.assert_allclose(
        np.asarray(stalign_transform_points(restored, ALIGN_PTS)),
        np.asarray(stalign_transform_points(fit, ALIGN_PTS)),
        rtol=0,
        atol=1e-6,
    )


def test_reading_a_key_that_is_not_a_fit_says_so() -> None:
    adata = make_adata(ALIGN_PTS)
    adata.uns["junk"] = {"affine": np.eye(3)}
    with pytest.raises(ValueError, match=r"carries no `rank`"):
        stalign_from_uns(adata, "junk")
    with pytest.raises(KeyError, match=r"no `uns\['missing'\]`"):
        stalign_from_uns(adata, "missing")


# --- deformation_grid at rank 3 -------------------------------------------------------


def test_volume_deformation_grid_is_the_transform_the_objective_uses() -> None:
    """Not an approximation for plotting: the same call on the same fitted state.

    Asserted as bit-for-bit equality, since that is what the docstring promises and what a
    comparison against an external 3D transform needs to be able to rely on.
    """
    import jax.numpy as jnp

    from squidpy.experimental.tl._align._stalign_impl._core import jax_dtype, transform_grid_row_col

    volume = np.random.default_rng(0).random((1, 5, 12, 12))
    result = stalign_align_volume(
        _sdata_image(volume, "volume"),
        _sdata_image(volume[:, 2], "section"),
        image_key=("volume", "section"),
        **VOLUME_SOLVER,
    )

    backward = stalign_deformation_grid(result)
    assert backward.shape == (3, 1, 12, 12), "the section is lifted onto z = 0, hence the length-1 z"
    assert stalign_deformation_grid(result, direction="forward").shape == (3, 5, 12, 12)

    internal = transform_grid_row_col(
        (jnp.zeros(1, dtype=jax_dtype()), *result["query_axes"]),
        result["velocity_grid"],
        result["velocity"],
        result["affine"],
        direction="backward",
    )
    assert jnp.array_equal(backward, internal)


def test_volume_deformation_grid_rejects_a_bad_direction() -> None:
    volume = np.random.default_rng(0).random((1, 5, 12, 12))
    result = stalign_align_volume(
        _sdata_image(volume, "volume"),
        _sdata_image(volume[:, 2], "section"),
        image_key=("volume", "section"),
        **VOLUME_SOLVER,
    )
    with pytest.raises(ValueError, match=r"'forward' or 'backward'"):
        stalign_deformation_grid(result, direction="sideways")
