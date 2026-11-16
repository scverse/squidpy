"""STalign port vs the original, on outputs upstream computed (scverse/squidpy#1243).

The reference values come from squidpy-ports' generator, published there as an AnnData in a zarr store;
https://theislab.github.io/squidpy-ports/reference-bundle.html says what it holds and why.
Rank 2 is section-to-section (upstream ``LDDMM``), rank 3 section-into-volume
(``LDDMM_3D_to_slice``). Needs ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

import importlib
import os
from typing import NamedTuple

import numpy as np
import pytest

# the jax CI env sets JAX_ENABLE_X64, and there a missing jax must fail rather than skip
jax = importlib.import_module("jax") if os.environ.get("JAX_ENABLE_X64") else pytest.importorskip("jax")
if not jax.config.jax_enable_x64:
    pytest.skip("needs JAX_ENABLE_X64=1: upstream is float64 throughout", allow_module_level=True)
pytestmark = [pytest.mark.jax, pytest.mark.internet]

import anndata as ad  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from spatialdata import SpatialData  # noqa: E402

from squidpy.experimental.im import sample_volume  # noqa: E402
from squidpy.experimental.tl import (  # noqa: E402
    StalignImageFit,
    StalignObsFit,
    StalignVolumeFit,
    stalign_align_image,
    stalign_align_obs,
    stalign_align_volume,
)

# private: the public entry points rebuild axes from an element's placement, which is not
# bit-exact for every grid, and the kernel is pinned here at ~1e-15
from squidpy.experimental.tl._align._stalign import fit_stalign_image, fit_stalign_volume  # noqa: E402
from squidpy.experimental.tl._align._stalign_impl import _core, _helpers  # noqa: E402

_BUNDLE_URL = (
    "https://github.com/theislab/squidpy-ports/releases/download/stalign-reference-v3/stalign_reference.zarr.zip"
)
_BUNDLE_SHA256 = "e5add8445590c9da2b5e42631f25b10f7838fb2b694b8438c7d762f49a899702"


class Bundle(NamedTuple):
    arrays: dict[str, dict[str, np.ndarray]]
    provenance: dict[str, dict]
    fixtures: dict


@pytest.fixture(scope="session")
def bundle() -> Bundle:
    """The reference bundle, downloaded once and read into memory."""
    import anndata as ad
    import pooch
    from scanpy import settings
    from zarr.storage import ZipStore

    path = pooch.retrieve(
        _BUNDLE_URL, known_hash=f"sha256:{_BUNDLE_SHA256}", path=settings.datasetdir, progressbar=False
    )
    with ZipStore(path, mode="r") as store:
        uns = dict(ad.read_zarr(store).uns)
    fixtures = uns.pop("fixtures")
    arrays = {
        name: {key: value for key, value in section.items() if key != "provenance"} for name, section in uns.items()
    }
    return Bundle(arrays, {name: section["provenance"] for name, section in uns.items()}, fixtures)


@pytest.fixture(scope="session")
def params(bundle) -> dict:
    """Every parameter the bundle's sections were generated with."""
    return bundle.fixtures


EXACT = 1e-12

_SOLVER_TAIL = {"muA": None, "muB": None, "tol": None, "patience": 25}


#: Upstream's result keys, and the ``lddmm`` / ``StalignFit`` field each one is compared to.
_FIELDS = {
    "A": "affine",
    "v": "velocity",
    "WM": "match_weights",
    "WA": "artifact_weights",
    "WB": "background_weights",
}


def rel(actual, expected) -> float:
    """Relative L2 error, the single measure used throughout."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    denominator = np.linalg.norm(expected)
    return float(np.linalg.norm(actual - expected) / max(denominator, np.finfo(float).tiny))


@pytest.fixture(scope="session")
def primitives(bundle):
    return bundle.arrays["primitives"]


@pytest.fixture(scope="session")
def source_grid(primitives):
    """The *moving* raster and its axes, as ``((y, x), image)``."""
    return (jnp.asarray(primitives["raster_query_y"]), jnp.asarray(primitives["raster_query_x"])), jnp.asarray(
        primitives["raster_query"]
    )


@pytest.fixture(scope="session")
def target_grid(primitives):
    """The *fixed* raster and its axes: the reference."""
    return (jnp.asarray(primitives["raster_ref_y"]), jnp.asarray(primitives["raster_ref_x"])), jnp.asarray(
        primitives["raster_ref"]
    )


@pytest.fixture(scope="session")
def velocity_grid(primitives):
    """Upstream's own velocity grid, which ``_lddmm_loss`` accepts directly."""
    return jnp.asarray(primitives["xv_upstream_0"]), jnp.asarray(primitives["xv_upstream_1"])


@pytest.fixture(scope="session")
def fitted(primitives, velocity_grid):
    """A :class:`StalignObsFit` carrying upstream's own fitted deformation."""
    return StalignObsFit(
        affine=jnp.asarray(primitives["to_A"]),
        velocity=jnp.asarray(primitives["velocity"]),
        velocity_grid=velocity_grid,
        aligned_points=jnp.zeros((0, 2)),
    )


def _transform_rc(result: StalignObsFit, points_rc, *, inverse: bool = False) -> np.ndarray:
    """``transform_points`` (or its inverse) on row-col points, for comparison with upstream."""
    method = result.inverse_transform_points if inverse else result.transform_points
    got = method(np.asarray(points_rc)[:, ::-1])
    return np.asarray(got)[:, ::-1]


def test_every_section_carries_provenance(bundle):
    """Every section records which upstream and which generator run produced it."""
    assert len(bundle.provenance) == 11
    for name, payload in bundle.provenance.items():
        assert payload["upstream_sha"] == "b2068edc98974efa54537eca194736e177bbe11d", name
        for key in ("ports_commit", "torch", "numpy", "python", "platform", "generated_utc"):
            assert payload[key], f"{name}: empty provenance field {key!r}"


def test_fixture_samples_are_off_grid(primitives):
    """No interpolation sample may sit on a grid line."""
    coords = np.asarray(primitives["interp_coords"])
    axes = (np.asarray(primitives["raster_ref_y"]), np.asarray(primitives["raster_ref_x"]))
    for axis, values in zip(axes, coords, strict=True):
        fractional = np.abs(np.modf((values - axis[0]) / (axis[1] - axis[0]))[0])
        assert np.minimum(fractional, 1.0 - fractional).min() > 1e-6


def test_fixture_stays_inside_velocity_grid(primitives, velocity_grid):
    """Keeps the out-of-domain padding difference out of every test that is not about it."""
    points = np.asarray(primitives["points"])
    for axis, column in zip(velocity_grid, points.T, strict=True):
        assert column.min() > float(axis[0]) and column.max() < float(axis[-1])


def test_interp_matches_upstream(primitives, source_grid):
    """``_interp`` vs ``STalign.interp(padding_mode='border')``."""
    axes, image = source_grid
    got = _core.interp(axes, image, jnp.asarray(primitives["interp_coords"]))
    error = rel(got, primitives["interp_border"])
    assert error < EXACT


def test_regularizer_matches_upstream(primitives, velocity_grid, params):
    """``LL``/``K``/``DV`` vs STalign.py:1078-1090."""
    kernel, ll, dv_prod = _core._build_regularizer(
        velocity_grid, a=params["LDDMM_PARAMS"]["a"], p=params["LDDMM_PARAMS"]["p"]
    )
    for name, got, expected in (
        ("LL", ll, primitives["regularizer_LL"]),
        ("K", kernel, primitives["regularizer_K"]),
        ("DV", dv_prod, primitives["regularizer_DV"]),
    ):
        error = rel(got, expected)
        assert error < EXACT, name


def test_transform_grid_backward_matches_upstream(primitives, source_grid, target_grid, velocity_grid):
    """The reference grid in the query frame, the one ``warp_image`` samples through, vs ``STalign.build_transform('b')``."""
    axes, _ = target_grid
    got = _core.transform_grid_row_col(
        axes, velocity_grid, jnp.asarray(primitives["velocity"]), jnp.asarray(primitives["to_A"]), direction="backward"
    )
    error = rel(np.moveaxis(np.asarray(got), 0, -1), primitives["grid_backward"])
    assert error < EXACT


def test_warp_image_uses_the_upstream_grid(primitives, source_grid, target_grid, velocity_grid):
    """``warp_image`` vs sampling on upstream's own backward grid."""
    source_axes, source_image = source_grid
    target_axes, _ = target_grid
    result = StalignImageFit(
        affine=jnp.asarray(primitives["to_A"]),
        velocity=jnp.asarray(primitives["velocity"]),
        velocity_grid=velocity_grid,
        query_axes=source_axes,
        ref_axes=target_axes,
    )
    upstream_grid = jnp.asarray(np.moveaxis(np.asarray(primitives["grid_backward"]), -1, 0))
    expected = _core.interp(source_axes, source_image, upstream_grid)

    error = rel(result.warp_image(source_image), expected)
    assert error < EXACT


def test_transform_points_forward_matches_upstream(fitted, primitives):
    """``transform_points`` at rank 2 vs ``STalign.transform_points_source_to_target``."""
    got = _transform_rc(fitted, primitives["points"])
    error = rel(got, primitives["points_forward"])
    assert error < EXACT


def test_to_affine_matches_upstream(primitives):
    """``_to_affine`` vs ``STalign.to_A``."""
    got = _core._to_affine(jnp.asarray(primitives["to_A_linear"]), jnp.asarray(primitives["to_A_translation"]))
    np.testing.assert_allclose(np.asarray(got), primitives["to_A"], rtol=0, atol=0)


def _loss_arguments(primitives, source_grid, target_grid, velocity_grid, params, *, nt, with_points):
    source_axes, source_image = source_grid
    target_axes, target_image = target_grid
    kernel, ll, dv_prod = _core._build_regularizer(
        velocity_grid, a=params["LDDMM_PARAMS"]["a"], p=params["LDDMM_PARAMS"]["p"]
    )
    empty = jnp.zeros((0, 2))
    landmarks_source = jnp.asarray(primitives["landmarks_query"])[:, ::-1]
    landmarks_target = jnp.asarray(primitives["landmarks_ref"])[:, ::-1]
    return (
        kernel,
        {
            "x_source": source_axes,
            "source_image": source_image,
            "x_target": target_axes,
            "target_image": target_image,
            "xv": velocity_grid,
            # LDDMM's initial state: uniform 0.5 matching weights (STalign.py:1102).
            "match_weights": jnp.full(target_image.shape[1:], 0.5),
            "ll": ll,
            "dv_prod": dv_prod,
            "points_source": landmarks_source if with_points else empty,
            "points_target": landmarks_target if with_points else empty,
            "sigmaM": params["LDDMM_PARAMS"]["sigmaM"],
            "sigmaR": params["LDDMM_PARAMS"]["sigmaR"],
            "sigmaP": params["LDDMM_PARAMS"]["sigmaP"],
        },
        jnp.zeros((nt, velocity_grid[0].shape[0], velocity_grid[1].shape[0], 2)),
    )


@pytest.mark.parametrize("nt", [1, 3])
@pytest.mark.parametrize("with_points", [False, True])
@pytest.mark.parametrize("warm", [False, True], ids=["v_zero", "v_nonzero"])
def test_energy_matches_upstream(
    primitives, source_grid, target_grid, velocity_grid, nt, with_points, warm, bundle, params
):
    """``_lddmm_loss`` vs upstream's ``E`` at iteration 0 -- the two optimise the same function."""
    energy = bundle.arrays["energy"]
    _, kwargs, velocity = _loss_arguments(
        primitives, source_grid, target_grid, velocity_grid, nt=nt, with_points=with_points, params=params
    )
    key = f"E_nt{nt}_{'points' if with_points else 'nopoints'}"
    if warm:
        velocity = jnp.asarray(energy[f"warm_velocity_nt{nt}"])
        key += "_warmv"

    total, _ = _core._lddmm_loss(jnp.asarray(energy["L"]), jnp.asarray(energy["T"]), velocity, **kwargs)
    error = rel(total, energy[key])
    assert error < EXACT, f"squidpy {float(total)!r} vs upstream {float(energy[key])!r}"


def test_energy_budget_is_not_vacuous(primitives, source_grid, target_grid, velocity_grid, bundle, params):
    """A tolerance that a perturbed run also passes is decoration, not a test."""
    energy = bundle.arrays["energy"]
    _, kwargs, velocity = _loss_arguments(
        primitives, source_grid, target_grid, velocity_grid, nt=3, with_points=True, params=params
    )
    kwargs["sigmaM"] *= 1.0001
    total, _ = _core._lddmm_loss(jnp.asarray(energy["L"]), jnp.asarray(energy["T"]), velocity, **kwargs)
    assert rel(total, energy["E_nt3_points"]) > EXACT


@pytest.fixture(scope="session", params=[False, True], ids=["v_zero", "v_nonzero"])
def measured_gradients(request, primitives, source_grid, target_grid, velocity_grid, bundle, params):
    """Gradients of ``_lddmm_loss``, at ``v = 0`` and at a non-zero velocity."""
    gradients = bundle.arrays["gradients"]
    kernel, kwargs, velocity = _loss_arguments(
        primitives,
        source_grid,
        target_grid,
        velocity_grid,
        nt=params["LDDMM_PARAMS"]["nt"],
        with_points=True,
        params=params,
    )
    suffix = ""
    if request.param:
        velocity = jnp.asarray(gradients["warm_velocity"])
        suffix = "_warmv"

    (_, _), (grad_l, grad_t, grad_v) = jax.value_and_grad(_core._lddmm_loss, argnums=(0, 1, 2), has_aux=True)(
        jnp.asarray(gradients["L"]), jnp.asarray(gradients["T"]), velocity, **kwargs
    )
    # Upstream stores the Sobolev-smoothed velocity gradient, since that is what actually
    # drives the step (STalign.py:1215).
    smoothed = jnp.fft.ifftn(jnp.fft.fftn(grad_v, axes=(1, 2)) * kernel[None, ..., None], axes=(1, 2)).real
    return gradients, {"L": grad_l, "T": grad_t, "v": smoothed}, suffix


@pytest.mark.parametrize(("component", "key"), [("L", "grad_L"), ("T", "grad_T"), ("v", "grad_v_smoothed")])
def test_gradients_match_upstream(measured_gradients, component, key):
    """``dE/dL``, ``dE/dT`` and the smoothed ``dE/dv`` vs upstream's, at iteration 0."""
    gradients, computed, suffix = measured_gradients
    error = rel(computed[component], gradients[key + suffix])
    assert error < EXACT


def test_contrast_transform_freezes_the_ridge_coefficients(params):
    """Regression guard, stated without reference to upstream."""
    rng = np.random.default_rng(params["SEED"])
    warped = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    target = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    weights = jnp.asarray(rng.uniform(0.2, 0.8, size=(12, 15)))

    def live(x):
        return jnp.sum(_core._contrast_transform(x, target, weights) ** 2)

    def leaky(x):
        # _contrast_transform without the stop_gradient
        flat_source = x.reshape(x.shape[0], -1)
        flat_target = target.reshape(target.shape[0], -1)
        design = jnp.concatenate((jnp.ones((1, flat_source.shape[1]), dtype=x.dtype), flat_source), axis=0)
        weighted = design * weights.reshape(-1)[None, :]
        coefficients = jnp.linalg.solve(
            weighted @ design.T + 0.1 * jnp.eye(design.shape[0], dtype=x.dtype),
            weighted @ flat_target.T,
        )
        return jnp.sum((coefficients.T @ design).reshape(target.shape) ** 2)

    np.testing.assert_allclose(float(live(warped)), float(leaky(warped)), rtol=1e-12)
    error = rel(jax.grad(live)(warped), jax.grad(leaky)(warped))
    assert error > 1e-6, "the ridge coefficients are being differentiated through again"


def test_rasterize_grid_matches_upstream(primitives, params):
    """The raster axes are identical to upstream's."""
    grid_x, grid_y, _ = _helpers.rasterize(primitives["ref"][:, 0], primitives["ref"][:, 1], **params["RASTER_PARAMS"])
    np.testing.assert_allclose(grid_x, primitives["raster_ref_x"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(grid_y, primitives["raster_ref_y"], rtol=0, atol=1e-9)


def test_velocity_grid_matches_upstream(primitives, source_grid, params):
    """The velocity grid has upstream's length, without its off-by-one."""
    axes, _ = source_grid
    built = _core._build_velocity_grid(axes, a=params["LDDMM_PARAMS"]["a"], expand=params["LDDMM_PARAMS"]["expand"])
    np.testing.assert_allclose(np.asarray(built[0]), primitives["xv_upstream_0"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(built[1]), primitives["xv_upstream_1"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("step", [30.0, 7.3, 0.017])
def test_grid_length_is_stable_against_float_rounding(step):
    """``np.arange(lo, hi + step, step)`` had an unstable length; the grid does not."""
    from squidpy.experimental.im._rasterize_points import axis

    assert axis is _core.stable_axis
    for start, n in ((-400.4123, 33), (0.1, 41), (-1234.567, 77)):
        assert axis(start, start + n * step, step).size == n


@pytest.mark.xfail(
    strict=True,
    reason=(
        "squidpy integrates the backward flow in reversed time order, "
        "upstream does not (STalign.py:1828-1843). squidpy is correct -- see "
        "test_backward_transform_inverts_better, which is the assertion that matters. "
        "This xfail exists to pin the literal difference, and should be deleted only if "
        "squidpy ever deliberately adopts upstream's ordering."
    ),
)
def test_transform_points_backward_matches_upstream(fitted, primitives):
    got = _transform_rc(fitted, primitives["points"], inverse=True)
    assert rel(got, primitives["points_backward"]) < EXACT


def test_backward_transform_inverts_better(fitted, primitives):
    """squidpy's backward map is the better inverse of the forward one."""
    points = np.asarray(primitives["points"])
    roundtrip = _transform_rc(fitted, primitives["points_forward"], inverse=True)
    ours = rel(roundtrip, points)
    theirs = rel(primitives["points_roundtrip"], points)
    assert ours <= theirs


@pytest.mark.xfail(
    strict=True,
    reason=(
        "upstream samples the velocity field with grid_sample's default "
        "padding_mode='zeros' (STalign.py:1163, :1167), squidpy uses "
        "map_coordinates(mode='nearest'). squidpy is correct -- zeros make a point that "
        "drifts off the velocity grid snap to no displacement at all."
    ),
)
def test_interp_outside_domain_matches_upstream(primitives, source_grid):
    axes, image = source_grid
    got = _core.interp(axes, image, jnp.asarray(primitives["interp_coords_outside"]))
    assert rel(got, primitives["interp_zeros_outside"]) < EXACT


def test_interp_outside_domain_is_border_padding(primitives, source_grid):
    """Outside the domain squidpy behaves exactly like upstream's 'border' mode."""
    axes, image = source_grid
    got = _core.interp(axes, image, jnp.asarray(primitives["interp_coords_outside"]))
    error = rel(got, primitives["interp_border_outside"])
    assert error < EXACT
    assert rel(got, primitives["interp_zeros_outside"]) > 0.1


RASTER_BUDGET = 0.06


def test_rasterize_stays_within_budget(primitives, params):
    """Rasterisation is a deliberate speedup, so it gets a measured budget rather than equality."""
    _, _, got = _helpers.rasterize(primitives["ref"][:, 0], primitives["ref"][:, 1], **params["RASTER_PARAMS"])
    expected = primitives["raster_ref"]
    got = np.asarray(got)
    assert got.shape == expected.shape

    for index, blur in enumerate(params["RASTER_PARAMS"]["blur"]):
        error = rel(got[index], expected[index])
        correlation = float(np.corrcoef(got[index].ravel(), expected[index].ravel())[0, 1])
        assert error < RASTER_BUDGET, f"blur={blur}: relL2 {error:.4%} exceeds {RASTER_BUDGET:.0%}"
        assert correlation > 0.99, f"blur={blur}: correlation {correlation:.5f}"


def test_rasterize_conserves_mass(primitives, params):
    """Every point contributes exactly one unit, wherever it sits."""
    n_points = primitives["ref"].shape[0]
    _, _, got = _helpers.rasterize(primitives["ref"][:, 0], primitives["ref"][:, 1], **params["RASTER_PARAMS"])

    for index, blur in enumerate(params["RASTER_PARAMS"]["blur"]):
        mass = float(np.asarray(got)[index].sum())
        assert mass == pytest.approx(n_points, rel=1e-9), f"blur={blur}: {mass:.2f} of {n_points}"


def _residual(linear, translation, source, target) -> float:
    source, target = np.asarray(source), np.asarray(target)
    return float(np.linalg.norm(source @ np.asarray(linear).T + np.asarray(translation) - target))


def test_affine_from_points_is_equivalent_when_well_conditioned(primitives, params):
    """The affine from landmarks: different estimators, equivalent when well conditioned."""
    source = np.asarray(primitives["landmarks_query"])[:, ::-1]
    target = np.asarray(primitives["landmarks_ref"])[:, ::-1]

    def wrap(points):
        """The points as the public API takes them: an ``obsm`` entry, stored verbatim."""
        adata = ad.AnnData(np.empty((np.asarray(points).shape[0], 0), dtype=float))
        adata.obsm["spatial"] = np.asarray(points, dtype=float)
        return adata

    fit = stalign_align_obs(
        wrap(primitives["ref"]),
        wrap(primitives["query"]),
        landmarks_query=primitives["landmarks_query"],
        landmarks_ref=primitives["landmarks_ref"],
        niter=0,
        dx=params["RASTER_PARAMS"]["dx"],
        blur=params["RASTER_PARAMS"]["blur"],
        raster_expand=params["RASTER_PARAMS"]["expand"],
    )
    affine = np.asarray(fit.affine)
    linear, translation = affine[:2, :2], affine[:2, 2]

    ours = _residual(linear, translation, source, target)
    theirs = _residual(primitives["lt_well_L"], primitives["lt_well_T"], source, target)
    assert abs(ours - theirs) / theirs < 1e-2, f"squidpy {ours:.6f} vs upstream {theirs:.6f}"


def test_affine_from_points_refuses_ill_conditioning(primitives):
    """On near-collinear landmarks upstream's ``inv(XᵀX)`` collapses; squidpy refuses."""
    source, target = primitives["ill_src"], primitives["ill_dst"]
    with pytest.raises(ValueError, match="not collinear"):
        _helpers.affine_from_points(jnp.asarray(source), jnp.asarray(target))


def _run_lddmm(primitives, source_grid, target_grid, snapshot, niter, params):
    return _core.lddmm(
        *source_grid,
        *target_grid,
        L=jnp.asarray(snapshot["L"]),
        T=jnp.asarray(snapshot["T"]),
        points_source=jnp.asarray(primitives["landmarks_query"])[:, ::-1],
        points_target=jnp.asarray(primitives["landmarks_ref"])[:, ::-1],
        niter=niter,
        **params["LDDMM_PARAMS"],
        **_SOLVER_TAIL,
    )


_TRAJECTORY_BUDGET = {1: 1e-10, 5: 1e-9, 50: 1e-6}


@pytest.mark.parametrize("niter", [1, 5, 50])
def test_trajectory_matches_upstream(primitives, niter, bundle, source_grid, target_grid, params):
    """Run the real loop for ``n`` steps and compare every state it carries."""
    snapshot = bundle.arrays[f"trajectory_n{niter}"]
    result = _run_lddmm(primitives, source_grid, target_grid, snapshot, niter, params=params)

    budget = _TRAJECTORY_BUDGET[niter]
    for name, field in _FIELDS.items():
        error = rel(result[field], snapshot[name])
        assert error < budget, f"n={niter} {name}: {error:.3e} exceeds {budget:.0e}"


def _placement(axis: np.ndarray) -> tuple[float, float]:
    """The ``(step, shift)`` of an evenly spaced ``axis``."""
    return float(axis[1] - axis[0]), float(axis[0])


def _as_sdata(array: np.ndarray, axes) -> SpatialData:
    """A one-image SpatialData placing ``array`` on ``axes``, as the public entry points read it."""
    from spatialdata.models import Image2DModel, Image3DModel
    from spatialdata.transformations import Scale, Translation
    from spatialdata.transformations import Sequence as TransformSequence

    spatial = ("z", "y", "x")[-len(axes) :]
    steps, shifts = zip(*(_placement(np.asarray(axis, dtype=float)) for axis in axes), strict=True)
    model = Image3DModel if len(axes) == 3 else Image2DModel
    transform = TransformSequence([Scale(list(steps), axes=spatial), Translation(list(shifts), axes=spatial)])
    element = model.parse(array, dims=("c", *spatial), transformations={"global": transform})
    return SpatialData(images={"image": element})


def _channels_first(image, *, ndim: int) -> np.ndarray:
    array = np.asarray(image, dtype=float)
    return array if array.ndim == ndim + 1 else array[None]


def _affine_xy(linear, translation) -> np.ndarray:
    """Upstream's row-col ``L``/``T`` as one homogeneous affine in ``(x, y[, z])``."""
    d = len(translation)
    affine = np.eye(d + 1)
    affine[:d, :d], affine[:d, d] = linear, translation
    swap = np.eye(d + 1)[[*reversed(range(d)), d]]
    return swap @ affine @ swap


@pytest.fixture(scope="session")
def image_reference(bundle):
    return bundle.arrays["image_trajectory"]


def _public_image_fit(fixture, params, *, swap_roles=False):
    """The image fit through the public ``stalign_align_image``, from SpatialData containers."""
    ax_moving = (fixture["source_axis_0"], fixture["source_axis_1"])
    ax_fixed = (fixture["target_axis_0"], fixture["target_axis_1"])
    moving = _as_sdata(_channels_first(fixture["query"], ndim=2), ax_moving)
    fixed = _as_sdata(_channels_first(fixture["ref"], ndim=2), ax_fixed)
    affine = _affine_xy(fixture["start_L"], fixture["start_T"])
    first, second = (moving, fixed) if swap_roles else (fixed, moving)
    return stalign_align_image(
        first, second, image_key="image", initial_affine=affine, niter=params["IMAGE_ITERS"], **params["IMAGE_PARAMS"]
    )


def test_public_image_call_reproduces_upstream(image_reference, params):
    """``stalign_align_image`` from SpatialData containers, against upstream's own result."""
    fit = _public_image_fit(image_reference, params=params)
    error = rel(fit.affine, image_reference["A"])
    assert error < 1e-10, f"the public call disagrees with upstream by {error:.3e}"


def test_public_image_call_is_role_sensitive(image_reference, params):
    """Swapping ref and query must be *loud*, not a 1e-5 wobble."""
    swapped = _public_image_fit(image_reference, swap_roles=True, params=params)
    assert rel(swapped.affine, image_reference["A"]) > 1e-3


@pytest.fixture(scope="session")
def image_mixed_units(bundle):
    """Source in one unit, target in another."""
    return bundle.arrays["image_mixed_units"]


def _fit_mixed_units(fixture, niter, params):
    """The image path on the mixed-unit axes, from the fixture's own ``a``."""
    params = {**params["IMAGE_PARAMS"], "a": float(fixture["a"])}
    return fit_stalign_image(
        fixture["ref"],
        fixture["query"],
        ref_axes=(fixture["target_axis_0"], fixture["target_axis_1"]),
        query_axes=(fixture["source_axis_0"], fixture["source_axis_1"]),
        niter=niter,
        **params,
    )


def test_mixed_unit_velocity_grid_matches_upstream(image_mixed_units, params):
    """The velocity grid upstream builds when the two sides disagree about units."""
    fit = _fit_mixed_units(image_mixed_units, 0, params=params)
    for axis, key in zip(fit.velocity_grid, ("xv_0", "xv_1"), strict=True):
        np.testing.assert_allclose(np.asarray(axis), image_mixed_units[key], rtol=0, atol=1e-9)


def test_mixed_unit_axes_are_actually_mixed(image_mixed_units, params):
    """The fixture is only meaningful while the two sides really carry different units."""
    source_step = float(image_mixed_units["source_axis_0"][1] - image_mixed_units["source_axis_0"][0])
    target_step = float(image_mixed_units["target_axis_0"][1] - image_mixed_units["target_axis_0"][0])
    assert source_step == pytest.approx(1.0)
    assert target_step == pytest.approx(params["MIXED_TARGET_STEP"])


def test_image_path_axes_match_the_reference(image_reference, params):
    """The image entry point's coordinate convention, checked before its results."""
    fit = fit_stalign_image(image_reference["ref"], image_reference["query"], niter=0, **params["IMAGE_PARAMS"])

    for got, expected in (
        (fit.query_axes[0], image_reference["source_axis_0"]),
        (fit.query_axes[1], image_reference["source_axis_1"]),
        (fit.ref_axes[0], image_reference["target_axis_0"]),
        (fit.ref_axes[1], image_reference["target_axis_1"]),
    ):
        np.testing.assert_allclose(np.asarray(got), expected, rtol=0, atol=1e-12)


def _run_image(fixture, params):
    """Drive the solver on the image fixture, from the reference's own starting affine."""
    source = (jnp.asarray(fixture["source_axis_0"]), jnp.asarray(fixture["source_axis_1"]))
    target = (jnp.asarray(fixture["target_axis_0"]), jnp.asarray(fixture["target_axis_1"]))
    return _core.lddmm(
        source,
        jnp.asarray(fixture["query"]),
        target,
        jnp.asarray(fixture["ref"]),
        L=fixture["start_L"],
        T=fixture["start_T"],
        niter=params["IMAGE_ITERS"],
        **params["IMAGE_PARAMS"],
        **_SOLVER_TAIL,
    )


@pytest.mark.parametrize("name", ["A", "v", "WM", "WA", "WB"])
def test_image_path_matches_upstream(image_reference, name, params):
    """The image entry point vs upstream's LDDMM on the same images, axes and start."""
    result = _run_image(image_reference, params=params)
    error = rel(result[_FIELDS[name]], image_reference[name])
    assert error < 1e-10, f"{name}: {error:.3e}"


def test_image_energy_trace_matches_upstream(image_reference, params):
    """Every iteration of the objective, not just the endpoint."""
    result = _run_image(image_reference, params=params)
    got = np.asarray(result["energies"])
    expected = np.asarray(image_reference["energies"])
    worst = float(np.max(np.abs(got - expected) / np.abs(expected)))
    assert worst < 1e-10


def test_image_warp_matches_upstream(image_reference, params):
    """``warp_image`` vs ``STalign.transform_image_source_to_target``."""
    result = _run_image(image_reference, params=params)
    fit = StalignImageFit(
        affine=result["affine"],
        velocity=result["velocity"],
        velocity_grid=result["velocity_grid"],
        query_axes=(jnp.asarray(image_reference["source_axis_0"]), jnp.asarray(image_reference["source_axis_1"])),
        ref_axes=(jnp.asarray(image_reference["target_axis_0"]), jnp.asarray(image_reference["target_axis_1"])),
    )
    error = rel(fit.warp_image(jnp.asarray(image_reference["query"])), image_reference["warped"])
    assert error < 1e-9


def test_on_grid_sampling_costs_six_orders_of_magnitude(bundle, params):
    """Sampling exactly on grid lines costs agreement, measured rather than asserted."""
    matched = bundle.arrays["image_trajectory_matched"]
    axes = (jnp.asarray(matched["axis_0"]), jnp.asarray(matched["axis_1"]))
    result = _core.lddmm(
        axes,
        jnp.asarray(matched["query"]),
        axes,
        jnp.asarray(matched["ref"]),
        L=matched["start_L"],
        T=matched["start_T"],
        niter=params["IMAGE_ITERS"],
        **params["IMAGE_PARAMS"],
        **_SOLVER_TAIL,
    )
    degraded = rel(result["velocity"], matched["v"])
    assert 1e-6 < degraded < 1e-1, (
        f"on-grid sampling now costs {degraded:.3e}; if this has become exact, the on-grid difference is gone "
        f"and the off-grid fixture design can be simplified"
    )


def test_converged_solution_matches_upstream(primitives, bundle, source_grid, target_grid, params):
    """500 iterations -- "enough to actually converge", per #1243."""
    snapshot = bundle.arrays["converged_n500"]
    result = _run_lddmm(primitives, source_grid, target_grid, snapshot, 500, params=params)

    energy = rel(result["energies"][result["n_iter"] - 1], snapshot["E_last"])
    assert energy < 0.01, f"final energy differs by {energy:.3%}"

    before = float(np.mean(np.linalg.norm(primitives["landmarks_query"] - primitives["landmarks_ref"], axis=1)))
    after = float(snapshot["tre_mean"])
    assert after < 0.4 * params["RASTER_PARAMS"]["dx"], f"reference did not converge: TRE {after:.2f}"
    assert after < before / 5.0, f"reference barely moved: TRE {before:.2f} -> {after:.2f}"

    converged = StalignObsFit(
        affine=result["affine"],
        velocity=result["velocity"],
        velocity_grid=result["velocity_grid"],
        aligned_points=jnp.zeros((0, 2)),
    )
    aligned = _transform_rc(converged, np.asarray(primitives["query"])[:, ::-1])
    displacement = np.linalg.norm(aligned - np.asarray(snapshot["aligned_points_rc"]), axis=1)
    percentile = float(np.percentile(displacement, 95))
    assert percentile < 0.1 * params["RASTER_PARAMS"]["dx"], (
        f"95th-percentile point disagreement {percentile:.4f} exceeds a tenth of a grid cell"
    )


@pytest.fixture(scope="session")
def slice_reference(bundle):
    return bundle.arrays["slice_trajectory"]


@pytest.fixture(scope="session")
def slice_velocity_grid(slice_reference):
    return tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3))


@pytest.fixture(scope="session")
def slice_axes(slice_reference):
    """``(reference (z, y, x) axes, section (y, x) axes)``, both as JAX arrays."""
    reference = tuple(jnp.asarray(slice_reference[f"ref_axis_{axis}"]) for axis in range(3))
    section = tuple(jnp.asarray(slice_reference[f"query_axis_{axis}"]) for axis in range(2))
    return reference, section


def test_slice_to_affine_matches_upstream(slice_reference):
    """``_to_affine`` at rank 3 vs ``STalign.to_A_3D``."""
    got = _core._to_affine(jnp.asarray(slice_reference["start_L"]), jnp.asarray(slice_reference["start_T"]))
    np.testing.assert_allclose(np.asarray(got), slice_reference["to_A"], rtol=0, atol=0)


def test_slice_velocity_grid_matches_upstream(slice_reference, slice_axes, params):
    """``_build_velocity_grid`` at rank 3 vs the grid upstream builds for itself."""
    reference, _ = slice_axes
    built = _core._build_velocity_grid(
        reference, a=params["SLICE_PARAMS"]["a"], expand=params["SLICE_PARAMS"]["expand"]
    )
    assert len(built) == 3
    for axis, got in enumerate(built):
        np.testing.assert_allclose(np.asarray(got), slice_reference[f"xv_{axis}"], rtol=0, atol=1e-9)


def test_slice_regularizer_matches_upstream(slice_reference, slice_velocity_grid, params):
    """``LL``/``K``/``DV`` at rank 3 vs STalign.py:1384-1397."""
    xv = slice_velocity_grid
    kernel, ll, dv_prod = _core._build_regularizer(xv, a=params["SLICE_PARAMS"]["a"], p=params["SLICE_PARAMS"]["p"])
    for name, got, expected in (
        ("LL", ll, slice_reference["regularizer_LL"]),
        ("K", kernel, slice_reference["regularizer_K"]),
        ("DV", dv_prod, slice_reference["regularizer_DV"]),
    ):
        error = rel(got, expected)
        assert error < EXACT, name


def test_slice_interp_matches_upstream(slice_reference, slice_axes):
    """``interp`` at rank 3 vs ``STalign.interp3D``."""
    reference, _ = slice_axes
    grid = jnp.asarray(np.moveaxis(np.asarray(slice_reference["grid_backward"]), -1, 0))
    got = _core.interp(reference, jnp.asarray(slice_reference["ref"]), grid)
    error = rel(got, slice_reference["interp_border"])
    assert error < EXACT


def test_slice_grid_backward_matches_upstream(slice_reference, slice_axes, slice_velocity_grid):
    """``deformation_grid`` at rank 3 vs ``build_transform3D``."""
    reference, section = slice_axes
    xv = slice_velocity_grid
    result = StalignVolumeFit(
        affine=jnp.asarray(slice_reference["A"]),
        velocity=jnp.asarray(slice_reference["velocity"]),
        velocity_grid=xv,
        ref_axes=reference,
        query_axes=section,
    )
    got = result.deformation_grid()
    error = rel(np.moveaxis(np.asarray(got), 0, -1), slice_reference["grid_backward"])
    assert error < EXACT


def _run_slice(fixture, params, niter=None):
    """Drive the rank-3 solver from the reference's own axes, images and starting affine."""
    niter = params["SLICE_ITERS"] if niter is None else niter
    reference = tuple(jnp.asarray(fixture[f"ref_axis_{axis}"]) for axis in range(3))
    section = tuple(jnp.asarray(fixture[f"query_axis_{axis}"]) for axis in range(2))
    target = (jnp.zeros(1, dtype=section[0].dtype), *section)
    return _core.lddmm(
        reference,
        jnp.asarray(fixture["ref"]),
        target,
        jnp.asarray(fixture["query"])[:, None],
        L=jnp.asarray(fixture["start_L"]),
        T=jnp.asarray(fixture["start_T"]),
        niter=niter,
        diffeo_start=niter + 1,
        **params["SLICE_PARAMS"],
        **_SOLVER_TAIL,
    )


@pytest.mark.parametrize("name", ["A", "WM", "WA", "WB"])
def test_slice_path_matches_upstream(slice_reference, name, params):
    """The rank-3 solver vs upstream's ``LDDMM_3D_to_slice`` on the same inputs."""
    result = _run_slice(slice_reference, params=params)
    error = rel(result[_FIELDS[name]], slice_reference[name])
    assert error < 1e-10, f"{name}: {error:.3e}"


def test_slice_energy_trace_matches_upstream(slice_reference, params):
    """Every iteration of the rank-3 objective, not just the endpoint."""
    result = _run_slice(slice_reference, params=params)
    got = np.asarray(result["energies"])
    expected = np.asarray(slice_reference["energies"])
    worst = float(np.max(np.abs(got - expected) / np.abs(expected)))
    assert worst < 1e-10


def test_slice_fixture_actually_descends(slice_reference):
    """A trajectory that sits still would agree with anything."""
    energies = np.asarray(slice_reference["energies"])
    assert energies[-1] < energies[0], f"reference did not descend: {energies[0]} -> {energies[-1]}"
    assert len(np.unique(energies)) == energies.size, "reference objective plateaued; retune epL/epT"


def test_slice_transform_matches_upstream_coords(slice_reference, slice_axes, slice_velocity_grid):
    """``transform_points`` at rank 3 vs upstream's ``coord0``/``coord1``/``coord2``."""
    reference, section = slice_axes
    result = StalignVolumeFit(
        affine=jnp.asarray(slice_reference["A"]),
        velocity=jnp.asarray(slice_reference["velocity"]),
        velocity_grid=slice_velocity_grid,
        ref_axes=reference,
        query_axes=section,
    )
    rows, cols = np.meshgrid(np.asarray(section[0]), np.asarray(section[1]), indexing="ij")
    points_xy = np.column_stack((cols.reshape(-1), rows.reshape(-1)))

    got = np.asarray(result.transform_points(points_xy))
    # `build_transform3D` is what `analyze3Dalign` calls, and it stores `(z, y, x)`;
    # the public function returns `(x, y, z)`.
    expected = np.asarray(slice_reference["grid_backward"])[0].reshape(-1, 3)[:, ::-1]
    error = rel(got, expected)
    assert error < EXACT


def test_slice_transform_matches_the_solver_own_sampling_grid(slice_reference, slice_axes, slice_velocity_grid):
    """``transform_points`` vs ``Xs``, the grid the fitted loop itself sampled the volume through."""
    reference, section = slice_axes
    result = StalignVolumeFit(
        affine=jnp.asarray(slice_reference["A_stale"]),
        velocity=jnp.zeros_like(jnp.asarray(slice_reference["velocity"])),
        velocity_grid=slice_velocity_grid,
        ref_axes=reference,
        query_axes=section,
    )
    rows, cols = np.meshgrid(np.asarray(section[0]), np.asarray(section[1]), indexing="ij")
    points_xy = np.column_stack((cols.reshape(-1), rows.reshape(-1)))

    got = np.asarray(result.transform_points(points_xy))
    expected = np.asarray(slice_reference["Xs"])[0].reshape(-1, 3)[:, ::-1]
    error = rel(got, expected)
    assert error < EXACT


@pytest.mark.parametrize(("order", "key"), [(1, "interp_border"), (0, "interp_nearest")])
def test_slice_sample_volume_matches_upstream(slice_reference, slice_axes, order, key):
    """``im.sample_volume`` vs ``interp3D`` on upstream's own backward grid."""
    reference, _ = slice_axes
    # `sample_volume` takes points in `(x, y, z)`, the reverse of `axes` in `(z, y, x)`.
    grid = np.asarray(slice_reference["grid_backward"])[0].reshape(-1, 3)[:, ::-1]
    got = sample_volume(jnp.asarray(slice_reference["ref"]), grid, axes=reference, order=order)
    expected = np.asarray(slice_reference[key]).reshape(2, -1)
    error = rel(got, expected)
    assert error < EXACT


def test_slice_regularizer_axes_diverge_from_upstream(slice_reference, slice_velocity_grid, slice_axes, params):
    """squidpy's rank-3 regulariser is the 3-axis one, upstream's the 2-axis one."""
    xv = slice_velocity_grid
    reference, section = slice_axes
    kernel, ll, dv_prod = _core._build_regularizer(xv, a=params["SLICE_PARAMS"]["a"], p=params["SLICE_PARAMS"]["p"])
    target_image = jnp.asarray(slice_reference["query"])[:, None]

    _, aux = _core._lddmm_loss(
        jnp.asarray(slice_reference["start_L"]),
        jnp.asarray(slice_reference["start_T"]),
        jnp.asarray(slice_reference["velocity"]),
        x_source=reference,
        source_image=jnp.asarray(slice_reference["ref"]),
        x_target=(jnp.zeros(1, dtype=section[0].dtype), *section),
        target_image=target_image,
        xv=xv,
        match_weights=jnp.full(target_image.shape[1:], 0.5),
        ll=ll,
        dv_prod=dv_prod,
        points_source=jnp.zeros((0, 3)),
        points_target=jnp.zeros((0, 3)),
        sigmaM=params["SLICE_PARAMS"]["sigmaM"],
        sigmaR=params["SLICE_PARAMS"]["sigmaR"],
        sigmaP=None,
    )
    reg_energy = aux[1]

    correct = rel(reg_energy, slice_reference["ER_3axis"])
    against_upstream = rel(reg_energy, slice_reference["ER_2axis"])
    assert correct < EXACT, f"squidpy no longer matches the 3-axis regulariser: {correct:.3e}"
    assert against_upstream > 0.1, (
        "squidpy now agrees with upstream's 2-axis regularisation energy; if that is "
        "deliberate, a known divergence has to be rewritten"
    )


def test_slice_fit_reaches_the_same_place_as_the_solver(slice_reference, params):
    """``fit_stalign_volume`` vs the solver it wraps, driven to the same starting affine."""
    swap = np.eye(4)[[2, 1, 0, 3]]
    fit = fit_stalign_volume(
        slice_reference["ref"],
        slice_reference["query"],
        ref_axes=[slice_reference[f"ref_axis_{axis}"] for axis in range(3)],
        query_axes=[slice_reference[f"query_axis_{axis}"] for axis in range(2)],
        initial_affine=np.linalg.inv(swap @ np.asarray(slice_reference["to_A"]) @ swap),
        niter=params["SLICE_ITERS"],
        diffeo_start=params["SLICE_ITERS"] + 1,
        **params["SLICE_PARAMS"],
        **_SOLVER_TAIL,
    )
    error = rel(fit.affine, slice_reference["A"])
    assert error < 1e-10


def _public_volume_fit(fixture, params, niter=None):
    """The rank-3 fit through the public ``stalign_align_volume``, from SpatialData containers."""
    niter = params["SLICE_ITERS"] if niter is None else niter
    reference_axes = [np.asarray(fixture[f"ref_axis_{axis}"]) for axis in range(3)]
    section_axes = [np.asarray(fixture[f"query_axis_{axis}"]) for axis in range(2)]
    forwarded = {"niter": niter, "diffeo_start": niter + 1, **params["SLICE_PARAMS"]}
    return stalign_align_volume(
        _as_sdata(_channels_first(fixture["ref"], ndim=3), reference_axes),
        _as_sdata(_channels_first(fixture["query"], ndim=2), section_axes),
        image_key="image",
        initial_affine=np.linalg.inv(_affine_xy(fixture["start_L"], fixture["start_T"])),
        **forwarded,
    )


def test_public_volume_call_keeps_the_axes_it_was_given(slice_reference, params):
    """``stalign_align_volume`` reads back exactly the axes its containers were given."""
    fit = _public_volume_fit(slice_reference, params=params)

    for axis in range(3):
        expected = np.asarray(slice_reference[f"ref_axis_{axis}"])
        np.testing.assert_allclose(np.asarray(fit.ref_axes[axis]), expected, rtol=0, atol=1e-9)
    for axis in range(2):
        expected = np.asarray(slice_reference[f"query_axis_{axis}"])
        np.testing.assert_allclose(np.asarray(fit.query_axes[axis]), expected, rtol=0, atol=1e-9)


def test_public_volume_call_reproduces_upstream(slice_reference, params):
    """``stalign_align_volume`` from SpatialData containers, against upstream's own result."""
    fit = _public_volume_fit(slice_reference, params=params)
    error = rel(fit.affine, slice_reference["A"])
    assert error < 1e-10, f"the public rank-3 call disagrees with upstream by {error:.3e}"


def test_rank_three_defaults_are_actually_different_from_rank_two():
    """The five knobs where upstream's 3D entry point departs from its 2D one."""
    from squidpy._params import defaults_of
    from squidpy.types import StalignObsParams, StalignVolumeParams

    planar, volume = defaults_of(StalignObsParams), defaults_of(StalignVolumeParams)
    differing = {name for name, value in volume.items() if name in planar and planar[name] != value}
    assert differing == {"expand", "epL", "epT", "epV", "sigmaR"}, sorted(differing)
