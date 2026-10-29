"""STalign port vs the original, on outputs upstream computed (scverse/squidpy#1243).

The reference values come from squidpy-ports' generator, published there as one zarr store;
https://theislab.github.io/squidpy-ports/reference-bundle.html says what it holds and why.
Rank 2 is section-to-section (upstream ``LDDMM``), rank 3 section-into-volume
(``LDDMM_3D_to_slice``). Needs ``JAX_ENABLE_X64=1``.
"""

from __future__ import annotations

from functools import cache

import numpy as np
import pytest

jax = pytest.importorskip("jax")

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
    "https://github.com/theislab/squidpy-ports/releases/download/stalign-reference-v1/stalign_reference.zarr.zip"
)
_BUNDLE_SHA256 = "8097705b69f23f2124d7ba3e2820a81ffdfb153d8e1c031301d51f43361a7f22"


@cache
def _store():
    import pooch
    import zarr
    from zarr.storage import ZipStore

    path = pooch.retrieve(_BUNDLE_URL, known_hash=f"sha256:{_BUNDLE_SHA256}", progressbar=False)
    return zarr.open_group(ZipStore(path, mode="r"), mode="r")


def _load(name: str) -> dict[str, np.ndarray]:
    return {key: array[...] for key, array in _store()[name].arrays()}


class _Params:
    """The fixture parameters the bundle was generated with, from its root attributes."""

    def __getattr__(self, name: str):
        return _store().attrs[name]


F = _Params()


EXACT = 1e-12

_SOLVER_TAIL = {"muA": None, "muB": None, "tol": None, "patience": 25}


def _skip_without_x64() -> None:
    """float64 is a precondition, and it cannot be turned on from inside a test."""
    if not jax.config.jax_enable_x64:
        pytest.skip(
            "STalign reference comparison needs float64 (upstream is double throughout). "
            "Set JAX_ENABLE_X64=1 in the environment -- not via jax.config.update, which "
            "would corrupt the float32 tests in this directory."
        )


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


@pytest.fixture(scope="session", autouse=True)
def _x64():
    _skip_without_x64()


@pytest.fixture(scope="session")
def primitives():
    return _load("primitives")


@pytest.fixture(scope="session")
def clouds(primitives):
    """The point clouds, in ``(x, y)``."""
    from types import SimpleNamespace

    return SimpleNamespace(**{key: primitives[key] for key in ("ref", "query", "landmarks_ref", "landmarks_query")})


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


def _transform_rc(result: StalignObsFit, points_rc, *, direction: str = "forward") -> np.ndarray:
    """``transform_points`` on row-col points, for comparison with upstream."""
    got = result.transform_points(np.asarray(points_rc)[:, ::-1], direction=direction)
    return np.asarray(got)[:, ::-1]


@pytest.mark.parametrize(
    "name",
    [
        "primitives",
        "energy",
        "gradients",
        "trajectory_n1",
        "trajectory_n5",
        "trajectory_n50",
        "converged_n500",
        "image_trajectory",
        "image_trajectory_matched",
        "slice_trajectory",
    ],
)
def test_every_fixture_carries_provenance(name):
    """Without this the .npz files are unfalsifiable magic numbers within a year."""
    payload = _store()[name].attrs["provenance"]
    assert payload["upstream_sha"] == "b2068edc98974efa54537eca194736e177bbe11d"
    for key in ("ports_commit", "torch", "numpy", "python", "platform", "generated_utc"):
        assert payload[key], f"{name}: empty provenance field {key!r}"


def test_fixture_samples_are_off_grid(primitives):
    """No interpolation sample may sit on a grid line.."""
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


def test_regularizer_matches_upstream(primitives, velocity_grid):
    """``LL``/``K``/``DV`` vs STalign.py:1078-1090."""
    kernel, ll, dv_prod = _core._build_regularizer(velocity_grid, a=F.LDDMM_PARAMS["a"], p=F.LDDMM_PARAMS["p"])
    for name, got, expected in (
        ("LL", ll, primitives["regularizer_LL"]),
        ("K", kernel, primitives["regularizer_K"]),
        ("DV", dv_prod, primitives["regularizer_DV"]),
    ):
        error = rel(got, expected)
        assert error < EXACT, name


def test_transform_grid_backward_matches_upstream(primitives, source_grid, target_grid, velocity_grid):
    """``deformation_grid(direction='backward')`` at rank 2 vs ``STalign.build_transform('b')``."""
    axes, _ = target_grid
    query_axes, _ = source_grid
    result = StalignImageFit(
        affine=jnp.asarray(primitives["to_A"]),
        velocity=jnp.asarray(primitives["velocity"]),
        velocity_grid=velocity_grid,
        query_axes=query_axes,
        ref_axes=axes,
    )
    # Backward evaluates the *reference* grid in the query frame.
    got = result.deformation_grid(direction="backward")
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


def _loss_arguments(primitives, source_grid, target_grid, velocity_grid, *, nt, with_points):
    source_axes, source_image = source_grid
    target_axes, target_image = target_grid
    kernel, ll, dv_prod = _core._build_regularizer(velocity_grid, a=F.LDDMM_PARAMS["a"], p=F.LDDMM_PARAMS["p"])
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
            "sigmaM": F.LDDMM_PARAMS["sigmaM"],
            "sigmaR": F.LDDMM_PARAMS["sigmaR"],
            "sigmaP": F.LDDMM_PARAMS["sigmaP"],
        },
        jnp.zeros((nt, velocity_grid[0].shape[0], velocity_grid[1].shape[0], 2)),
    )


@pytest.mark.parametrize("nt", [1, 3])
@pytest.mark.parametrize("with_points", [False, True])
@pytest.mark.parametrize("warm", [False, True], ids=["v_zero", "v_nonzero"])
def test_energy_matches_upstream(primitives, source_grid, target_grid, velocity_grid, nt, with_points, warm):
    """``_lddmm_loss`` vs upstream's ``E`` at iteration 0 -- the two optimise the same function."""
    energy = _load("energy")
    _, kwargs, velocity = _loss_arguments(
        primitives, source_grid, target_grid, velocity_grid, nt=nt, with_points=with_points
    )
    key = f"E_nt{nt}_{'points' if with_points else 'nopoints'}"
    if warm:
        velocity = jnp.asarray(energy[f"warm_velocity_nt{nt}"])
        key += "_warmv"

    total, _ = _core._lddmm_loss(jnp.asarray(energy["L"]), jnp.asarray(energy["T"]), velocity, **kwargs)
    error = rel(total, energy[key])
    assert error < EXACT, f"squidpy {float(total)!r} vs upstream {float(energy[key])!r}"


def test_energy_budget_is_not_vacuous(primitives, source_grid, target_grid, velocity_grid):
    """A tolerance that a perturbed run also passes is decoration, not a test."""
    energy = _load("energy")
    _, kwargs, velocity = _loss_arguments(primitives, source_grid, target_grid, velocity_grid, nt=3, with_points=True)
    kwargs["sigmaM"] *= 1.0001
    total, _ = _core._lddmm_loss(jnp.asarray(energy["L"]), jnp.asarray(energy["T"]), velocity, **kwargs)
    assert rel(total, energy["E_nt3_points"]) > EXACT


@pytest.fixture(scope="session", params=[False, True], ids=["v_zero", "v_nonzero"])
def measured_gradients(request, primitives, source_grid, target_grid, velocity_grid):
    """Gradients of ``_lddmm_loss``, at ``v = 0`` and at a non-zero velocity."""
    gradients = _load("gradients")
    kernel, kwargs, velocity = _loss_arguments(
        primitives, source_grid, target_grid, velocity_grid, nt=F.LDDMM_PARAMS["nt"], with_points=True
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


def test_contrast_transform_freezes_the_ridge_coefficients():
    """Regression guard, stated without reference to upstream."""
    rng = np.random.default_rng(F.SEED)
    warped = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    target = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    weights = jnp.asarray(rng.uniform(0.2, 0.8, size=(12, 15)))

    def live(x):
        return jnp.sum(_core._contrast_transform(x, target, weights) ** 2)

    def leaky(x):
        # _contrast_transform verbatim, minus the stop_gradient: the bug as it was.
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


def test_rasterize_grid_matches_upstream(primitives, clouds):
    """The raster axes are identical to upstream's."""
    grid_x, grid_y, _ = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)
    np.testing.assert_allclose(grid_x, primitives["raster_ref_x"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(grid_y, primitives["raster_ref_y"], rtol=0, atol=1e-9)


def test_velocity_grid_matches_upstream(primitives, source_grid):
    """The velocity grid has upstream's length, without its off-by-one."""
    axes, _ = source_grid
    built = _core._build_velocity_grid(axes, a=F.LDDMM_PARAMS["a"], expand=F.LDDMM_PARAMS["expand"])
    np.testing.assert_allclose(np.asarray(built[0]), primitives["xv_upstream_0"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(built[1]), primitives["xv_upstream_1"], rtol=0, atol=1e-9)


@pytest.mark.parametrize("step", [30.0, 7.3, 0.017])
def test_grid_length_is_stable_against_float_rounding(step):
    """``np.arange(lo, hi + step, step)`` had an unstable length; the grid does not."""
    from squidpy.experimental.im._rasterize_points import axis

    assert axis is _core._axis
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
    got = _transform_rc(fitted, primitives["points"], direction="backward")
    assert rel(got, primitives["points_backward"]) < EXACT


def test_backward_transform_inverts_better(fitted, primitives):
    """squidpy's backward map is the better inverse of the forward one."""
    points = np.asarray(primitives["points"])
    roundtrip = _transform_rc(fitted, primitives["points_forward"], direction="backward")
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


def test_rasterize_stays_within_budget(primitives, clouds):
    """Rasterisation is a deliberate speedup, so it gets a measured budget rather than equality."""
    _, _, got = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)
    expected = primitives["raster_ref"]
    got = np.asarray(got)
    assert got.shape == expected.shape

    for index, blur in enumerate(F.RASTER_PARAMS["blur"]):
        error = rel(got[index], expected[index])
        correlation = float(np.corrcoef(got[index].ravel(), expected[index].ravel())[0, 1])
        assert error < RASTER_BUDGET, f"blur={blur}: relL2 {error:.4%} exceeds {RASTER_BUDGET:.0%}"
        assert correlation > 0.99, f"blur={blur}: correlation {correlation:.5f}"


def test_rasterize_conserves_mass(clouds):
    """Every point contributes exactly one unit, wherever it sits."""
    n_points = clouds.ref.shape[0]
    _, _, got = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)

    for index, blur in enumerate(F.RASTER_PARAMS["blur"]):
        mass = float(np.asarray(got)[index].sum())
        assert mass == pytest.approx(n_points, rel=1e-9), f"blur={blur}: {mass:.2f} of {n_points}"


def _residual(linear, translation, source, target) -> float:
    source, target = np.asarray(source), np.asarray(target)
    return float(np.linalg.norm(source @ np.asarray(linear).T + np.asarray(translation) - target))


def test_affine_from_points_is_equivalent_when_well_conditioned(primitives):
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
        dx=F.RASTER_PARAMS["dx"],
        blur=F.RASTER_PARAMS["blur"],
        raster_expand=F.RASTER_PARAMS["expand"],
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


def _run_lddmm(primitives, snapshot, niter):
    source_axes = (jnp.asarray(primitives["raster_query_y"]), jnp.asarray(primitives["raster_query_x"]))
    target_axes = (jnp.asarray(primitives["raster_ref_y"]), jnp.asarray(primitives["raster_ref_x"]))
    return _core.lddmm(
        source_axes,
        jnp.asarray(primitives["raster_query"]),
        target_axes,
        jnp.asarray(primitives["raster_ref"]),
        L=jnp.asarray(snapshot["L"]),
        T=jnp.asarray(snapshot["T"]),
        points_source=jnp.asarray(primitives["landmarks_query"])[:, ::-1],
        points_target=jnp.asarray(primitives["landmarks_ref"])[:, ::-1],
        niter=niter,
        **F.LDDMM_PARAMS,
        **_SOLVER_TAIL,
    )


_TRAJECTORY_BUDGET = {1: 1e-10, 5: 1e-9, 50: 1e-6}


@pytest.mark.parametrize("niter", [1, 5, 50])
def test_trajectory_matches_upstream(primitives, niter):
    """Run the real loop for ``n`` steps and compare every state it carries."""
    snapshot = _load(f"trajectory_n{niter}")
    result = _run_lddmm(primitives, snapshot, niter)

    budget = _TRAJECTORY_BUDGET[niter]
    for name, got, expected in (
        ("A", result["affine"], snapshot["A"]),
        ("v", result["velocity"], snapshot["v"]),
        ("WM", result["match_weights"], snapshot["WM"]),
        ("WA", result["artifact_weights"], snapshot["WA"]),
        ("WB", result["background_weights"], snapshot["WB"]),
    ):
        error = rel(got, expected)
        assert error < budget, f"n={niter} {name}: {error:.3e} exceeds {budget:.0e}"


_IMAGE_PARAMS = {
    "a": 8.0,
    "p": 2.0,
    "expand": 2.0,
    "nt": 2,
    "diffeo_start": 4,
    "epV": 1.0,
    "epL": 2e-8,
    "epT": 2e-1,
    "sigmaM": 1.0,
    "sigmaB": 2.0,
    "sigmaA": 5.0,
    "sigmaR": 5e5,
    "sigmaP": 2e1,
}
_IMAGE_ITERS = 12


_IMAGE_KEY = "image"


def _placement(axis: np.ndarray) -> tuple[float, float]:
    """The ``(step, shift)`` whose ``arange(n) * step + shift`` best reproduces ``axis``."""
    n, first = axis.size, float(axis[0])
    candidates = (
        (float(axis[-1] - axis[0]) / (n - 1), first),
        (float(axis[1] - axis[0]), first),
        (-2.0 * first / (n - 1), first),
        (float(np.median(np.diff(axis))), first),
    )
    return min(candidates, key=lambda c: np.max(np.abs(np.arange(n) * c[0] + c[1] - axis)))


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
    return SpatialData(images={_IMAGE_KEY: element})


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
def image_reference():
    return _load("image_trajectory")


def _harness_shaped_fit(fixture, *, swap_roles=False):
    """The image fit exactly as the retired harness issued it."""
    ax_moving = (fixture["source_axis_0"], fixture["source_axis_1"])
    ax_fixed = (fixture["target_axis_0"], fixture["target_axis_1"])
    moving = _as_sdata(_channels_first(fixture["query"], ndim=2), ax_moving)
    fixed = _as_sdata(_channels_first(fixture["ref"], ndim=2), ax_fixed)
    affine = _affine_xy(fixture["start_L"], fixture["start_T"])
    first, second = (moving, fixed) if swap_roles else (fixed, moving)
    return stalign_align_image(
        first, second, image_key=_IMAGE_KEY, initial_affine=affine, niter=_IMAGE_ITERS, **_IMAGE_PARAMS
    )


def test_replay_image_call_reproduces_upstream(image_reference):
    """The call the notebook replay actually issues, against upstream's own result."""
    fit = _harness_shaped_fit(image_reference)
    error = rel(fit.affine, image_reference["A"])
    assert error < 1e-10, f"the replay's own call disagrees with upstream by {error:.3e}"


def test_replay_image_call_is_role_sensitive(image_reference):
    """Swapping ref and query must be *loud*, not a 1e-5 wobble."""
    swapped = _harness_shaped_fit(image_reference, swap_roles=True)
    assert rel(swapped.affine, image_reference["A"]) > 1e-3


@pytest.fixture(scope="session")
def image_mixed_units():
    """Source in one unit, target in another -- see :func:`generate._write_image_mixed_units`."""
    return _load("image_mixed_units")


def _fit_mixed_units(fixture, niter):
    """The image path on the mixed-unit axes, from the fixture's own ``a``."""
    params = {**_IMAGE_PARAMS, "a": float(fixture["a"])}
    return fit_stalign_image(
        fixture["ref"],
        fixture["query"],
        ref_axes=(fixture["target_axis_0"], fixture["target_axis_1"]),
        query_axes=(fixture["source_axis_0"], fixture["source_axis_1"]),
        niter=niter,
        **params,
    )


def test_mixed_unit_velocity_grid_matches_upstream(image_mixed_units):
    """The velocity grid upstream builds when the two sides disagree about units."""
    fit = _fit_mixed_units(image_mixed_units, _IMAGE_ITERS)
    got = tuple(int(np.asarray(axis).size) for axis in fit.velocity_grid)
    expected = (int(image_mixed_units["xv_0"].size), int(image_mixed_units["xv_1"].size))
    assert got == expected, f"velocity grid {got} against upstream's {expected}"


def test_mixed_unit_axes_are_actually_mixed(image_mixed_units):
    """The fixture is only meaningful while the two sides really carry different units."""
    source_step = float(image_mixed_units["source_axis_0"][1] - image_mixed_units["source_axis_0"][0])
    target_step = float(image_mixed_units["target_axis_0"][1] - image_mixed_units["target_axis_0"][0])
    assert source_step == pytest.approx(1.0)
    assert target_step == pytest.approx(F.MIXED_TARGET_STEP)


def test_image_path_axes_match_the_reference(image_reference):
    """The image entry point's coordinate convention, checked before its results."""
    fit = fit_stalign_image(image_reference["ref"], image_reference["query"], niter=0, **_IMAGE_PARAMS)

    for got, expected in (
        (fit.query_axes[0], image_reference["source_axis_0"]),
        (fit.query_axes[1], image_reference["source_axis_1"]),
        (fit.ref_axes[0], image_reference["target_axis_0"]),
        (fit.ref_axes[1], image_reference["target_axis_1"]),
    ):
        np.testing.assert_allclose(np.asarray(got), expected, rtol=0, atol=1e-12)


def _run_image(fixture, axes_ref: bool = True):
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
        niter=_IMAGE_ITERS,
        **_IMAGE_PARAMS,
        **_SOLVER_TAIL,
    )


@pytest.mark.parametrize("name", ["A", "v", "WM", "WA", "WB"])
def test_image_path_matches_upstream(image_reference, name):
    """The image entry point vs upstream's LDDMM on the same images, axes and start."""
    result = _run_image(image_reference)
    error = rel(result[_FIELDS[name]], image_reference[name])
    assert error < 1e-10, f"{name}: {error:.3e}"


def test_image_energy_trace_matches_upstream(image_reference):
    """Every iteration of the objective, not just the endpoint."""
    result = _run_image(image_reference)
    got = np.asarray(result["energies"])
    expected = np.asarray(image_reference["energies"])
    worst = max(abs(got[i] - expected[i]) / abs(expected[i]) for i in range(expected.size))
    assert worst < 1e-10


def test_image_warp_matches_upstream(image_reference):
    """``warp_image`` vs ``STalign.transform_image_source_to_target``."""
    result = _run_image(image_reference)
    fit = StalignImageFit(
        affine=result["affine"],
        velocity=result["velocity"],
        velocity_grid=result["velocity_grid"],
        query_axes=(jnp.asarray(image_reference["source_axis_0"]), jnp.asarray(image_reference["source_axis_1"])),
        ref_axes=(jnp.asarray(image_reference["target_axis_0"]), jnp.asarray(image_reference["target_axis_1"])),
    )
    error = rel(fit.warp_image(jnp.asarray(image_reference["query"])), image_reference["warped"])
    assert error < 1e-9


def test_on_grid_sampling_costs_six_orders_of_magnitude():
    """Sampling exactly on grid lines costs agreement, measured rather than asserted."""
    matched = _load("image_trajectory_matched")
    axes = (jnp.asarray(matched["axis_0"]), jnp.asarray(matched["axis_1"]))
    result = _core.lddmm(
        axes,
        jnp.asarray(matched["query"]),
        axes,
        jnp.asarray(matched["ref"]),
        L=matched["start_L"],
        T=matched["start_T"],
        niter=_IMAGE_ITERS,
        **_IMAGE_PARAMS,
        **_SOLVER_TAIL,
    )
    degraded = rel(result["velocity"], matched["v"])
    assert 1e-6 < degraded < 1e-1, (
        f"on-grid sampling now costs {degraded:.3e}; if this has become exact, the on-grid difference is gone "
        f"and the off-grid fixture design can be simplified"
    )


def test_velocity_grid_is_the_one_the_reference_used(primitives, source_grid):
    """The trajectory comparison is only meaningful on a shared velocity grid."""
    axes, _ = source_grid
    built = _core._build_velocity_grid(axes, a=F.LDDMM_PARAMS["a"], expand=F.LDDMM_PARAMS["expand"])
    snapshot = _load("trajectory_n1")
    np.testing.assert_allclose(np.asarray(built[0]), snapshot["xv_0"], rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(built[1]), snapshot["xv_1"], rtol=0, atol=1e-9)


def test_converged_solution_matches_upstream(primitives):
    """500 iterations -- "enough to actually converge", per #1243."""
    snapshot = _load("converged_n500")
    result = _run_lddmm(primitives, snapshot, 500)

    energy = rel(result["energies"][result["n_iter"] - 1], snapshot["E_last"])
    assert energy < 0.01, f"final energy differs by {energy:.3%}"

    before = float(np.mean(np.linalg.norm(primitives["landmarks_query"] - primitives["landmarks_ref"], axis=1)))
    after = float(snapshot["tre_mean"])
    assert after < 0.4 * F.RASTER_PARAMS["dx"], f"reference did not converge: TRE {after:.2f}"
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
    assert percentile < 0.1 * F.RASTER_PARAMS["dx"], (
        f"95th-percentile point disagreement {percentile:.4f} exceeds a tenth of a grid cell"
    )


_SLICE_PARAMS = {
    "a": 8.0,
    "p": 2.0,
    "expand": 1.25,
    "nt": 2,
    "epL": 1e-9,
    "epT": 1e-2,
    "epV": 1.0,
    "sigmaM": 1.0,
    "sigmaB": 2.0,
    "sigmaA": 5.0,
    "sigmaR": 1e8,
}
_SLICE_ITERS = 12


@pytest.fixture(scope="session")
def slice_reference():
    return _load("slice_trajectory")


@pytest.fixture(scope="session")
def slice_axes(slice_reference):
    """``(reference (z, y, x) axes, section (y, x) axes)``, both as JAX arrays."""
    reference = tuple(jnp.asarray(slice_reference[f"ref_axis_{axis}"]) for axis in range(3))
    section = tuple(jnp.asarray(slice_reference[f"query_axis_{axis}"]) for axis in range(2))
    return reference, section


@pytest.fixture(scope="session")
def slice_target_axes(slice_reference, slice_axes):
    """The section's axes as the rank-3 solver sees them: a single-sample z prepended."""
    _, section = slice_axes
    return (jnp.zeros(1, dtype=section[0].dtype), *section)


def test_slice_to_affine_matches_upstream(slice_reference):
    """``_to_affine`` at rank 3 vs ``STalign.to_A_3D``."""
    got = _core._to_affine(jnp.asarray(slice_reference["start_L"]), jnp.asarray(slice_reference["start_T"]))
    np.testing.assert_allclose(np.asarray(got), slice_reference["to_A"], rtol=0, atol=0)


def test_slice_velocity_grid_matches_upstream(slice_reference, slice_axes):
    """``_build_velocity_grid`` at rank 3 vs the grid upstream builds for itself."""
    reference, _ = slice_axes
    built = _core._build_velocity_grid(reference, a=_SLICE_PARAMS["a"], expand=_SLICE_PARAMS["expand"])
    assert len(built) == 3
    for axis, got in enumerate(built):
        np.testing.assert_allclose(np.asarray(got), slice_reference[f"xv_{axis}"], rtol=0, atol=1e-9)


def test_slice_regularizer_matches_upstream(slice_reference):
    """``LL``/``K``/``DV`` at rank 3 vs STalign.py:1384-1397."""
    xv = tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3))
    kernel, ll, dv_prod = _core._build_regularizer(xv, a=_SLICE_PARAMS["a"], p=_SLICE_PARAMS["p"])
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


def test_slice_grid_backward_matches_upstream(slice_reference, slice_axes):
    """``deformation_grid(direction='backward')`` at rank 3 vs ``build_transform3D``."""
    reference, section = slice_axes
    xv = tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3))
    result = StalignVolumeFit(
        affine=jnp.asarray(slice_reference["A"]),
        velocity=jnp.asarray(slice_reference["velocity"]),
        velocity_grid=xv,
        ref_axes=reference,
        query_axes=section,
    )
    got = result.deformation_grid(direction="backward")
    error = rel(np.moveaxis(np.asarray(got), 0, -1), slice_reference["grid_backward"])
    assert error < EXACT


def _run_slice(fixture, niter=_SLICE_ITERS):
    """Drive the rank-3 solver from the reference's own axes, images and starting affine."""
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
        sigmaP=2e1,
        **_SLICE_PARAMS,
        **_SOLVER_TAIL,
    )


@pytest.mark.parametrize("name", ["A", "WM", "WA", "WB"])
def test_slice_path_matches_upstream(slice_reference, name):
    """The rank-3 solver vs upstream's ``LDDMM_3D_to_slice`` on the same inputs."""
    result = _run_slice(slice_reference)
    error = rel(result[_FIELDS[name]], slice_reference[name])
    assert error < 1e-10, f"{name}: {error:.3e}"


def test_slice_energy_trace_matches_upstream(slice_reference):
    """Every iteration of the rank-3 objective, not just the endpoint."""
    result = _run_slice(slice_reference)
    got = np.asarray(result["energies"])
    expected = np.asarray(slice_reference["energies"])
    worst = max(abs(got[i] - expected[i]) / abs(expected[i]) for i in range(expected.size))
    assert worst < 1e-10


def test_slice_fixture_actually_descends(slice_reference):
    """A trajectory that sits still would agree with anything."""
    energies = np.asarray(slice_reference["energies"])
    assert energies[-1] < energies[0], f"reference did not descend: {energies[0]} -> {energies[-1]}"
    assert len(np.unique(energies)) == energies.size, "reference objective plateaued; retune epL/epT"


def test_slice_transform_matches_upstream_coords(slice_reference, slice_axes):
    """``transform_points`` at rank 3 vs upstream's ``coord0``/``coord1``/``coord2``."""
    reference, section = slice_axes
    result = StalignVolumeFit(
        affine=jnp.asarray(slice_reference["A"]),
        velocity=jnp.asarray(slice_reference["velocity"]),
        velocity_grid=tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3)),
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


def test_slice_transform_matches_the_solver_own_sampling_grid(slice_reference, slice_axes):
    """``transform_points`` vs ``Xs``, the grid the fitted loop itself sampled the volume through."""
    reference, section = slice_axes
    result = StalignVolumeFit(
        affine=jnp.asarray(slice_reference["A_stale"]),
        velocity=jnp.zeros_like(jnp.asarray(slice_reference["velocity"])),
        velocity_grid=tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3)),
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


def test_slice_regularizer_axes_diverge_from_upstream(slice_reference):
    """squidpy's rank-3 regulariser is the 3-axis one, upstream's the 2-axis one."""
    xv = tuple(jnp.asarray(slice_reference[f"xv_{axis}"]) for axis in range(3))
    section = tuple(jnp.asarray(slice_reference[f"query_axis_{axis}"]) for axis in range(2))
    reference = tuple(jnp.asarray(slice_reference[f"ref_axis_{axis}"]) for axis in range(3))
    kernel, ll, dv_prod = _core._build_regularizer(xv, a=_SLICE_PARAMS["a"], p=_SLICE_PARAMS["p"])
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
        sigmaM=_SLICE_PARAMS["sigmaM"],
        sigmaR=_SLICE_PARAMS["sigmaR"],
        sigmaP=2e1,
    )
    reg_energy = aux[3]

    correct = rel(reg_energy, slice_reference["ER_3axis"])
    against_upstream = rel(reg_energy, slice_reference["ER_2axis"])
    assert correct < EXACT, f"squidpy no longer matches the 3-axis regulariser: {correct:.3e}"
    assert against_upstream > 0.1, (
        "squidpy now agrees with upstream's 2-axis regularisation energy; if that is "
        "deliberate, a known divergence has to be rewritten"
    )


def test_slice_fit_reaches_the_same_place_as_the_solver(slice_reference):
    """``fit_stalign_volume`` vs the solver it wraps, driven to the same starting affine."""
    swap = np.eye(4)[[2, 1, 0, 3]]
    fit = fit_stalign_volume(
        slice_reference["ref"],
        slice_reference["query"],
        ref_axes=[slice_reference[f"ref_axis_{axis}"] for axis in range(3)],
        query_axes=[slice_reference[f"query_axis_{axis}"] for axis in range(2)],
        initial_affine=swap @ np.asarray(slice_reference["to_A"]) @ swap,
        niter=_SLICE_ITERS,
        diffeo_start=_SLICE_ITERS + 1,
        **_SLICE_PARAMS,
        **{key: value for key, value in _SOLVER_TAIL.items() if key != "sigmaP"},
    )
    error = rel(fit.affine, slice_reference["A"])
    assert error < 1e-10


def _harness_shaped_volume_fit(fixture, niter=_SLICE_ITERS):
    """The rank-3 fit exactly as the retired harness issued it."""
    reference_axes = [np.asarray(fixture[f"ref_axis_{axis}"]) for axis in range(3)]
    section_axes = [np.asarray(fixture[f"query_axis_{axis}"]) for axis in range(2)]
    forwarded = {"niter": niter, "diffeo_start": niter + 1, **_SLICE_PARAMS}
    return stalign_align_volume(
        _as_sdata(_channels_first(fixture["ref"], ndim=3), reference_axes),
        _as_sdata(_channels_first(fixture["query"], ndim=2), section_axes),
        image_key=_IMAGE_KEY,
        initial_affine=_affine_xy(fixture["start_L"], fixture["start_T"]),
        **forwarded,
    )


def test_replay_volume_call_resolves_the_axes_it_was_given(slice_reference):
    """The rank-3 half of the question the rank-2 seam answered: are the axes bit-faithful?"""
    fit = _harness_shaped_volume_fit(slice_reference)

    for axis in range(3):
        expected = np.asarray(slice_reference[f"ref_axis_{axis}"])
        np.testing.assert_allclose(np.asarray(fit.ref_axes[axis]), expected, rtol=0, atol=1e-9)
    for axis in range(2):
        expected = np.asarray(slice_reference[f"query_axis_{axis}"])
        np.testing.assert_allclose(np.asarray(fit.query_axes[axis]), expected, rtol=0, atol=1e-9)


def test_replay_volume_call_reproduces_upstream(slice_reference):
    """The rank-3 counterpart of ``test_replay_image_call_reproduces_upstream``."""
    fit = _harness_shaped_volume_fit(slice_reference)
    error = rel(fit.affine, slice_reference["A"])
    assert error < 1e-10, f"the replay's own rank-3 call disagrees with upstream by {error:.3e}"


def test_rank_three_defaults_are_actually_different_from_rank_two():
    """The five knobs where upstream's 3D entry point departs from its 2D one."""
    from squidpy._params import defaults_of
    from squidpy.types import StalignObsParams, StalignVolumeParams

    planar, volume = defaults_of(StalignObsParams), defaults_of(StalignVolumeParams)
    differing = {name for name, value in volume.items() if name in planar and planar[name] != value}
    assert differing == {"expand", "epL", "epT", "epV", "sigmaR"}, sorted(differing)
