"""Numerical comparison of the STalign port against the original implementation.

Implements scverse/squidpy#1243. The reference values in ``tests/_data/stalign_reference``
are produced out of band by https://github.com/scverse/squidpy-ports, which vendors the
PyTorch STalign at a pinned commit -- so torch never becomes a squidpy dependency.

These are **excluded from normal runs**. See ``STALIGN_DIVERGENCES.md`` next to this file
for what each comparison found and why the known-divergent ones are pinned rather than
fixed here.

Run them with::

    JAX_ENABLE_X64=1 pytest tests/experimental/methods/test_stalign_reference.py -m reference
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from squidpy.experimental.methods.align_samples._stalign_impl import _core, _helpers  # noqa: E402

from . import _stalign_fixtures as F  # noqa: E402

pytestmark = pytest.mark.reference

DATA = Path(__file__).parent.parent.parent / "_data" / "stalign_reference"
LEDGER = Path(__file__).parent / "STALIGN_DIVERGENCES.md"

# Tolerances are calibrated from the measured gaps recorded in STALIGN_DIVERGENCES.md,
# not guessed. Everything the port reproduces faithfully lands at 1e-15 or better, so a
# 1e-12 budget leaves three orders of headroom while still being three orders tighter
# than any divergence we are pinning.
EXACT = 1e-12


def _skip_without_x64() -> None:
    """float64 is a precondition, and it cannot be turned on from inside a test.

    ``jax.config.update`` is process-global; under ``-n auto`` every worker imports every
    test module, so setting it here would silently flip the float32 tests in
    ``test_stalign.py`` to float64 in the same worker. It has to come from the
    environment.
    """
    if not jax.config.jax_enable_x64:
        pytest.skip(
            "STalign reference comparison needs float64 (upstream is double throughout). "
            "Set JAX_ENABLE_X64=1 in the environment -- not via jax.config.update, which "
            "would corrupt the float32 tests in this directory."
        )


def _load(name: str) -> np.lib.npyio.NpzFile:
    path = DATA / f"{name}.npz"
    if not path.is_file():
        pytest.skip(f"reference bundle missing at {path}; regenerate it with scverse/squidpy-ports")
    return np.load(path)


def rel(actual, expected) -> float:
    """Relative L2 error, the single measure used throughout."""
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    denominator = np.linalg.norm(expected)
    return float(np.linalg.norm(actual - expected) / max(denominator, np.finfo(float).tiny))


@pytest.fixture(scope="module", autouse=True)
def _x64():
    _skip_without_x64()


@pytest.fixture(scope="module")
def primitives():
    return _load("primitives")


@pytest.fixture(scope="module")
def clouds():
    return F.make_clouds()


@pytest.fixture(scope="module")
def source_grid(primitives):
    """Upstream's rasterised reference image and its axes, as ``((y, x), image)``."""
    return (jnp.asarray(primitives["raster_ref_y"]), jnp.asarray(primitives["raster_ref_x"])), jnp.asarray(
        primitives["raster_ref"]
    )


@pytest.fixture(scope="module")
def target_grid(primitives):
    """Upstream's rasterised query image and its axes."""
    return (jnp.asarray(primitives["raster_query_y"]), jnp.asarray(primitives["raster_query_x"])), jnp.asarray(
        primitives["raster_query"]
    )


@pytest.fixture(scope="module")
def velocity_grid(primitives):
    """Upstream's own velocity grid, which ``_lddmm_loss`` accepts directly."""
    return jnp.asarray(primitives["xv_upstream_0"]), jnp.asarray(primitives["xv_upstream_1"])


# --------------------------------------------------------------------------------------
# Provenance -- the fixtures have to stay falsifiable
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["primitives", "energy", "gradients", "trajectory_n1", "trajectory_n5", "trajectory_n50", "converged_n500"],
)
def test_every_fixture_carries_provenance(name):
    """Without this the .npz files are unfalsifiable magic numbers within a year."""
    payload = json.loads(str(_load(name)["__provenance__"]))
    assert payload["upstream_sha"] == "b2068edc98974efa54537eca194736e177bbe11d"
    for key in ("ports_commit", "torch", "numpy", "python", "platform", "generated_utc"):
        assert payload[key], f"{name}: empty provenance field {key!r}"


def test_fixture_definitions_have_not_drifted():
    """The generator and these tests must build inputs from the same file, byte for byte.

    ``_stalign_fixtures.py`` is vendored from squidpy-ports. If someone edits one copy
    without regenerating, every comparison below silently compares different inputs.
    """
    recorded = json.loads(str(_load("primitives")["__provenance__"]))["fixtures_checksum"]
    assert F.checksum() == recorded, (
        "_stalign_fixtures.py differs from the copy the reference bundle was generated "
        "with. Re-sync it from squidpy-ports and regenerate the bundle."
    )


# --------------------------------------------------------------------------------------
# Fixture validation -- runs before anything that depends on it
# --------------------------------------------------------------------------------------


def test_fixture_samples_are_off_grid(primitives):
    """No interpolation sample may sit on a grid line. See ledger row D10.

    Upstream and squidpy compute the fractional index by formulas that agree to ~1 ulp;
    exactly on a grid line they can floor() to different neighbours, giving an O(1)
    difference that says nothing about the port.
    """
    coords = np.asarray(primitives["interp_coords"])
    axes = (np.asarray(primitives["raster_ref_y"]), np.asarray(primitives["raster_ref_x"]))
    for axis, values in zip(axes, coords, strict=True):
        fractional = np.abs(np.modf((values - axis[0]) / (axis[1] - axis[0]))[0])
        assert np.minimum(fractional, 1.0 - fractional).min() > 1e-6


def test_fixture_stays_inside_velocity_grid(primitives, velocity_grid):
    """Keeps the padding divergence (D5) out of every test that is not about it."""
    points = np.asarray(primitives["points"])
    for axis, column in zip(velocity_grid, points.T, strict=True):
        assert column.min() > float(axis[0]) and column.max() < float(axis[-1])


# --------------------------------------------------------------------------------------
# Primitives that the port reproduces faithfully
# --------------------------------------------------------------------------------------


def test_interp_matches_upstream(primitives, source_grid, record_property):
    """``_interp`` vs ``STalign.interp(padding_mode='border')``."""
    axes, image = source_grid
    got = _core._interp(axes, image, jnp.asarray(primitives["interp_coords"]))
    error = rel(got, primitives["interp_border"])
    record_property("rel_error", error)
    assert error < EXACT


def test_regularizer_matches_upstream(primitives, velocity_grid, record_property):
    """``LL``/``K``/``DV`` vs STalign.py:1078-1090.

    This is a precondition, not a nicety: the regulariser sets the scale of the whole
    velocity term, so if it disagreed every later comparison would be meaningless.
    """
    kernel, ll, dv_prod = _core._build_regularizer(velocity_grid, a=F.LDDMM_PARAMS["a"], p=F.LDDMM_PARAMS["p"])
    for name, got, expected in (
        ("LL", ll, primitives["regularizer_LL"]),
        ("K", kernel, primitives["regularizer_K"]),
        ("DV", dv_prod, primitives["regularizer_DV"]),
    ):
        error = rel(got, expected)
        record_property(f"rel_error_{name}", error)
        assert error < EXACT, name


def test_transform_grid_backward_matches_upstream(primitives, target_grid, velocity_grid, record_property):
    """``_transform_grid_backward`` vs ``STalign.build_transform(direction='b')``.

    This is the inner loop of the objective: invert the affine, then integrate ``-v``
    backwards in time. Upstream returns ``(H, W, 2)``; squidpy returns ``(2, H, W)``.
    """
    axes, _ = target_grid
    got = _core._transform_grid_backward(
        axes, velocity_grid, jnp.asarray(primitives["velocity"]), jnp.asarray(primitives["to_A"])
    )
    error = rel(np.moveaxis(np.asarray(got), 0, -1), primitives["grid_backward"])
    record_property("rel_error", error)
    assert error < EXACT


def test_transform_points_forward_matches_upstream(primitives, velocity_grid, record_property):
    """``direction='forward'`` vs ``STalign.transform_points_source_to_target``."""
    got = _core.transform_points_row_col(
        velocity_grid,
        jnp.asarray(primitives["velocity"]),
        jnp.asarray(primitives["to_A"]),
        jnp.asarray(primitives["points"]),
        direction="forward",
    )
    error = rel(got, primitives["points_forward"])
    record_property("rel_error", error)
    assert error < EXACT


def test_to_affine_matches_upstream(primitives):
    """``_to_affine`` vs ``STalign.to_A``."""
    got = _core._to_affine(jnp.asarray(primitives["to_A_linear"]), jnp.asarray(primitives["to_A_translation"]))
    np.testing.assert_allclose(np.asarray(got), primitives["to_A"], rtol=0, atol=0)


# --------------------------------------------------------------------------------------
# The objective itself
# --------------------------------------------------------------------------------------


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
def test_energy_matches_upstream(
    primitives, source_grid, target_grid, velocity_grid, nt, with_points, warm, record_property
):
    """``_lddmm_loss`` vs upstream's ``E`` at iteration 0.

    The strongest single result in this suite: it says the two implementations optimise
    the *same function*. Divergence D3 is inert here because no gradient is taken.

    The ``v_nonzero`` half is not redundant. LDDMM starts at ``v = 0``, where the
    regularisation term ``ER`` contributes exactly nothing -- so evaluating only there
    would compare equal even if ``ER`` were negated. Verified: with ``ER``'s sign flipped
    in ``_lddmm_loss``, only these cases fail.
    """
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
    record_property("rel_error", error)
    assert error < EXACT, f"squidpy {float(total)!r} vs upstream {float(energy[key])!r}"


def test_energy_budget_is_not_vacuous(primitives, source_grid, target_grid, velocity_grid):
    """A tolerance that a perturbed run also passes is decoration, not a test.

    Nudge one physically meaningful knob and the same assertion must fail.
    """
    energy = _load("energy")
    _, kwargs, velocity = _loss_arguments(primitives, source_grid, target_grid, velocity_grid, nt=3, with_points=True)
    kwargs["sigmaM"] *= 1.0001
    total, _ = _core._lddmm_loss(jnp.asarray(energy["L"]), jnp.asarray(energy["T"]), velocity, **kwargs)
    assert rel(total, energy["E_nt3_points"]) > EXACT


# --------------------------------------------------------------------------------------
# Gradients -- ledger row D3
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module", params=[False, True], ids=["v_zero", "v_nonzero"])
def measured_gradients(request, primitives, source_grid, target_grid, velocity_grid):
    """Gradients of ``_lddmm_loss``, at ``v = 0`` and at a non-zero velocity.

    Both are needed for the same reason as the energy test: ``dER/dv`` vanishes at the
    origin, so gradients taken only there cannot see the regularisation term.
    """
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
@pytest.mark.xfail(
    strict=True,
    reason=(
        "ledger row D3: _contrast_transform differentiates through jnp.linalg.solve, "
        "while upstream solves the ridge coefficients under torch.no_grad "
        "(STalign.py:1184-1188). The ridge fit is an EM M-step; differentiating through "
        "it turns alternating minimisation into joint optimisation. Fix is "
        "jax.lax.stop_gradient around the solve -- when it lands, delete this xfail, "
        "update STALIGN_DIVERGENCES.md row D3, and add a changelog entry."
    ),
)
def test_gradients_match_upstream(measured_gradients, component, key, record_property):
    gradients, computed, suffix = measured_gradients
    error = rel(computed[component], gradients[key + suffix])
    record_property("rel_error", error)
    assert error < EXACT


def test_gradient_gap_is_the_contrast_transform(measured_gradients, record_property):
    """The positive form of D3: the gap is real, bounded, and where we say it is.

    Everything else in the objective agrees to ~1e-15, so a gradient discrepancy three
    orders of magnitude larger cannot be floating-point noise.
    """
    gradients, computed, suffix = measured_gradients
    for component, key in (("L", "grad_L"), ("T", "grad_T"), ("v", "grad_v_smoothed")):
        error = rel(computed[component], gradients[key + suffix])
        record_property(f"rel_error_{component}", error)
        assert 1e-5 < error < 1e-2, f"{component}: {error:.3e} is outside the recorded D3 band"


def test_contrast_transform_leaks_gradient_through_the_solve(record_property):
    """Isolates D3 without reference to upstream at all.

    Upstream's ``no_grad`` makes the ridge coefficients a constant of the optimisation.
    Here, freezing them changes the gradient -- which is exactly the bug, stated as a
    property of squidpy's own code rather than as a disagreement with someone else's.
    """
    rng = np.random.default_rng(F.SEED)
    warped = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    target = jnp.asarray(rng.normal(size=(3, 12, 15)) ** 2)
    weights = jnp.asarray(rng.uniform(0.2, 0.8, size=(12, 15)))

    def live(x):
        return jnp.sum(_core._contrast_transform(x, target, weights) ** 2)

    def frozen(x):
        # _contrast_transform verbatim, with the one change upstream's no_grad makes.
        flat_source = x.reshape(x.shape[0], -1)
        flat_target = target.reshape(target.shape[0], -1)
        design = jnp.concatenate((jnp.ones((1, flat_source.shape[1]), dtype=x.dtype), flat_source), axis=0)
        weighted = design * weights.reshape(-1)[None, :]
        coefficients = jax.lax.stop_gradient(
            jnp.linalg.solve(
                weighted @ design.T + 0.1 * jnp.eye(design.shape[0], dtype=x.dtype),
                weighted @ flat_target.T,
            )
        )
        return jnp.sum((coefficients.T @ design).reshape(target.shape) ** 2)

    # Same value, different gradient: the leak is entirely in the backward pass.
    np.testing.assert_allclose(float(live(warped)), float(frozen(warped)), rtol=1e-12)
    error = rel(jax.grad(live)(warped), jax.grad(frozen)(warped))
    record_property("rel_error", error)
    assert error > 1e-6


# --------------------------------------------------------------------------------------
# Pinned divergences
# --------------------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ledger row D2: squidpy uses np.arange(lo, hi + dx, dx) where upstream uses "
        "np.arange(lo, hi, dx), so every axis is one point longer -- and because `hi` is "
        "a float sum, the surplus is one or two points depending on rounding. Fix is "
        "lo + dx * np.arange(int(np.ceil((hi - lo) / dx))). Needs its own PR: it changes "
        "user-visible output shapes."
    ),
)
def test_rasterize_grid_length_matches_upstream(primitives, clouds):
    grid_x, grid_y, _ = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)
    assert grid_x.size == primitives["raster_ref_x"].size
    assert grid_y.size == primitives["raster_ref_y"].size


def test_rasterize_grid_agrees_where_it_overlaps(primitives, clouds):
    """The positive half of D2: the surplus is purely trailing, the spacing is right."""
    grid_x, grid_y, _ = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)
    upstream_x, upstream_y = primitives["raster_ref_x"], primitives["raster_ref_y"]
    np.testing.assert_allclose(grid_x[: upstream_x.size], upstream_x, rtol=0, atol=1e-9)
    np.testing.assert_allclose(grid_y[: upstream_y.size], upstream_y, rtol=0, atol=1e-9)
    assert grid_x.size > upstream_x.size


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ledger row D2: _build_velocity_grid has the same np.arange(lo, hi + step, step) "
        "off-by-one as rasterize. Fixed by the same change."
    ),
)
def test_velocity_grid_length_matches_upstream(primitives, source_grid):
    axes, _ = source_grid
    built = _core._build_velocity_grid(axes, a=F.LDDMM_PARAMS["a"], expand=F.LDDMM_PARAMS["expand"])
    assert built[0].shape[0] == primitives["xv_upstream_0"].size
    assert built[1].shape[0] == primitives["xv_upstream_1"].size


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ledger row D6: squidpy integrates the backward flow in reversed time order, "
        "upstream does not (STalign.py:1828-1843). squidpy is correct -- see "
        "test_backward_transform_inverts_better, which is the assertion that matters. "
        "This xfail exists to pin the literal difference, and should be deleted only if "
        "squidpy ever deliberately adopts upstream's ordering."
    ),
)
def test_transform_points_backward_matches_upstream(primitives, velocity_grid):
    got = _core.transform_points_row_col(
        velocity_grid,
        jnp.asarray(primitives["velocity"]),
        jnp.asarray(primitives["to_A"]),
        jnp.asarray(primitives["points"]),
        direction="backward",
    )
    assert rel(got, primitives["points_backward"]) < EXACT


def test_backward_transform_inverts_better(primitives, velocity_grid, record_property):
    """D6, stated usefully: squidpy's backward map is the better inverse of the forward one."""
    points = jnp.asarray(primitives["points"])
    roundtrip = _core.transform_points_row_col(
        velocity_grid,
        jnp.asarray(primitives["velocity"]),
        jnp.asarray(primitives["to_A"]),
        jnp.asarray(primitives["points_forward"]),
        direction="backward",
    )
    ours = rel(roundtrip, points)
    theirs = rel(primitives["points_roundtrip"], points)
    record_property("roundtrip_squidpy", ours)
    record_property("roundtrip_upstream", theirs)
    assert ours <= theirs


@pytest.mark.xfail(
    strict=True,
    reason=(
        "ledger row D5: upstream samples the velocity field with grid_sample's default "
        "padding_mode='zeros' (STalign.py:1163, :1167), squidpy uses "
        "map_coordinates(mode='nearest'). squidpy is correct -- zeros make a point that "
        "drifts off the velocity grid snap to no displacement at all."
    ),
)
def test_interp_outside_domain_matches_upstream(primitives, source_grid):
    axes, image = source_grid
    got = _core._interp(axes, image, jnp.asarray(primitives["interp_coords_outside"]))
    assert rel(got, primitives["interp_zeros_outside"]) < EXACT


def test_interp_outside_domain_is_border_padding(primitives, source_grid, record_property):
    """The positive half of D5: squidpy's behaviour is exactly upstream's 'border' mode."""
    axes, image = source_grid
    got = _core._interp(axes, image, jnp.asarray(primitives["interp_coords_outside"]))
    error = rel(got, primitives["interp_border_outside"])
    record_property("rel_error", error)
    assert error < EXACT
    assert rel(got, primitives["interp_zeros_outside"]) > 0.1


# --------------------------------------------------------------------------------------
# Budgeted divergences
# --------------------------------------------------------------------------------------

#: Per-blur relative-L2 budget for the rasteriser (ledger row D1). Measured 6.20 % /
#: 1.98 % / 5.96 % at blur 2.0 / 1.0 / 0.5; 9 % leaves headroom without going vacuous.
RASTER_BUDGET = 0.09
#: Upstream conserves each point's mass exactly; squidpy's mode="constant" leaks at the
#: border. Measured worst case 777.3 of 800 = 2.8 % at blur=2.0.
RASTER_MASS_BUDGET = 0.95


def test_rasterize_stays_within_budget(primitives, clouds, record_property):
    """D1 is a deliberate speedup, so it gets a measured budget rather than equality."""
    _, _, got = _helpers.rasterize(clouds.ref[:, 0], clouds.ref[:, 1], **F.RASTER_PARAMS)
    expected = primitives["raster_ref"]
    cropped = np.asarray(got)[:, : expected.shape[1], : expected.shape[2]]

    for index, blur in enumerate(F.RASTER_PARAMS["blur"]):
        error = rel(cropped[index], expected[index])
        mass = float(np.asarray(got)[index].sum()) / float(expected[index].sum())
        correlation = float(np.corrcoef(cropped[index].ravel(), expected[index].ravel())[0, 1])
        record_property(f"rel_error_blur{blur}", error)
        record_property(f"mass_ratio_blur{blur}", mass)
        assert error < RASTER_BUDGET, f"blur={blur}: relL2 {error:.4%} exceeds {RASTER_BUDGET:.0%}"
        assert correlation > 0.99, f"blur={blur}: correlation {correlation:.5f}"
        assert RASTER_MASS_BUDGET < mass <= 1.0, f"blur={blur}: mass ratio {mass:.4f}"


def _residual(linear, translation, source, target) -> float:
    source, target = np.asarray(source), np.asarray(target)
    return float(np.linalg.norm(source @ np.asarray(linear).T + np.asarray(translation) - target))


def test_affine_from_points_is_equivalent_when_well_conditioned(primitives, record_property):
    """D7: the two are different estimators, not the same one implemented twice.

    Upstream solves the normal equations for the plain least-squares fit; skimage solves
    a Hartley-normalised homogeneous system by SVD, minimising algebraic rather than
    geometric error. So their coefficients differ by ~1e-3 even on clean landmarks, and
    asserting agreement would be wrong. What must hold is that neither is meaningfully
    worse at the job.
    """
    source = np.asarray(primitives["landmarks_query"])[:, ::-1]
    target = np.asarray(primitives["landmarks_ref"])[:, ::-1]
    linear, translation = _helpers.affine_from_points(jnp.asarray(source), jnp.asarray(target))

    ours = _residual(linear, translation, source, target)
    theirs = _residual(primitives["lt_well_L"], primitives["lt_well_T"], source, target)
    record_property("residual_squidpy", ours)
    record_property("residual_upstream", theirs)
    record_property("rel_error_L", rel(linear, primitives["lt_well_L"]))
    assert abs(ours - theirs) / theirs < 1e-2, f"squidpy {ours:.6f} vs upstream {theirs:.6f}"


def test_affine_from_points_survives_ill_conditioning(primitives, record_property):
    """D7, the half that matters: upstream's ``inv(XᵀX)`` collapses, skimage does not."""
    source, target = primitives["ill_src"], primitives["ill_dst"]
    linear, translation = _helpers.affine_from_points(jnp.asarray(source), jnp.asarray(target))

    ours = _residual(linear, translation, source, target)
    theirs = _residual(primitives["lt_ill_L"], primitives["lt_ill_T"], source, target)
    record_property("residual_squidpy", ours)
    record_property("residual_upstream", theirs)
    assert ours < theirs * 1e-6, (
        f"expected upstream to lose badly on near-collinear landmarks, got squidpy {ours:.3e} vs upstream {theirs:.3e}"
    )


# --------------------------------------------------------------------------------------
# The ledger has to stay in sync
# --------------------------------------------------------------------------------------


def test_divergences_doc_covers_all_xfails():
    """Every strict xfail cites a ledger row, and every cited row exists."""
    ledger = LEDGER.read_text()
    documented = set(re.findall(r"\*\*(D\d+)\*\*", ledger))
    assert documented, "no ledger rows found; STALIGN_DIVERGENCES.md changed shape"

    source = Path(__file__).read_text()
    cited = set(re.findall(r"ledger row (D\d+)", source))
    assert cited, "no test cites a ledger row"

    missing = cited - documented
    assert not missing, f"tests cite ledger rows that do not exist: {sorted(missing)}"
