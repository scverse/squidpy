"""Core JAX implementation for experimental STalign point registration."""

from __future__ import annotations

import os
from collections.abc import Sequence
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from squidpy.experimental.im import _rasterize_points

__all__ = ["interp", "jax_dtype", "lddmm", "reverse_axes", "transform_grid_row_col", "transform_points_row_col"]

#: An ordered per-axis coordinate vector, one entry per spatial axis. Two entries for the
#: 2D section-to-section fits, three for fitting a section into a reference volume; the
#: solver reads its rank off ``len(axes)`` rather than assuming either.
Axes = Sequence[jax.Array]

#: Iteration at which the mixture-weight E step switches on (STalign.py:1233). Before
#: this the weights are frozen at their initial values, so the objective changes
#: definition here and its value jumps discontinuously.
MIXTURE_E_STEP_START = 50


def jax_dtype() -> jnp.dtype:
    """Resolve the active JAX float dtype at call time, not import time."""
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def reverse_axes(ndim: int) -> jax.Array:
    """Homogeneous ``(ndim + 1, ndim + 1)`` matrix reversing the spatial axis order.

    The solver works in array order -- ``(y, x)`` at rank 2, ``(z, y, x)`` at rank 3 --
    while callers speak ``(x, y)`` / ``(x, y, z)``. Conjugating an affine by this matrix
    converts between the two conventions, and being its own inverse it serves both
    directions.
    """
    # Built in NumPy: it is a compile-time constant, and JAX rejects a plain list as a
    # multidimensional index.
    return jnp.asarray(np.eye(ndim + 1)[[*reversed(range(ndim)), ndim]], dtype=jax_dtype())


def _to_affine(linear: jax.Array, translation: jax.Array) -> jax.Array:
    ndim = linear.shape[0]
    dtype = linear.dtype
    return jnp.concatenate(
        (
            jnp.concatenate((linear, translation[:, None]), axis=1),
            jnp.concatenate((jnp.zeros((1, ndim), dtype=dtype), jnp.ones((1, 1), dtype=dtype)), axis=1),
        ),
        axis=0,
    )


def _grid_points(x: Axes) -> jax.Array:
    return jnp.stack(jnp.meshgrid(*x, indexing="ij"))


def interp(
    x: Axes,
    image: jax.Array,
    phii: jax.Array,
    *,
    order: int = 1,
) -> jax.Array:
    """Interpolate a channels-first image on physical coordinates, at any rank.

    ``mode`` is the out-of-domain rule, ``"nearest"`` being upstream's
    ``padding_mode="border"``. ``order`` is the interpolation: 1 for the linear sampling
    the objective uses, 0 for nearest-neighbour, which is what reading integer structure
    ids off an annotation volume requires -- averaging two ids yields a third, unrelated
    one.
    """
    arr = jnp.asarray(image)
    coords = jnp.asarray(phii)
    ndim = len(x)
    if coords.shape[0] != ndim:
        raise ValueError(
            f"Expected interpolation coordinates to have leading axis of size {ndim}, found `{coords.shape}`."
        )
    # A single-sample axis has no step to divide by, and `x[axis][1]` on it does not raise
    # -- JAX clamps out-of-bounds indices, so the step comes out as zero and every sampled
    # value is silently inf or nan. Guarded here because this is the one place every
    # caller's coordinates get converted to indices.
    for axis, values in enumerate(x):
        if values.shape[0] < 2:
            raise ValueError(
                f"Expected interpolation axis {axis} to have at least two coordinates, found "
                f"{values.shape[0]}. A single sample defines no spacing."
            )

    if arr.ndim == ndim:
        arr = arr[None, ...]

    idx = jnp.stack([((coords[axis] - x[axis][0]) / (x[axis][1] - x[axis][0])).reshape(-1) for axis in range(ndim)])

    def _sample(channel: jax.Array) -> jax.Array:
        values = jsp.ndimage.map_coordinates(channel, idx, order=order, mode="nearest")
        return values.reshape(coords.shape[1:])

    return jax.vmap(_sample)(arr)


def transform_points_row_col(
    xv: Axes,
    velocity: jax.Array,
    affine: jax.Array,
    points: np.ndarray | jax.Array,
    *,
    direction: Literal["forward", "backward"] = "forward",
) -> jax.Array:
    pts = jnp.asarray(points)
    ndim = pts.shape[-1]
    n_steps = velocity.shape[0]
    time_steps = range(n_steps)
    flow_sign = 1.0
    if direction == "backward":
        affine = jnp.linalg.inv(affine)
        pts = pts @ affine[:ndim, :ndim].T + affine[:ndim, -1]
        flow_sign = -1.0
        time_steps = reversed(time_steps)

    for t in time_steps:
        disp = interp(
            xv,
            jnp.moveaxis(flow_sign * velocity[t], -1, 0),
            pts.T[:, :, None],
        )[:, :, 0].T
        pts = pts + disp / n_steps

    if direction == "forward":
        pts = pts @ affine[:ndim, :ndim].T + affine[:ndim, -1]

    return pts


def transform_grid_row_col(
    axes: Axes,
    xv: Axes,
    velocity: jax.Array,
    affine: jax.Array,
    *,
    direction: Literal["forward", "backward"] = "forward",
) -> jax.Array:
    """Map the dense grid spanned by ``axes``, returned as ``(len(axes), *grid_shape)``."""
    ndim = len(axes)
    grid = _grid_points(axes)
    points = jnp.moveaxis(grid, 0, -1).reshape((-1, ndim))
    transformed = transform_points_row_col(xv, velocity, affine, points, direction=direction)
    return jnp.moveaxis(transformed.reshape((*grid.shape[1:], ndim)), -1, 0)


def _contrast_transform(source_image: jax.Array, target_image: jax.Array, weights: jax.Array) -> jax.Array:
    """Weighted ridge fit mapping source intensities onto target intensities.

    The coefficients are held constant with respect to the optimisation. This is an
    expectation-maximisation M step, solved exactly at the current estimate, not a
    quantity to descend on -- differentiating through the solve would silently turn the
    alternating minimisation into a joint one and change the search direction.
    """
    flat_source = source_image.reshape(source_image.shape[0], -1)
    flat_target = target_image.reshape(target_image.shape[0], -1)
    flat_weights = weights.reshape(-1)

    design = jnp.concatenate((jnp.ones((1, flat_source.shape[1]), dtype=source_image.dtype), flat_source), axis=0)
    weighted_design = design * flat_weights[None, :]
    design_cov = weighted_design @ design.T
    target_cov = weighted_design @ flat_target.T
    regularized = design_cov + 0.1 * jnp.eye(design_cov.shape[0], dtype=design_cov.dtype)
    coefficients = jax.lax.stop_gradient(jnp.linalg.solve(regularized, target_cov))
    return (coefficients.T @ design).reshape(target_image.shape)


#: The stable-length axis builder, shared with the point rasteriser so the velocity grid
#: and the density grid cannot drift apart. It lives in :mod:`squidpy.experimental.im`
#: because it is pure NumPy and that module carries no JAX dependency; the one JAX caller
#: here casts on the way out.
_axis = _rasterize_points.axis


def _build_velocity_grid(x_source: Axes, *, a: float, expand: float) -> tuple[jax.Array, ...]:
    minimum = np.array([axis[0] for axis in x_source], dtype=float)
    maximum = np.array([axis[-1] for axis in x_source], dtype=float)
    center = (minimum + maximum) / 2.0
    half_width = (maximum - minimum) * expand / 2.0
    step = a * 0.5
    dtype = jax_dtype()
    return tuple(
        jnp.asarray(_axis(mid - half, mid + half, step), dtype=dtype)
        for mid, half in zip(center, half_width, strict=True)
    )


def _build_regularizer(
    xv: Axes,
    *,
    a: float,
    p: float,
) -> tuple[jax.Array, jax.Array, float | jax.Array]:
    dv = jnp.array([axis[1] - axis[0] for axis in xv])
    shape = tuple(axis.shape[0] for axis in xv)
    frequencies = [
        jnp.arange(size, dtype=axis.dtype) / (size * step) for size, axis, step in zip(shape, xv, dv, strict=True)
    ]
    frequency_grid = jnp.stack(jnp.meshgrid(*frequencies, indexing="ij"), axis=-1)
    ll = (1.0 + 2.0 * a**2 * jnp.sum((1.0 - jnp.cos(2.0 * np.pi * frequency_grid * dv)) / (dv**2), axis=-1)) ** (
        2.0 * p
    )
    kernel = 1.0 / ll
    dv_prod = jnp.prod(dv)
    return kernel, ll, dv_prod


def _update_mixture_weights(
    transformed_source: jax.Array,
    target_image: jax.Array,
    match_weights: jax.Array,
    artifact_weights: jax.Array,
    background_weights: jax.Array,
    *,
    sigmaM: float,
    sigmaA: float,
    sigmaB: float,
    estimate_muA: bool,
    estimate_muB: bool,
    muA: jax.Array,
    muB: jax.Array,
    iteration: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    # Every spatial axis, however many there are: `(-1, -2)` at rank 2, `(-1, -2, -3)`
    # when the target is a volume. What survives the sum is the per-channel mean.
    spatial_axes = tuple(range(1, target_image.ndim))
    if estimate_muA:
        muA = jnp.sum(artifact_weights * target_image, axis=spatial_axes) / jnp.maximum(
            jnp.sum(artifact_weights), 1e-12
        )
    if estimate_muB:
        muB = jnp.sum(background_weights * target_image, axis=spatial_axes) / jnp.maximum(
            jnp.sum(background_weights), 1e-12
        )

    def _channelwise(mean: jax.Array) -> jax.Array:
        """``(c,)`` means broadcast against a ``(c, *spatial)`` image."""
        return mean.reshape((-1, *([1] * len(spatial_axes))))

    def _e_step() -> tuple[jax.Array, jax.Array, jax.Array]:
        weights = jnp.stack((match_weights, artifact_weights, background_weights))
        mixing = jnp.sum(weights, axis=tuple(range(1, weights.ndim)))
        mixing = mixing + jnp.max(mixing) * 1e-6
        mixing = mixing / jnp.sum(mixing)

        n_channels = target_image.shape[0]
        norm_match = (2.0 * np.pi * sigmaM**2) ** (n_channels / 2.0)
        norm_artifact = (2.0 * np.pi * sigmaA**2) ** (n_channels / 2.0)
        norm_background = (2.0 * np.pi * sigmaB**2) ** (n_channels / 2.0)

        match = mixing[0] * jnp.exp(-jnp.sum((transformed_source - target_image) ** 2, axis=0) / (2.0 * sigmaM**2))
        match = match / norm_match
        artifact = mixing[1] * jnp.exp(-jnp.sum((_channelwise(muA) - target_image) ** 2, axis=0) / (2.0 * sigmaA**2))
        artifact = artifact / norm_artifact
        background = mixing[2] * jnp.exp(-jnp.sum((_channelwise(muB) - target_image) ** 2, axis=0) / (2.0 * sigmaB**2))
        background = background / norm_background

        total = match + artifact + background
        total = total + jnp.max(total) * 1e-6
        return match / total, artifact / total, background / total

    # Before the E step switches on the weights stay at their initial 0.5/0.4/0.1,
    # while muA/muB are still re-estimated every 5th iteration.
    match_weights, artifact_weights, background_weights = jax.lax.cond(
        iteration >= MIXTURE_E_STEP_START,
        _e_step,
        lambda: (match_weights, artifact_weights, background_weights),
    )
    return match_weights, artifact_weights, background_weights, muA, muB


def _velocity_axes(velocity: jax.Array) -> tuple[int, ...]:
    """The spatial axes of a ``(nt, *grid, ndim)`` velocity field.

    Both the regularisation energy and the Sobolev smoothing of its gradient have to
    transform the *same* axes, or the regulariser being descended on is not the one being
    measured. Upstream's 3D path transforms only two of the three spatial axes in the
    energy while smoothing over all three (STalign.py:1215 vs its 3D energy term), and
    autograd carries that discrepancy into the search direction; squidpy uses every
    spatial axis in both places. At rank 2 the two readings coincide, so the 2D path is
    unaffected.
    """
    return tuple(range(1, velocity.ndim - 1))


#: Comparison-only escape hatch, read once at trace time so it stays a static branch.
#: Set it and the regularisation *energy* transforms only the first two spatial axes,
#: reproducing upstream's rank-3 line (``STalign.py:1504``) instead of the correct one.
#: Never set this for real work -- see :func:`_reg_energy_axes`.
_UPSTREAM_REG_ENERGY_AXES = "SQUIDPY_STALIGN_UPSTREAM_REG_ENERGY_AXES"


def _reg_energy_axes(velocity: jax.Array) -> tuple[int, ...]:
    """The spatial axes the regularisation *energy* transforms.

    Every spatial axis, matching :func:`_velocity_axes` and therefore the Sobolev smoothing
    applied to this energy's gradient. That agreement is the whole point, so this returns the
    same axes -- unless :envvar:`SQUIDPY_STALIGN_UPSTREAM_REG_ENERGY_AXES` is set, in which
    case it returns only the first two and squidpy reproduces upstream's rank-3 energy.

    That switch exists to *measure* the divergence, not to offer it. Upstream's rank-3 energy
    line is byte-identical to its rank-2 one, where two axes is all of them; the gradient it
    descends was updated to three (``:1527``) and the energy was not, so it reports one
    objective and descends another. Reproducing it here is the only way to attribute a
    volume-to-section difference to that line rather than to everything else that differs at
    rank 3. See row D11 of squidpy-ports' divergence ledger.

    Inert at rank 2, where the first two spatial axes are every spatial axis.
    """
    axes = _velocity_axes(velocity)
    if os.environ.get(_UPSTREAM_REG_ENERGY_AXES):
        return axes[:2]
    return axes


def _lddmm_loss(
    linear: jax.Array,
    translation: jax.Array,
    velocity: jax.Array,
    *,
    x_source: Axes,
    source_image: jax.Array,
    x_target: Axes,
    target_image: jax.Array,
    xv: Axes,
    match_weights: jax.Array,
    ll: jax.Array,
    dv_prod: float | jax.Array,
    points_source: jax.Array,
    points_target: jax.Array,
    sigmaM: float,
    sigmaR: float,
    sigmaP: float,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]]:
    affine = _to_affine(linear, translation)
    source_grid = transform_grid_row_col(x_target, xv, velocity, affine, direction="backward")
    warped_source = interp(x_source, source_image, source_grid)
    contrast_source = _contrast_transform(warped_source, target_image, match_weights)

    match_energy = jnp.sum((contrast_source - target_image) ** 2 * match_weights) / (2.0 * sigmaM**2)
    spatial = _reg_energy_axes(velocity)
    fft_velocity = jnp.fft.fftn(velocity, axes=spatial)
    # Sum over time and the vector component, leaving one term per velocity-grid cell.
    reg_energy = jnp.sum(jnp.sum(jnp.abs(fft_velocity) ** 2, axis=(0, velocity.ndim - 1)) * ll) * dv_prod / 2.0
    # One division per axis rather than one by their product: `(x / n) / m` and
    # `x / (n * m)` differ in the last bit, and the 2D path is pinned bit-for-bit against
    # the reference bundle.
    # One division per *transformed* axis, in axis order. With the default axes this is
    # `velocity.shape[1:-1]` exactly as before, so the rank-2 path stays bit-for-bit; under
    # the upstream switch it divides by two of three, which is upstream's `/v.shape[1]
    # /v.shape[2]` (`:1504`). `(x / n) / m` and `x / (n * m)` differ in the last bit and the
    # 2D path is pinned against the reference bundle, so the loop stays a loop.
    for axis in spatial:
        reg_energy = reg_energy / velocity.shape[axis]
    reg_energy = reg_energy / sigmaR**2

    transformed_points = transform_points_row_col(xv, velocity, affine, points_source, direction="forward")
    if points_source.shape[0] == 0:
        point_energy = jnp.array(0.0, dtype=source_image.dtype)
    else:
        point_energy = jnp.sum((transformed_points - points_target) ** 2) / (2.0 * sigmaP**2)

    total = match_energy + reg_energy + point_energy
    return total, (contrast_source, transformed_points, match_energy, reg_energy, point_energy)


# Only what genuinely cannot be traced is static: `niter` sizes the energy trace, `tol`
# is tested against `None` in Python, and `estimate_mu*` pick a branch in Python. The
# tuning scalars are traced, so retuning any of them reuses the compiled loop instead of
# paying for a fresh trace of `value_and_grad` through the interpolation and FFTs.
@partial(jax.jit, static_argnames=("niter", "tol", "estimate_muA", "estimate_muB"))
def _lddmm_run(
    linear,
    translation,
    velocity,
    match_weights,
    artifact_weights,
    background_weights,
    muA,
    muB,
    *,
    x_source,
    source_image,
    x_target,
    target_image,
    xv,
    kernel,
    ll,
    dv_prod,
    source_landmarks,
    target_landmarks,
    niter,
    diffeo_start,
    steps_before,
    steps_after,
    epV,
    sigmaM,
    sigmaA,
    sigmaB,
    sigmaR,
    sigmaP,
    tol,
    patience,
    estimate_muA,
    estimate_muB,
):
    """The gradient descent, as one compiled loop.

    Jitted as a whole rather than per-iteration: `lax.while_loop` outside a `jit` re-traces
    its body on every call, and tracing `value_and_grad` through the interpolation and FFTs
    costs about as much as a thousand iterations of actually running it.
    """
    loss_and_grad = jax.value_and_grad(_lddmm_loss, argnums=(0, 1, 2), has_aux=True)

    dtype = jax_dtype()
    # `niter=0` means "evaluate the initial state and stop"; the trace still needs a
    # slot so the carry has a fixed shape.
    energies = jnp.full((max(niter, 1),), jnp.nan, dtype=dtype)
    initial = (
        jnp.asarray(0),
        linear,
        translation,
        velocity,
        match_weights,
        artifact_weights,
        background_weights,
        muA,
        muB,
        energies,
    )

    def _step(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        iteration, linear, translation, velocity, wm, wa, wb, muA, muB, energies = carry

        (energy, aux), (grad_linear, grad_translation, grad_velocity) = loss_and_grad(
            linear,
            translation,
            velocity,
            x_source=x_source,
            source_image=source_image,
            x_target=x_target,
            target_image=target_image,
            xv=xv,
            match_weights=wm,
            ll=ll,
            dv_prod=dv_prod,
            points_source=source_landmarks,
            points_target=target_landmarks,
            sigmaM=sigmaM,
            sigmaR=sigmaR,
            sigmaP=sigmaP,
        )
        contrast_source, *_ = aux

        diffeo = iteration >= diffeo_start
        step_linear = jnp.where(diffeo, steps_after[0], steps_before[0])
        step_translation = jnp.where(diffeo, steps_after[1], steps_before[1])
        linear = linear - step_linear * grad_linear
        translation = translation - step_translation * grad_translation

        grad_velocity = jnp.fft.ifftn(
            jnp.fft.fftn(grad_velocity, axes=_velocity_axes(velocity)) * kernel[None, ..., None],
            axes=_velocity_axes(velocity),
        ).real
        velocity = jnp.where(diffeo, velocity - epV * grad_velocity, velocity)

        wm, wa, wb, muA, muB = jax.lax.cond(
            iteration % 5 == 0,
            lambda: _update_mixture_weights(
                contrast_source,
                target_image,
                wm,
                wa,
                wb,
                sigmaM=sigmaM,
                sigmaA=sigmaA,
                sigmaB=sigmaB,
                estimate_muA=estimate_muA,
                estimate_muB=estimate_muB,
                muA=muA,
                muB=muB,
                iteration=iteration,
            ),
            lambda: (wm, wa, wb, muA, muB),
        )
        return (
            iteration + 1,
            linear,
            translation,
            velocity,
            wm,
            wa,
            wb,
            muA,
            muB,
            energies.at[iteration].set(energy),
        )

    def _keep_going(carry: tuple[Any, ...]) -> jax.Array:
        iteration, energies = carry[0], carry[-1]
        if tol is None:
            return iteration < niter
        # Compare against `patience` iterations ago rather than the previous step: the
        # weights only move every 5th iteration, so consecutive energies plateau and
        # then jump, and a one-step test would stop on the plateau.
        recent = energies[jnp.maximum(iteration - 1, 0)]
        older = energies[jnp.maximum(iteration - 1 - patience, 0)]
        improving = (older - recent) > tol * jnp.abs(older)
        # The whole comparison window has to sit after the E step switches on. The
        # objective changes definition at `MIXTURE_E_STEP_START` and its value jumps
        # upward there, which reads as "no longer improving" to any stopping rule; the
        # first energy computed with the new weights is at `MIXTURE_E_STEP_START + 1`, so
        # the oldest index we may look at is that, hence the `+ 2` once the window and the
        # one-step lag are accounted for.
        warming_up = iteration < MIXTURE_E_STEP_START + 2 + patience
        return (iteration < niter) & (warming_up | improving)

    return jax.lax.while_loop(_keep_going, _step, initial)


def lddmm(
    xI: Sequence[np.ndarray | jax.Array],
    I: np.ndarray | jax.Array,
    xJ: Sequence[np.ndarray | jax.Array],
    J: np.ndarray | jax.Array,
    *,
    L: np.ndarray | jax.Array,
    T: np.ndarray | jax.Array,
    initial_velocity: np.ndarray | jax.Array | None = None,
    velocity_grid: Sequence[np.ndarray | jax.Array] | None = None,
    points_source: np.ndarray | jax.Array | None = None,
    points_target: np.ndarray | jax.Array | None = None,
    a: float,
    p: float,
    expand: float,
    nt: int,
    niter: int,
    diffeo_start: int,
    epL: float,
    epT: float,
    epV: float,
    sigmaM: float,
    sigmaB: float,
    sigmaA: float,
    sigmaR: float,
    sigmaP: float,
    muA: np.ndarray | jax.Array | None,
    muB: np.ndarray | jax.Array | None,
    tol: float | None,
    patience: int,
) -> dict[str, Any]:
    """Fit an LDDMM registration of ``I`` onto ``J`` by gradient descent.

    The whole descent runs as a single ``lax.while_loop``, so XLA compiles the loop body
    once and fuses across it instead of paying per-iteration dispatch from Python.

    Every tuning parameter is required: the defaults live in ``_stalign._SOLVER_DEFAULTS``
    so they exist in exactly one place, and the ``fit_stalign_*`` wrappers resolve them
    before calling in. Only the ``initial_velocity`` / ``velocity_grid`` / ``points_*``
    sentinels default to ``None``, which means "not supplied" rather than a tuned value.

    Parameters
    ----------
    tol
        Stop once the objective's relative improvement over the last ``patience``
        iterations falls below this. ``None`` always runs the full ``niter``.
    patience
        Window for the ``tol`` test. Compared against ``patience`` iterations ago rather
        than the previous step because the mixture weights only move every 5th iteration,
        so the objective plateaus and then jumps -- a one-step test would stop on a
        plateau.

    The rank is read off ``len(xI)``: two axes registers a section onto a section, three
    registers a section into a reference volume. Nothing in the descent is rank-specific.

    Returns
    -------
    A dict with the fitted ``A``/``v``/``xv``, the mixture weights, the per-iteration
    ``energies`` trace, and ``n_iter`` actually run.
    """
    x_source = tuple(jnp.asarray(axis) for axis in xI)
    x_target = tuple(jnp.asarray(axis) for axis in xJ)
    ndim = len(x_source)
    if len(x_target) != ndim:
        raise ValueError(f"Expected `xI` and `xJ` to have the same number of axes, found {ndim} and {len(x_target)}.")
    source_image = jnp.asarray(I, dtype=jax_dtype())
    target_image = jnp.asarray(J, dtype=jax_dtype())
    linear = jnp.asarray(L, dtype=jax_dtype())
    translation = jnp.asarray(T, dtype=jax_dtype())

    if points_source is None:
        source_landmarks = jnp.zeros((0, ndim), dtype=jax_dtype())
        target_landmarks = jnp.zeros((0, ndim), dtype=jax_dtype())
    else:
        source_landmarks = jnp.asarray(points_source, dtype=jax_dtype())
        target_landmarks = jnp.asarray(points_target, dtype=jax_dtype())

    if (initial_velocity is None) != (velocity_grid is None):
        raise ValueError("Expected `initial_velocity` and `velocity_grid` to be provided together.")
    if velocity_grid is None:
        xv = _build_velocity_grid(x_source, a=a, expand=expand)
        velocity = jnp.zeros((nt, *(axis.shape[0] for axis in xv), ndim), dtype=jax_dtype())
    else:
        xv = tuple(jnp.asarray(axis, dtype=jax_dtype()) for axis in velocity_grid)
        if len(xv) != ndim:
            raise ValueError(f"Expected `velocity_grid` to have {ndim} axes, found {len(xv)}.")
        velocity = jnp.asarray(initial_velocity, dtype=jax_dtype())
        expected = (velocity.shape[0] if velocity.ndim else 0, *(axis.shape[0] for axis in xv), ndim)
        if velocity.shape != expected:
            raise ValueError(
                f"Expected `initial_velocity` to have shape (nt, "
                f"{', '.join(str(axis.shape[0]) for axis in xv)}, {ndim}), found {velocity.shape}."
            )
    kernel, ll, dv_prod = _build_regularizer(xv, a=a, p=p)

    match_weights = jnp.full(target_image.shape[1:], 0.5, dtype=target_image.dtype)
    background_weights = jnp.full(target_image.shape[1:], 0.4, dtype=target_image.dtype)
    artifact_weights = jnp.full(target_image.shape[1:], 0.1, dtype=target_image.dtype)
    n_channels = target_image.shape[0]

    def mixture_mean(value: np.ndarray | jax.Array | None, *, name: str, default: jax.Array) -> jax.Array:
        if value is None:
            return default
        mean = jnp.asarray(value, dtype=target_image.dtype)
        if mean.shape != (n_channels,):
            raise ValueError(f"Expected `{name}` to have shape ({n_channels},), found {mean.shape}.")
        return mean

    spatial_axes = tuple(range(1, target_image.ndim))
    artifact_mean = mixture_mean(muA, name="muA", default=jnp.mean(target_image, axis=spatial_axes))
    background_mean = mixture_mean(muB, name="muB", default=jnp.zeros(n_channels, dtype=target_image.dtype))
    estimate_muA = muA is None
    estimate_muB = muB is None

    # Precomputed here in Python so the diffeo-phase step sizes stay bit-identical to
    # `epL / 10.0` -- the `(it >= diffeo_start) * 9` scaling at STalign.py:1205-1206 --
    # rather than being derived from a traced scalar at the solver's active precision.
    final = _lddmm_run(
        linear,
        translation,
        velocity,
        match_weights,
        artifact_weights,
        background_weights,
        artifact_mean,
        background_mean,
        x_source=x_source,
        source_image=source_image,
        x_target=x_target,
        target_image=target_image,
        xv=xv,
        kernel=kernel,
        ll=ll,
        dv_prod=dv_prod,
        source_landmarks=source_landmarks,
        target_landmarks=target_landmarks,
        niter=niter,
        diffeo_start=diffeo_start,
        steps_before=(epL, epT),
        steps_after=(epL / 10.0, epT / 10.0),
        epV=epV,
        sigmaM=sigmaM,
        sigmaA=sigmaA,
        sigmaB=sigmaB,
        sigmaR=sigmaR,
        sigmaP=sigmaP,
        tol=tol,
        patience=patience,
        estimate_muA=estimate_muA,
        estimate_muB=estimate_muB,
    )
    (
        completed,
        linear,
        translation,
        velocity,
        match_weights,
        artifact_weights,
        background_weights,
        muA,
        muB,
        energies,
    ) = final

    affine = _to_affine(linear, translation)
    return {
        "A": affine,
        "v": velocity,
        "xv": xv,
        "WM": match_weights,
        "WB": background_weights,
        "WA": artifact_weights,
        # Per-iteration objective, so a caller can tell a converged run from a diverged
        # one without running it again. Trailing entries stay NaN if `tol` stopped early.
        "energies": energies[:niter],
        "n_iter": completed,
    }
