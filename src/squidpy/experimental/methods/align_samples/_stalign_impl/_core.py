"""Core JAX implementation for experimental STalign point registration."""

from __future__ import annotations

from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

__all__ = ["jax_dtype", "lddmm", "transform_grid_row_col", "transform_points_row_col"]

#: Iteration at which the mixture-weight E step switches on (STalign.py:1233). Before
#: this the weights are frozen at their initial values, so the objective changes
#: definition here and its value jumps discontinuously.
MIXTURE_E_STEP_START = 50


def jax_dtype() -> jnp.dtype:
    """Resolve the active JAX float dtype at call time, not import time."""
    return jnp.float64 if jax.config.x64_enabled else jnp.float32


def _to_affine(linear: jax.Array, translation: jax.Array) -> jax.Array:
    return jnp.array(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=linear.dtype,
    )


def _grid_points(x: tuple[jax.Array, jax.Array]) -> jax.Array:
    yy, xx = jnp.meshgrid(x[0], x[1], indexing="ij")
    return jnp.stack((yy, xx))


def _interp(
    x: tuple[jax.Array, jax.Array],
    image: jax.Array,
    phii: jax.Array,
    *,
    mode: str = "nearest",
) -> jax.Array:
    """Interpolate a channels-first image on physical row-column coordinates."""
    arr = jnp.asarray(image)
    coords = jnp.asarray(phii)
    if coords.shape[0] != 2:
        raise ValueError(f"Expected interpolation coordinates to have leading axis of size 2, found `{coords.shape}`.")

    if arr.ndim == 2:
        arr = arr[None, ...]

    row_step = x[0][1] - x[0][0]
    col_step = x[1][1] - x[1][0]
    row_idx = (coords[0] - x[0][0]) / row_step
    col_idx = (coords[1] - x[1][0]) / col_step
    idx = jnp.stack((row_idx.reshape(-1), col_idx.reshape(-1)))

    def _sample(channel: jax.Array) -> jax.Array:
        values = jsp.ndimage.map_coordinates(channel, idx, order=1, mode=mode)
        return values.reshape(coords.shape[1:])

    return jax.vmap(_sample)(arr)


def transform_points_row_col(
    xv: tuple[jax.Array, jax.Array],
    velocity: jax.Array,
    affine: jax.Array,
    points: np.ndarray | jax.Array,
    *,
    direction: Literal["forward", "backward"] = "forward",
) -> jax.Array:
    pts = jnp.asarray(points)
    n_steps = velocity.shape[0]
    time_steps = range(n_steps)
    flow_sign = 1.0
    if direction == "backward":
        affine = jnp.linalg.inv(affine)
        pts = pts @ affine[:2, :2].T + affine[:2, -1]
        flow_sign = -1.0
        time_steps = reversed(time_steps)

    for t in time_steps:
        disp = _interp(
            xv,
            jnp.moveaxis(flow_sign * velocity[t], -1, 0),
            pts.T[:, :, None],
            mode="nearest",
        )[:, :, 0].T
        pts = pts + disp / n_steps

    if direction == "forward":
        pts = pts @ affine[:2, :2].T + affine[:2, -1]

    return pts


def transform_grid_row_col(
    axes: tuple[jax.Array, jax.Array],
    xv: tuple[jax.Array, jax.Array],
    velocity: jax.Array,
    affine: jax.Array,
    *,
    direction: Literal["forward", "backward"] = "forward",
) -> jax.Array:
    """Map the dense grid spanned by ``axes``, returned as ``(2, rows, columns)``."""
    grid = _grid_points(axes)
    points = jnp.moveaxis(grid, 0, -1).reshape((-1, 2))
    transformed = transform_points_row_col(xv, velocity, affine, points, direction=direction)
    return jnp.moveaxis(transformed.reshape((*grid.shape[1:], 2)), -1, 0)


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


def _axis(start: float, stop: float, step: float) -> np.ndarray:
    """``step``-spaced samples covering ``[start, stop)``, with a stable length.

    ``arange(start, stop, step)`` on floats derives its length from the arguments by
    floating-point division, so a ``stop`` that is itself a sum of floats can yield one
    more or one fewer sample than intended. Taking the count first makes the length a
    function of the interval alone.

    NumPy rather than JAX so the point-cloud rasterizer in ``._helpers`` can share it;
    the one JAX caller casts on the way out.
    """
    count = max(int(np.ceil((stop - start) / step)), 1)
    return start + step * np.arange(count, dtype=float)


def _build_velocity_grid(
    x_source: tuple[jax.Array, jax.Array], *, a: float, expand: float
) -> tuple[jax.Array, jax.Array]:
    minimum = np.array([x_source[0][0], x_source[1][0]], dtype=float)
    maximum = np.array([x_source[0][-1], x_source[1][-1]], dtype=float)
    center = (minimum + maximum) / 2.0
    half_width = (maximum - minimum) * expand / 2.0
    step = a * 0.5
    dtype = jax_dtype()
    return (
        jnp.asarray(_axis(center[0] - half_width[0], center[0] + half_width[0], step), dtype=dtype),
        jnp.asarray(_axis(center[1] - half_width[1], center[1] + half_width[1], step), dtype=dtype),
    )


def _build_regularizer(
    xv: tuple[jax.Array, jax.Array],
    *,
    a: float,
    p: float,
) -> tuple[jax.Array, jax.Array, float | jax.Array]:
    dv = jnp.array([xv[0][1] - xv[0][0], xv[1][1] - xv[1][0]])
    shape = (xv[0].shape[0], xv[1].shape[0])
    fy = jnp.arange(shape[0], dtype=xv[0].dtype) / (shape[0] * dv[0])
    fx = jnp.arange(shape[1], dtype=xv[1].dtype) / (shape[1] * dv[1])
    frequency_grid = jnp.stack(jnp.meshgrid(fy, fx, indexing="ij"), axis=-1)
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
    if estimate_muA:
        muA = jnp.sum(artifact_weights * target_image, axis=(-1, -2)) / jnp.maximum(jnp.sum(artifact_weights), 1e-12)
    if estimate_muB:
        muB = jnp.sum(background_weights * target_image, axis=(-1, -2)) / jnp.maximum(
            jnp.sum(background_weights), 1e-12
        )

    def _e_step() -> tuple[jax.Array, jax.Array, jax.Array]:
        weights = jnp.stack((match_weights, artifact_weights, background_weights))
        mixing = jnp.sum(weights, axis=(1, 2))
        mixing = mixing + jnp.max(mixing) * 1e-6
        mixing = mixing / jnp.sum(mixing)

        n_channels = target_image.shape[0]
        norm_match = (2.0 * np.pi * sigmaM**2) ** (n_channels / 2.0)
        norm_artifact = (2.0 * np.pi * sigmaA**2) ** (n_channels / 2.0)
        norm_background = (2.0 * np.pi * sigmaB**2) ** (n_channels / 2.0)

        match = mixing[0] * jnp.exp(-jnp.sum((transformed_source - target_image) ** 2, axis=0) / (2.0 * sigmaM**2))
        match = match / norm_match
        artifact = mixing[1] * jnp.exp(-jnp.sum((muA[:, None, None] - target_image) ** 2, axis=0) / (2.0 * sigmaA**2))
        artifact = artifact / norm_artifact
        background = mixing[2] * jnp.exp(-jnp.sum((muB[:, None, None] - target_image) ** 2, axis=0) / (2.0 * sigmaB**2))
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


def _lddmm_loss(
    linear: jax.Array,
    translation: jax.Array,
    velocity: jax.Array,
    *,
    x_source: tuple[jax.Array, jax.Array],
    source_image: jax.Array,
    x_target: tuple[jax.Array, jax.Array],
    target_image: jax.Array,
    xv: tuple[jax.Array, jax.Array],
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
    warped_source = _interp(x_source, source_image, source_grid, mode="nearest")
    contrast_source = _contrast_transform(warped_source, target_image, match_weights)

    match_energy = jnp.sum((contrast_source - target_image) ** 2 * match_weights) / (2.0 * sigmaM**2)
    fft_velocity = jnp.fft.fftn(velocity, axes=(1, 2))
    reg_energy = (
        jnp.sum(jnp.sum(jnp.abs(fft_velocity) ** 2, axis=(0, 3)) * ll)
        * dv_prod
        / 2.0
        / velocity.shape[1]
        / velocity.shape[2]
        / sigmaR**2
    )

    transformed_points = transform_points_row_col(xv, velocity, affine, points_source, direction="forward")
    if points_source.shape[0] == 0:
        point_energy = jnp.array(0.0, dtype=source_image.dtype)
    else:
        point_energy = jnp.sum((transformed_points - points_target) ** 2) / (2.0 * sigmaP**2)

    total = match_energy + reg_energy + point_energy
    return total, (contrast_source, transformed_points, match_energy, reg_energy, point_energy)


@partial(
    jax.jit,
    static_argnames=(
        "niter",
        "diffeo_start",
        "epL",
        "epT",
        "epV",
        "sigmaM",
        "sigmaA",
        "sigmaB",
        "sigmaR",
        "sigmaP",
        "tol",
        "patience",
        "estimate_muA",
        "estimate_muB",
    ),
)
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
    epL,
    epT,
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

    # Precomputed in Python so the two step sizes are bit-identical to `epL / 1.0` and
    # `epL / 10.0` -- the `(it >= diffeo_start) * 9` scaling at STalign.py:1205-1206.
    steps_before = (epL, epT)
    steps_after = (epL / 10.0, epT / 10.0)

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
        jnp.asarray(jnp.nan, dtype=dtype),
        source_landmarks,
        energies,
    )

    def _step(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        iteration, linear, translation, velocity, wm, wa, wb, muA, muB, _, _, energies = carry

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
        contrast_source, transformed_points, _, _, _ = aux

        diffeo = iteration >= diffeo_start
        step_linear = jnp.where(diffeo, steps_after[0], steps_before[0])
        step_translation = jnp.where(diffeo, steps_after[1], steps_before[1])
        linear = linear - step_linear * grad_linear
        translation = translation - step_translation * grad_translation

        grad_velocity = jnp.fft.ifftn(
            jnp.fft.fftn(grad_velocity, axes=(1, 2)) * kernel[None, ..., None],
            axes=(1, 2),
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
            energy,
            transformed_points,
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
    xI: tuple[np.ndarray | jax.Array, np.ndarray | jax.Array],
    I: np.ndarray | jax.Array,
    xJ: tuple[np.ndarray | jax.Array, np.ndarray | jax.Array],
    J: np.ndarray | jax.Array,
    *,
    L: np.ndarray | jax.Array,
    T: np.ndarray | jax.Array,
    initial_velocity: np.ndarray | jax.Array | None = None,
    velocity_grid: tuple[np.ndarray | jax.Array, np.ndarray | jax.Array] | None = None,
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

    Returns
    -------
    A dict with the fitted ``A``/``v``/``xv``, the mixture weights, the final energy
    ``E``, the per-iteration ``energies`` trace, and ``n_iter`` actually run.
    """
    x_source = (jnp.asarray(xI[0]), jnp.asarray(xI[1]))
    x_target = (jnp.asarray(xJ[0]), jnp.asarray(xJ[1]))
    source_image = jnp.asarray(I, dtype=jax_dtype())
    target_image = jnp.asarray(J, dtype=jax_dtype())
    linear = jnp.asarray(L, dtype=jax_dtype())
    translation = jnp.asarray(T, dtype=jax_dtype())

    if points_source is None:
        source_landmarks = jnp.zeros((0, 2), dtype=jax_dtype())
        target_landmarks = jnp.zeros((0, 2), dtype=jax_dtype())
    else:
        source_landmarks = jnp.asarray(points_source, dtype=jax_dtype())
        target_landmarks = jnp.asarray(points_target, dtype=jax_dtype())

    if (initial_velocity is None) != (velocity_grid is None):
        raise ValueError("Expected `initial_velocity` and `velocity_grid` to be provided together.")
    if velocity_grid is None:
        xv = _build_velocity_grid(x_source, a=a, expand=expand)
        velocity = jnp.zeros((nt, xv[0].shape[0], xv[1].shape[0], 2), dtype=jax_dtype())
    else:
        xv = (jnp.asarray(velocity_grid[0], dtype=jax_dtype()), jnp.asarray(velocity_grid[1], dtype=jax_dtype()))
        velocity = jnp.asarray(initial_velocity, dtype=jax_dtype())
        if velocity.ndim != 4:
            raise ValueError(f"Expected `initial_velocity` to be four-dimensional, found {velocity.shape}.")
        expected = (velocity.shape[0], xv[0].shape[0], xv[1].shape[0], 2)
        if velocity.shape != expected:
            raise ValueError(
                "Expected `initial_velocity` to have shape "
                f"(nt, {xv[0].shape[0]}, {xv[1].shape[0]}, 2), found {velocity.shape}."
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

    artifact_mean = mixture_mean(muA, name="muA", default=jnp.mean(target_image, axis=(1, 2)))
    background_mean = mixture_mean(muB, name="muB", default=jnp.zeros(n_channels, dtype=target_image.dtype))
    estimate_muA = muA is None
    estimate_muB = muB is None

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
        epL=epL,
        epT=epT,
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
        energy,
        transformed_points,
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
        "E": energy,
        "points": transformed_points,
        # Per-iteration objective, so a caller can tell a converged run from a diverged
        # one without running it again. Trailing entries stay NaN if `tol` stopped early.
        "energies": energies[:niter],
        "n_iter": completed,
    }
