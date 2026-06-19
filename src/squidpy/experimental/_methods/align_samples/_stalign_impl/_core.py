"""Core JAX implementation for experimental STalign point registration."""

from __future__ import annotations

from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

__all__ = ["jax_dtype", "lddmm", "transform_points_row_col"]


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


def _transform_grid_backward(
    x_target: tuple[jax.Array, jax.Array],
    xv: tuple[jax.Array, jax.Array],
    velocity: jax.Array,
    affine: jax.Array,
) -> jax.Array:
    target_grid = _grid_points(x_target)
    affine_inv = jnp.linalg.inv(affine)
    source_grid = jnp.einsum("ij,jhw->ihw", affine_inv[:2, :2], target_grid) + affine_inv[:2, -1][:, None, None]

    for t in range(velocity.shape[0] - 1, -1, -1):
        disp = _interp(xv, jnp.moveaxis(-velocity[t], -1, 0), source_grid, mode="nearest")
        source_grid = source_grid + disp / velocity.shape[0]

    return source_grid


def _contrast_transform(source_image: jax.Array, target_image: jax.Array, weights: jax.Array) -> jax.Array:
    flat_source = source_image.reshape(source_image.shape[0], -1)
    flat_target = target_image.reshape(target_image.shape[0], -1)
    flat_weights = weights.reshape(-1)

    design = jnp.concatenate((jnp.ones((1, flat_source.shape[1]), dtype=source_image.dtype), flat_source), axis=0)
    weighted_design = design * flat_weights[None, :]
    design_cov = weighted_design @ design.T
    target_cov = weighted_design @ flat_target.T
    regularized = design_cov + 0.1 * jnp.eye(design_cov.shape[0], dtype=design_cov.dtype)
    coefficients = jnp.linalg.solve(regularized, target_cov)
    return (coefficients.T @ design).reshape(target_image.shape)


def _build_velocity_grid(
    x_source: tuple[jax.Array, jax.Array], *, a: float, expand: float
) -> tuple[jax.Array, jax.Array]:
    minimum = jnp.array([x_source[0][0], x_source[1][0]])
    maximum = jnp.array([x_source[0][-1], x_source[1][-1]])
    center = (minimum + maximum) / 2.0
    half_width = (maximum - minimum) * expand / 2.0
    step = a * 0.5
    return (
        jnp.arange(center[0] - half_width[0], center[0] + half_width[0] + step, step),
        jnp.arange(center[1] - half_width[1], center[1] + half_width[1] + step, step),
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

    if iteration < 50:
        return match_weights, artifact_weights, background_weights, muA, muB

    weights = jnp.stack((match_weights, artifact_weights, background_weights))
    mixing = jnp.sum(weights, axis=(1, 2))
    mixing = mixing + jnp.max(mixing) * 1e-6
    mixing = mixing / jnp.sum(mixing)

    n_channels = target_image.shape[0]
    norm_match = (2.0 * np.pi * sigmaM**2) ** (n_channels / 2.0)
    norm_artifact = (2.0 * np.pi * sigmaA**2) ** (n_channels / 2.0)
    norm_background = (2.0 * np.pi * sigmaB**2) ** (n_channels / 2.0)

    match_weights = mixing[0] * jnp.exp(-jnp.sum((transformed_source - target_image) ** 2, axis=0) / (2.0 * sigmaM**2))
    match_weights = match_weights / norm_match
    artifact_weights = mixing[1] * jnp.exp(
        -jnp.sum((muA[:, None, None] - target_image) ** 2, axis=0) / (2.0 * sigmaA**2)
    )
    artifact_weights = artifact_weights / norm_artifact
    background_weights = mixing[2] * jnp.exp(
        -jnp.sum((muB[:, None, None] - target_image) ** 2, axis=0) / (2.0 * sigmaB**2)
    )
    background_weights = background_weights / norm_background

    total = match_weights + artifact_weights + background_weights
    total = total + jnp.max(total) * 1e-6
    return match_weights / total, artifact_weights / total, background_weights / total, muA, muB


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
    source_grid = _transform_grid_backward(x_target, xv, velocity, affine)
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


def lddmm(
    xI: tuple[np.ndarray | jax.Array, np.ndarray | jax.Array],
    I: np.ndarray | jax.Array,
    xJ: tuple[np.ndarray | jax.Array, np.ndarray | jax.Array],
    J: np.ndarray | jax.Array,
    *,
    L: np.ndarray | jax.Array,
    T: np.ndarray | jax.Array,
    points_source: np.ndarray | jax.Array | None = None,
    points_target: np.ndarray | jax.Array | None = None,
    a: float = 500.0,
    p: float = 2.0,
    expand: float = 2.0,
    nt: int = 3,
    niter: int = 5000,
    diffeo_start: int = 0,
    epL: float = 2e-8,
    epT: float = 2e-1,
    epV: float = 2e3,
    sigmaM: float = 1.0,
    sigmaB: float = 2.0,
    sigmaA: float = 5.0,
    sigmaR: float = 5e5,
    sigmaP: float = 2e1,
) -> dict[str, Any]:
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

    xv = _build_velocity_grid(x_source, a=a, expand=expand)
    velocity = jnp.zeros((nt, xv[0].shape[0], xv[1].shape[0], 2), dtype=jax_dtype())
    kernel, ll, dv_prod = _build_regularizer(xv, a=a, p=p)

    match_weights = jnp.full(target_image.shape[1:], 0.5, dtype=target_image.dtype)
    background_weights = jnp.full(target_image.shape[1:], 0.4, dtype=target_image.dtype)
    artifact_weights = jnp.full(target_image.shape[1:], 0.1, dtype=target_image.dtype)
    muA = jnp.mean(target_image, axis=(1, 2))
    muB = jnp.zeros_like(muA)
    estimate_muA = True
    estimate_muB = True

    loss_and_grad = jax.jit(jax.value_and_grad(_lddmm_loss, argnums=(0, 1, 2), has_aux=True))

    for iteration in range(niter):
        (energy, aux), (grad_linear, grad_translation, grad_velocity) = loss_and_grad(
            linear,
            translation,
            velocity,
            x_source=x_source,
            source_image=source_image,
            x_target=x_target,
            target_image=target_image,
            xv=xv,
            match_weights=match_weights,
            ll=ll,
            dv_prod=dv_prod,
            points_source=source_landmarks,
            points_target=target_landmarks,
            sigmaM=sigmaM,
            sigmaR=sigmaR,
            sigmaP=sigmaP,
        )
        contrast_source, transformed_points, _, _, _ = aux

        affine_scale = 1.0 + 9.0 * float(iteration >= diffeo_start)
        linear = linear - (epL / affine_scale) * grad_linear
        translation = translation - (epT / affine_scale) * grad_translation

        grad_velocity = jnp.fft.ifftn(
            jnp.fft.fftn(grad_velocity, axes=(1, 2)) * kernel[None, ..., None],
            axes=(1, 2),
        ).real
        if iteration >= diffeo_start:
            velocity = velocity - epV * grad_velocity

        if iteration % 5 == 0:
            match_weights, artifact_weights, background_weights, muA, muB = _update_mixture_weights(
                contrast_source,
                target_image,
                match_weights,
                artifact_weights,
                background_weights,
                sigmaM=sigmaM,
                sigmaA=sigmaA,
                sigmaB=sigmaB,
                estimate_muA=estimate_muA,
                estimate_muB=estimate_muB,
                muA=muA,
                muB=muB,
                iteration=iteration,
            )

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
    }
