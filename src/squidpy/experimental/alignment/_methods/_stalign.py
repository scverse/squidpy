"""STalign alignment method implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from anndata import AnnData

if TYPE_CHECKING:
    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any


JAX_DTYPE = jnp.float64 if jax.config.x64_enabled else jnp.float32


def _to_affine(linear: Any, translation: Any) -> Any:
    return jnp.array(
        [
            [linear[0, 0], linear[0, 1], translation[0]],
            [linear[1, 0], linear[1, 1], translation[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=linear.dtype,
    )


def _grid_points(x: tuple[Any, Any]) -> Any:
    yy, xx = jnp.meshgrid(x[0], x[1], indexing="ij")
    return jnp.stack((yy, xx))


def _interp(
    x: tuple[Any, Any],
    image: Any,
    phii: Any,
    *,
    mode: str = "nearest",
) -> Any:
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

    def _sample(channel: Any) -> Any:
        values = jsp.ndimage.map_coordinates(channel, idx, order=1, mode=mode)
        return values.reshape(coords.shape[1:])

    return jax.vmap(_sample)(arr)


def transform_points_row_col(
    xv: tuple[Any, Any],
    velocity: Any,
    affine: Any,
    points: Any,
    *,
    direction: Literal["forward", "backward"] = "forward",
) -> Any:
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
    x_target: tuple[Any, Any],
    xv: tuple[Any, Any],
    velocity: Any,
    affine: Any,
) -> Any:
    target_grid = _grid_points(x_target)
    affine_inv = jnp.linalg.inv(affine)
    source_grid = jnp.einsum("ij,jhw->ihw", affine_inv[:2, :2], target_grid) + affine_inv[:2, -1][:, None, None]

    for t in range(velocity.shape[0] - 1, -1, -1):
        disp = _interp(xv, jnp.moveaxis(-velocity[t], -1, 0), source_grid, mode="nearest")
        source_grid = source_grid + disp / velocity.shape[0]

    return source_grid


def _contrast_transform(source_image: Any, target_image: Any, weights: Any) -> Any:
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


def _build_velocity_grid(x_source: tuple[Any, Any], *, a: float, expand: float) -> tuple[Any, Any]:
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
    xv: tuple[Any, Any],
    *,
    a: float,
    p: float,
) -> tuple[Any, Any, Any]:
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
    transformed_source: Any,
    target_image: Any,
    match_weights: Any,
    artifact_weights: Any,
    background_weights: Any,
    *,
    sigmaM: float,
    sigmaA: float,
    sigmaB: float,
    estimate_muA: bool,
    estimate_muB: bool,
    muA: Any,
    muB: Any,
    iteration: int,
) -> tuple[Any, Any, Any, Any, Any]:
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
    linear: Any,
    translation: Any,
    velocity: Any,
    *,
    x_source: tuple[Any, Any],
    source_image: Any,
    x_target: tuple[Any, Any],
    target_image: Any,
    xv: tuple[Any, Any],
    match_weights: Any,
    ll: Any,
    dv_prod: Any,
    points_source: Any,
    points_target: Any,
    sigmaM: float,
    sigmaR: float,
    sigmaP: float,
) -> tuple[Any, tuple[Any, Any, Any, Any, Any]]:
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
    xI: tuple[Any, Any],
    I: Any,
    xJ: tuple[Any, Any],
    J: Any,
    *,
    L: Any,
    T: Any,
    points_source: Any | None = None,
    points_target: Any | None = None,
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
    source_image = jnp.asarray(I, dtype=JAX_DTYPE)
    target_image = jnp.asarray(J, dtype=JAX_DTYPE)
    linear = jnp.asarray(L, dtype=JAX_DTYPE)
    translation = jnp.asarray(T, dtype=JAX_DTYPE)

    if points_source is None:
        source_landmarks = jnp.zeros((0, 2), dtype=JAX_DTYPE)
        target_landmarks = jnp.zeros((0, 2), dtype=JAX_DTYPE)
    else:
        source_landmarks = jnp.asarray(points_source, dtype=JAX_DTYPE)
        target_landmarks = jnp.asarray(points_target, dtype=JAX_DTYPE)

    xv = _build_velocity_grid(x_source, a=a, expand=expand)
    velocity = jnp.zeros((nt, xv[0].shape[0], xv[1].shape[0], 2), dtype=JAX_DTYPE)
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


PointOrder = Literal["row_col", "xy"]


def _validate_points(points: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected `{name}` to have shape `(n, 2)`, found `{arr.shape}`.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Expected `{name}` to contain only finite values.")
    return arr


def extract_points(adata: AnnData, key: str = "spatial") -> np.ndarray:
    """Return a validated coordinate array from ``adata.obsm``."""
    if key not in adata.obsm:
        raise KeyError(f"Key `{key}` not found in `adata.obsm`.")

    return _validate_points(np.asarray(adata.obsm[key]), name=f"adata.obsm[{key!r}]")


def extract_landmarks(adata: AnnData, key: str) -> np.ndarray:
    """Return a validated landmark array from ``adata.obsm`` or ``adata.uns``."""
    if key in adata.obsm:
        arr = np.asarray(adata.obsm[key], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError(f"Expected `adata.obsm[{key!r}]` to have shape `(n, 2)`, found `{arr.shape}`.")
        mask = np.all(np.isfinite(arr), axis=1)
        landmarks = arr[mask]
        if landmarks.size == 0:
            raise ValueError(f"No finite landmark rows were found in `adata.obsm[{key!r}]`.")
        return landmarks

    if key in adata.uns:
        return _validate_points(np.asarray(adata.uns[key]), name=f"adata.uns[{key!r}]")

    raise KeyError(f"Key `{key}` not found in `adata.obsm` or `adata.uns`.")


# TODO: are these duplicated? I would imagine its
# better to keep image transform functions under some place


def to_row_col(points: np.ndarray, *, point_order: PointOrder) -> np.ndarray:
    """Convert coordinates to row-column order."""
    arr = _validate_points(points, name="points")
    if point_order == "row_col":
        return arr
    if point_order == "xy":
        return arr[:, [1, 0]]
    raise ValueError(f"Unknown `point_order`: `{point_order}`.")


def from_row_col(points: np.ndarray, *, point_order: PointOrder) -> np.ndarray:
    """Convert row-column coordinates to the requested order."""
    arr = _validate_points(points, name="points")
    if point_order == "row_col":
        return arr
    if point_order == "xy":
        return arr[:, [1, 0]]
    raise ValueError(f"Unknown `point_order`: `{point_order}`.")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmin, vmax):
        return np.ones_like(values, dtype=float)
    return (values - vmin) / (vmax - vmin)


def rasterize(
    x: np.ndarray,
    y: np.ndarray,
    *,
    g: np.ndarray | None = None,
    dx: float = 30.0,
    blur: float | list[float] = 1.0,
    expand: float = 1.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rasterize a point cloud into a multi-scale density image."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Expected `x` and `y` to be 1D arrays with the same length.")
    if x.size == 0:
        raise ValueError("Expected at least one point to rasterize.")
    if dx <= 0:
        raise ValueError("Expected `dx` to be positive.")
    if expand <= 0:
        raise ValueError("Expected `expand` to be positive.")

    blur_values = np.atleast_1d(np.asarray(blur, dtype=float))
    if blur_values.ndim != 1 or np.any(blur_values <= 0):
        raise ValueError("Expected `blur` to be a positive scalar or a 1D sequence of positive values.")

    if g is None:
        weights = np.ones_like(x, dtype=float)
    else:
        weights = np.asarray(g, dtype=float)
        if weights.shape != x.shape:
            raise ValueError("Expected `g` to have the same shape as `x` and `y`.")
        if not np.allclose(weights, 1.0):
            weights = _normalize(weights)

    min_x = float(np.min(x))
    max_x = float(np.max(x))
    min_y = float(np.min(y))
    max_y = float(np.max(y))

    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    half_x = (max_x - min_x) * expand / 2.0
    half_y = (max_y - min_y) * expand / 2.0

    grid_x = np.arange(center_x - half_x, center_x + half_x + dx, dx, dtype=float)
    grid_y = np.arange(center_y - half_y, center_y + half_y + dx, dx, dtype=float)
    if grid_x.size < 2 or grid_y.size < 2:
        raise ValueError("Rasterized grid is too small. Increase the point spread or lower `dx`.")

    mesh_x, mesh_y = np.meshgrid(grid_x, grid_y)
    out = np.zeros((len(blur_values), grid_y.size, grid_x.size), dtype=float)
    radius = int(np.ceil(float(np.max(blur_values)) * 4.0))

    for x_i, y_i, w_i in zip(x, y, weights, strict=False):
        col = int(np.rint((x_i - grid_x[0]) / dx))
        row = int(np.rint((y_i - grid_y[0]) / dx))

        row0 = max(row - radius, 0)
        row1 = min(row + radius, out.shape[1] - 1)
        col0 = max(col - radius, 0)
        col1 = min(col + radius, out.shape[2] - 1)

        patch_x = mesh_x[row0 : row1 + 1, col0 : col1 + 1]
        patch_y = mesh_y[row0 : row1 + 1, col0 : col1 + 1]
        denom = 2.0 * (dx * blur_values * 2.0) ** 2

        kernels = np.exp(-((patch_x[..., None] - x_i) ** 2 + (patch_y[..., None] - y_i) ** 2) / denom)
        kernels_sum = kernels.sum(axis=(0, 1), keepdims=True)
        kernels /= np.where(kernels_sum == 0.0, 1.0, kernels_sum)
        out[:, row0 : row1 + 1, col0 : col1 + 1] += np.moveaxis(kernels * w_i, -1, 0)

    return grid_x, grid_y, out


def affine_from_points(
    points_source: np.ndarray,
    points_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an affine initialization from corresponding landmarks."""
    source = _validate_points(points_source, name="points_source")
    target = _validate_points(points_target, name="points_target")
    if source.shape != target.shape:
        raise ValueError(
            f"Expected `points_source` and `points_target` to have the same shape, found "
            f"`{source.shape}` and `{target.shape}`."
        )

    if source.shape[0] < 3:
        linear = np.eye(2, dtype=float)
        translation = np.mean(target, axis=0) - np.mean(source, axis=0)
        return linear, translation

    source_h = np.concatenate((source, np.ones((source.shape[0], 1), dtype=float)), axis=1)
    target_h = np.concatenate((target, np.ones((target.shape[0], 1), dtype=float)), axis=1)
    affine = np.linalg.lstsq(source_h, target_h, rcond=None)[0].T
    return affine[:2, :2], affine[:2, -1]


BlurScales: TypeAlias = float | tuple[float, ...] | list[float]

__all__ = [
    "STalignConfig",
    "STalignPreprocessConfig",
    "STalignPreprocessResult",
    "STalignRegistrationConfig",
    "STalignResult",
    "stalign_points",
    "stalign_preprocess",
    "transform_points",
]


@dataclass(slots=True)
class STalignPreprocessConfig:
    dx: float = 30.0
    blur: BlurScales = (2.0, 1.0, 0.5)
    expand: float = 1.1


@dataclass(slots=True)
class STalignRegistrationConfig:
    a: float = 500.0
    p: float = 2.0
    expand: float = 2.0
    nt: int = 3
    niter: int = 5000
    diffeo_start: int = 0
    epL: float = 2e-8
    epT: float = 2e-1
    epV: float = 2e3
    sigmaM: float = 1.0
    sigmaB: float = 2.0
    sigmaA: float = 5.0
    sigmaR: float = 5e5
    sigmaP: float = 2e1


@dataclass(slots=True)
class STalignConfig:
    preprocess: STalignPreprocessConfig = field(default_factory=STalignPreprocessConfig)
    registration: STalignRegistrationConfig = field(default_factory=STalignRegistrationConfig)


@dataclass(slots=True)
class STalignPreprocessResult:
    source_grid: tuple[np.ndarray, np.ndarray]
    source_image: np.ndarray
    target_grid: tuple[np.ndarray, np.ndarray]
    target_image: np.ndarray


@dataclass(slots=True)
class STalignResult:
    affine: JaxArray
    velocity: JaxArray
    velocity_grid: tuple[JaxArray, JaxArray]
    aligned_points: JaxArray
    point_order: PointOrder = "row_col"

    def transform_points(
        self,
        points: np.ndarray,
        *,
        direction: Literal["forward", "backward"] = "forward",
        point_order: PointOrder | None = None,
    ) -> JaxArray:
        """Transform arbitrary point arrays with the fitted map."""
        return transform_points(
            self.velocity_grid,
            self.velocity,
            self.affine,
            points,
            direction=direction,
            point_order=self.point_order if point_order is None else point_order,
        )

    def transform_adata(
        self,
        adata: AnnData,
        *,
        spatial_key: str = "spatial",
        key_added: str | None = None,
        direction: Literal["forward", "backward"] = "forward",
        inplace: bool = False,
    ) -> np.ndarray | None:
        """
        Apply the fitted transform to coordinates stored on an AnnData object.

        If ``inplace=False``, return the transformed coordinates without
        modifying ``adata``. If ``inplace=True``, write the transformed
        coordinates to ``adata.obsm[spatial_key]`` or ``adata.obsm[key_added]``
        and return ``None``.
        """
        points = extract_points(adata, key=spatial_key)
        transformed = np.asarray(self.transform_points(points, direction=direction, point_order="xy"))
        if not inplace:
            return transformed

        adata.obsm[spatial_key if key_added is None else key_added] = transformed
        return None


def stalign_preprocess(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    config: STalignPreprocessConfig | None = None,
) -> STalignPreprocessResult:
    """Rasterize source and target point clouds for LDDMM registration."""
    config = STalignPreprocessConfig() if config is None else config
    source_points = to_row_col(source_points, point_order="row_col")
    target_points = to_row_col(target_points, point_order="row_col")

    source_x, source_y, source_image = rasterize(
        source_points[:, 1],
        source_points[:, 0],
        dx=config.dx,
        blur=config.blur,
        expand=config.expand,
    )
    target_x, target_y, target_image = rasterize(
        target_points[:, 1],
        target_points[:, 0],
        dx=config.dx,
        blur=config.blur,
        expand=config.expand,
    )

    return STalignPreprocessResult(
        source_grid=(source_y, source_x),
        source_image=source_image,
        target_grid=(target_y, target_x),
        target_image=target_image,
    )


def transform_points(
    xv: tuple[JaxArray, JaxArray],
    v: JaxArray,
    A: JaxArray,
    points: np.ndarray,
    *,
    direction: Literal["forward", "backward"] = "forward",
    point_order: PointOrder = "row_col",
) -> JaxArray:
    """Transform point arrays with a fitted STalign map."""
    points_rc = to_row_col(points, point_order=point_order)
    transformed = transform_points_row_col(
        xv,
        jnp.asarray(v),
        jnp.asarray(A),
        jnp.asarray(points_rc, dtype=JAX_DTYPE),
        direction=direction,
    )
    return jnp.asarray(from_row_col(np.asarray(transformed), point_order=point_order))


def stalign_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
    *,
    preprocessed: STalignPreprocessResult | None = None,
    config: STalignConfig | None = None,
    landmarks_source: np.ndarray | None = None,
    landmarks_target: np.ndarray | None = None,
) -> STalignResult:
    """Align source point cloud to target with a JAX LDDMM solver."""
    config = STalignConfig() if config is None else config
    registration = config.registration
    source_points = to_row_col(source_points, point_order="row_col")
    target_points = to_row_col(target_points, point_order="row_col")
    if preprocessed is None:
        preprocessed = stalign_preprocess(source_points, target_points, config=config.preprocess)

    if (landmarks_source is None) != (landmarks_target is None):
        raise ValueError("Expected both landmark arrays to be provided together.")

    if landmarks_source is None:
        linear = np.eye(2, dtype=float)
        translation = np.zeros(2, dtype=float)
        source_landmarks = None
        target_landmarks = None
    else:
        source_landmarks = to_row_col(landmarks_source, point_order="row_col")
        target_landmarks = to_row_col(landmarks_target, point_order="row_col")
        linear, translation = affine_from_points(source_landmarks, target_landmarks)

    result = lddmm(
        preprocessed.source_grid,
        preprocessed.source_image,
        preprocessed.target_grid,
        preprocessed.target_image,
        L=jnp.asarray(linear, dtype=JAX_DTYPE),
        T=jnp.asarray(translation, dtype=JAX_DTYPE),
        points_source=None if source_landmarks is None else jnp.asarray(source_landmarks, dtype=JAX_DTYPE),
        points_target=None if target_landmarks is None else jnp.asarray(target_landmarks, dtype=JAX_DTYPE),
        a=registration.a,
        p=registration.p,
        expand=registration.expand,
        nt=registration.nt,
        niter=registration.niter,
        diffeo_start=registration.diffeo_start,
        epL=registration.epL,
        epT=registration.epT,
        epV=registration.epV,
        sigmaM=registration.sigmaM,
        sigmaB=registration.sigmaB,
        sigmaA=registration.sigmaA,
        sigmaR=registration.sigmaR,
        sigmaP=registration.sigmaP,
    )
    aligned_points = transform_points(
        result["xv"],
        result["v"],
        result["A"],
        source_points,
        direction="forward",
        point_order="row_col",
    )
    return STalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=aligned_points,
        point_order="row_col",
    )
