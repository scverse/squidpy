"""STalign estimator: JAX LDDMM point-cloud registration.

Holds both the estimator adapter :func:`fit_stalign` and its result type
:class:`StalignResult`; the pure numerics live under :mod:`._stalign_impl`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy.typing as npt

from squidpy.experimental.methods.registry import ALIGN_IMAGES, ALIGN_SAMPLES

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any


@dataclass(slots=True)
class StalignResult:
    """A fitted STalign diffeomorphism, ready to transform arbitrary points.

    :meth:`transform` works in ``(x, y)``; ``aligned_points`` is the fitted query
    cloud already mapped into the reference frame.
    """

    affine: JaxArray
    velocity: JaxArray
    velocity_grid: tuple[JaxArray, JaxArray]
    aligned_points: JaxArray
    #: Row-col axes of the query and reference rasters the fit ran on, when it ran on
    #: images. ``None`` for point-cloud fits, where no raster survives the call.
    query_axes: tuple[JaxArray, JaxArray] | None = None
    ref_axes: tuple[JaxArray, JaxArray] | None = None

    def warp_image(self, image: JaxArray) -> JaxArray:
        """Resample a query-frame ``(c, y, x)`` image onto the reference grid.

        A diffeomorphism cannot be expressed as a SpatialData transformation -- the
        available types are affine at most -- so an aligned image has to be materialised
        rather than registered.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import _interp, _transform_grid_backward, jax_dtype

        if self.query_axes is None or self.ref_axes is None:
            raise ValueError(
                "This result was fitted on point clouds, so it carries no raster axes to "
                "warp an image with. Fit with `align(in_='images/<name>', ...)` instead."
            )
        arr = jnp.asarray(image, dtype=jax_dtype())
        if arr.ndim == 2:
            arr = arr[None]
        grid = _transform_grid_backward(self.ref_axes, self.velocity_grid, self.velocity, self.affine)
        return _interp(self.query_axes, arr, grid)

    def transform(
        self,
        points: JaxArray,
        *,
        direction: Literal["forward", "backward"] = "forward",
    ) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` points with the fitted diffeomorphism."""
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_points_row_col

        pts = jnp.asarray(points, dtype=jax_dtype())
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) `(x, y)` array, found shape {pts.shape}.")
        transformed_rc = transform_points_row_col(
            self.velocity_grid,
            self.velocity,
            self.affine,
            pts[:, ::-1],
            direction=direction,
        )
        return transformed_rc[:, ::-1]


@ALIGN_SAMPLES.register("stalign", requires=("jax",))
def fit_stalign(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    landmarks_source: npt.ArrayLike | None = None,
    landmarks_target: npt.ArrayLike | None = None,
    # rasterization
    dx: float = 30.0,
    blur: float | Sequence[float] = (2.0, 1.0, 0.5),
    raster_expand: float = 1.1,
    # LDDMM registration
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
) -> StalignResult:
    """Fit a deformation mapping ``query`` onto ``ref``.

    Parameters
    ----------
    ref, query
        ``(N, 2)`` / ``(M, 2)`` reference and query point clouds in ``(x, y)``
        order; the query is aligned onto the reference. Both are plain in-memory
        arrays -- extracting them from an ``AnnData`` / ``SpatialData`` is the
        caller's responsibility.
    landmarks_source, landmarks_target
        Optional corresponding ``(x, y)`` landmark arrays used to initialise the
        affine. Must be provided together.
    dx, blur, raster_expand
        Rasterization of the point clouds into density images: grid spacing,
        Gaussian blur scale(s), and field-of-view padding factor.
    a, p, expand, nt, niter, diffeo_start
        LDDMM controls: kernel width ``a``, regularisation power ``p``,
        velocity-grid padding ``expand``, number of integration time steps
        ``nt``, iterations ``niter``, and the iteration at which the
        diffeomorphic (non-affine) part starts updating ``diffeo_start``.
    epL, epT, epV
        Gradient-descent step sizes for the linear part, translation, and
        velocity field.
    sigmaM, sigmaB, sigmaA, sigmaR, sigmaP
        Noise scales for the matching, background, artifact, regularisation, and
        landmark-point terms of the objective.

    Returns
    -------
    A :class:`StalignResult` whose :meth:`~StalignResult.transform` maps
    ``(x, y)`` points into the reference frame; ``aligned_points`` is the fitted
    ``query`` already mapped.

    Notes
    -----
    Runs in JAX's active float precision, which is **single** unless x64 is enabled.
    The original STalign is double throughout, so results differ correspondingly. For
    ``niter`` in the thousands, or a large ``sigmaR``, enable double precision before
    importing JAX::

        import jax

        jax.config.update("jax_enable_x64", True)
    """
    # Import the JAX-backed solver only after the registry's requirements check
    # passes, so callers without JAX get a clean ImportError rather than a
    # confusing failure from a module-level `import jax`.
    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype, lddmm, transform_points_row_col
    from ._stalign_impl._helpers import affine_from_points, rasterize_cloud, validate_points

    if (landmarks_source is None) != (landmarks_target is None):
        raise ValueError("Expected both landmark arrays to be provided together.")

    # The solver runs internally in row-col (y, x); inputs are (x, y) -- swap at the boundary.
    source_rc = validate_points(query, name="query")[:, ::-1]
    target_rc = validate_points(ref, name="ref")[:, ::-1]
    source_grid, source_image = rasterize_cloud(source_rc, dx=dx, blur=blur, expand=raster_expand)
    target_grid, target_image = rasterize_cloud(target_rc, dx=dx, blur=blur, expand=raster_expand)

    dtype = jax_dtype()
    if landmarks_source is None:
        linear, translation = jnp.eye(2, dtype=dtype), jnp.zeros(2, dtype=dtype)
        src_lm = tgt_lm = None
    else:
        src_lm = validate_points(landmarks_source, name="landmarks_source")[:, ::-1]
        tgt_lm = validate_points(landmarks_target, name="landmarks_target")[:, ::-1]
        linear_np, translation_np = affine_from_points(src_lm, tgt_lm)
        linear, translation = jnp.asarray(linear_np, dtype=dtype), jnp.asarray(translation_np, dtype=dtype)

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        points_source=src_lm,
        points_target=tgt_lm,
        a=a,
        p=p,
        expand=expand,
        nt=nt,
        niter=niter,
        diffeo_start=diffeo_start,
        epL=epL,
        epT=epT,
        epV=epV,
        sigmaM=sigmaM,
        sigmaB=sigmaB,
        sigmaA=sigmaA,
        sigmaR=sigmaR,
        sigmaP=sigmaP,
    )
    aligned_rc = transform_points_row_col(result["xv"], result["v"], result["A"], source_rc, direction="forward")
    return StalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=aligned_rc[:, ::-1],
        # No raster axes: the grids here are the internal density rasters at `dx`
        # resolution, not a frame any real image lives on. Offering `warp_image` off them
        # would quietly resample the caller's image onto a coarse, unrelated grid.
    )


@ALIGN_IMAGES.register("stalign", requires=("jax",))
def fit_stalign_image(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    ref_scale: tuple[float, float] = (1.0, 1.0),
    query_scale: tuple[float, float] = (1.0, 1.0),
    # LDDMM registration
    a: float = 20.0,
    p: float = 2.0,
    expand: float = 2.0,
    nt: int = 3,
    niter: int = 200,
    diffeo_start: int = 100,
    epL: float = 2e-8,
    epT: float = 2e-1,
    epV: float = 1.0,
    sigmaM: float = 1.0,
    sigmaB: float = 2.0,
    sigmaA: float = 5.0,
    sigmaR: float = 5e5,
    sigmaP: float = 2e1,
) -> StalignResult:
    """Fit a deformation mapping the ``query`` image onto the ``ref`` image.

    Parameters
    ----------
    ref, query
        Channels-first ``(c, y, x)`` rasters (a bare ``(y, x)`` array is promoted). The
        query is aligned onto the reference; they need not share a shape.
    ref_scale, query_scale
        Physical size of one pixel as ``(y, x)``. Defaults to pixel units. Pass the
        element's scale when the two images have different resolutions, otherwise the
        fit is done in mismatched coordinates.
    a, p, expand, nt, niter, diffeo_start
        LDDMM controls, as in :func:`fit_stalign`. Note ``a`` is a length in the *same*
        units as ``ref_scale`` -- the default of 20 suits pixel units, where
        :func:`fit_stalign`'s 500 would exceed most images. ``diffeo_start`` defaults to
        half of ``niter`` so the affine settles before the deformable part switches on;
        starting both at once lets the velocity field absorb what is really a
        translation, and it fits it worse than the affine would have.
    epL, epT, epV
        Gradient-descent step sizes for the linear part, translation, and velocity field.
        These are **scale dependent**: they are tuned here for images in pixel units, so
        a non-unit ``ref_scale`` will need them rescaled to match. ``epV`` is the one to
        reach for first -- too large and the deformation overwhelms the affine.
    sigmaM, sigmaB, sigmaA, sigmaR, sigmaP
        Noise scales for the matching, background, artifact, regularisation, and
        landmark-point terms of the objective.

    Returns
    -------
    A :class:`StalignResult`. Its :meth:`~StalignResult.transform` maps ``(x, y)`` points
    in query pixel coordinates into the reference frame, and
    :meth:`~StalignResult.warp_image` resamples a query image onto the reference grid.
    """
    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype, lddmm

    dtype = jax_dtype()

    def as_chw(image: npt.ArrayLike, name: str) -> JaxArray:
        # Not `jnp.atleast_3d`: it appends the new axis, turning a (y, x) image into
        # (y, x, 1) -- y channels of x by 1 -- instead of a single (1, y, x) channel.
        arr = jnp.asarray(image, dtype=dtype)
        if arr.ndim == 2:
            return arr[None]
        if arr.ndim != 3:
            raise ValueError(f"Expected `{name}` to be a `(y, x)` or `(c, y, x)` image, found shape {arr.shape}.")
        return arr

    source_image = as_chw(query, "query")
    target_image = as_chw(ref, "ref")
    if source_image.shape[0] != target_image.shape[0]:
        raise ValueError(
            f"Expected `ref` and `query` to have the same number of channels, found "
            f"{target_image.shape[0]} and {source_image.shape[0]}."
        )

    def axes(image: JaxArray, scale: tuple[float, float]) -> tuple[JaxArray, JaxArray]:
        # Row-col physical coordinates, centred so the affine initialises near identity.
        rows, cols = image.shape[1], image.shape[2]
        return (
            (jnp.arange(rows, dtype=dtype) - (rows - 1) / 2.0) * scale[0],
            (jnp.arange(cols, dtype=dtype) - (cols - 1) / 2.0) * scale[1],
        )

    source_grid = axes(source_image, query_scale)
    target_grid = axes(target_image, ref_scale)

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=jnp.eye(2, dtype=dtype),
        T=jnp.zeros(2, dtype=dtype),
        a=a,
        p=p,
        expand=expand,
        nt=nt,
        niter=niter,
        diffeo_start=diffeo_start,
        epL=epL,
        epT=epT,
        epV=epV,
        sigmaM=sigmaM,
        sigmaB=sigmaB,
        sigmaA=sigmaA,
        sigmaR=sigmaR,
        sigmaP=sigmaP,
    )
    return StalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=jnp.zeros((0, 2), dtype=dtype),
        query_axes=source_grid,
        ref_axes=target_grid,
    )
