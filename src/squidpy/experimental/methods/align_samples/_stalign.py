"""STalign estimator: JAX LDDMM point-cloud registration.

Holds the estimator adapters :func:`fit_stalign_obs` / :func:`fit_stalign_image`, their
result type :class:`StalignResult`, and the solver-kwargs TypedDicts the public wrappers
in :mod:`squidpy.experimental.tl` are typed against; the pure numerics live under
:mod:`._stalign_impl`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

import numpy.typing as npt

from squidpy.experimental.methods._common import requires

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:  # pragma: no cover - typing only
    JaxArray = Any


class StalignSolverKwargs(TypedDict, total=False):
    """LDDMM solver tuning accepted by the ``align_stalign_*`` functions.

    Every key is optional; defaults live on the estimator signatures
    (:func:`fit_stalign_obs` / :func:`fit_stalign_image`) and differ between the two
    where noted. See those functions for the full parameter semantics.

    - ``initial_affine`` -- homogeneous ``(3, 3)`` affine in ``(x, y)`` to start from.
    - ``initial_velocity``, ``velocity_grid`` -- continuation state from a prior fit,
      in the solver's row-column convention.
    - ``a``, ``p``, ``expand``, ``nt``, ``niter``, ``diffeo_start`` -- LDDMM controls:
      kernel width, regularisation power, velocity-grid padding, integration steps,
      iterations, and the iteration the diffeomorphic part starts updating.
      Defaults obs/image: ``a`` 500/20, ``niter`` 5000/200, ``diffeo_start`` 0/100.
    - ``epL``, ``epT``, ``epV`` -- gradient-descent step sizes for the linear part,
      translation, and velocity field. Default ``epV`` obs/image: 2e3/1.0.
    - ``sigmaM``, ``sigmaB``, ``sigmaA``, ``sigmaR``, ``sigmaP`` -- noise scales for
      the matching, background, artifact, regularisation, and landmark terms.
    - ``muA``, ``muB`` -- optional fixed per-channel artifact/background means;
      ``None`` estimates them during fitting.
    - ``tol``, ``patience`` -- early stopping on relative objective improvement;
      ``tol=None`` (default) always runs ``niter``.
    """

    initial_affine: npt.ArrayLike
    initial_velocity: npt.ArrayLike
    velocity_grid: tuple[npt.ArrayLike, npt.ArrayLike]
    a: float
    p: float
    expand: float
    nt: int
    niter: int
    diffeo_start: int
    epL: float
    epT: float
    epV: float
    sigmaM: float
    sigmaB: float
    sigmaA: float
    sigmaR: float
    sigmaP: float
    muA: npt.ArrayLike | None
    muB: npt.ArrayLike | None
    tol: float | None
    patience: int


class StalignObsSolverKwargs(StalignSolverKwargs, total=False):
    """:class:`StalignSolverKwargs` plus the point-cloud rasterization knobs.

    - ``dx`` -- grid spacing of the density rasters (default 30).
    - ``blur`` -- Gaussian blur scale(s) applied to the rasters (default (2, 1, 0.5)).
    - ``raster_expand`` -- field-of-view padding factor (default 1.1).
    """

    dx: float
    blur: float | Sequence[float]
    raster_expand: float


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
    match_weights: JaxArray | None = None
    artifact_weights: JaxArray | None = None
    background_weights: JaxArray | None = None
    energies: JaxArray | None = None
    n_iter: int | None = None

    def deformation_grid(
        self,
        *,
        direction: Literal["forward", "backward"] = "forward",
        query_axes: tuple[JaxArray, JaxArray] | None = None,
        ref_axes: tuple[JaxArray, JaxArray] | None = None,
    ) -> JaxArray:
        """Return a dense row-column coordinate transform for visualisation.

        ``direction="forward"`` evaluates the query grid in the reference frame;
        ``"backward"`` evaluates the reference grid in the query frame. The returned
        array has shape ``(2, rows, columns)``.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import _grid_points, _transform_grid_backward, transform_points_row_col

        source_axes = query_axes if query_axes is not None else self.query_axes
        target_axes = ref_axes if ref_axes is not None else self.ref_axes
        if source_axes is None or target_axes is None:
            raise ValueError(
                "This result was fitted on point clouds and carries no raster axes. "
                "Pass both `query_axes=` and `ref_axes=`, or fit with "
                "`align_stalign_image`."
            )
        if direction == "backward":
            return _transform_grid_backward(target_axes, self.velocity_grid, self.velocity, self.affine)
        if direction != "forward":
            raise ValueError(f"Expected `direction` to be 'forward' or 'backward', found {direction!r}.")

        grid = _grid_points(source_axes)
        points = jnp.moveaxis(grid, 0, -1).reshape((-1, 2))
        transformed = transform_points_row_col(
            self.velocity_grid, self.velocity, self.affine, points, direction="forward"
        )
        return jnp.moveaxis(transformed.reshape((*grid.shape[1:], 2)), -1, 0)

    def warp_image(
        self,
        image: JaxArray,
        *,
        direction: Literal["forward", "backward"] = "forward",
        query_axes: tuple[JaxArray, JaxArray] | None = None,
        ref_axes: tuple[JaxArray, JaxArray] | None = None,
    ) -> JaxArray:
        """Resample an image through the fitted transformation.

        A diffeomorphism cannot be expressed as a SpatialData transformation -- the
        available types are affine at most -- so an aligned image has to be materialised
        rather than registered. ``direction="forward"`` maps a query-frame image onto
        the reference grid; ``"backward"`` maps a reference-frame image onto the query
        grid. Explicit axes allow results fitted from point clouds to warp their density
        rasters without pretending those rasters are original image elements.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import _interp, jax_dtype

        arr = jnp.asarray(image, dtype=jax_dtype())
        if arr.ndim == 2:
            arr = arr[None]
        if direction not in {"forward", "backward"}:
            raise ValueError(f"Expected `direction` to be 'forward' or 'backward', found {direction!r}.")
        source_axes = query_axes if query_axes is not None else self.query_axes
        target_axes = ref_axes if ref_axes is not None else self.ref_axes
        grid = self.deformation_grid(
            direction="backward" if direction == "forward" else "forward",
            query_axes=source_axes,
            ref_axes=target_axes,
        )
        sampling_axes = source_axes if direction == "forward" else target_axes
        if sampling_axes is None:  # guarded by deformation_grid; keeps the type checker honest
            raise AssertionError("missing sampling axes")
        return _interp(sampling_axes, arr, grid)

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


@requires("jax")
def fit_stalign_obs(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    landmarks_ref: npt.ArrayLike | None = None,
    landmarks_query: npt.ArrayLike | None = None,
    initial_affine: npt.ArrayLike | None = None,
    initial_velocity: npt.ArrayLike | None = None,
    velocity_grid: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
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
    muA: npt.ArrayLike | None = None,
    muB: npt.ArrayLike | None = None,
    tol: float | None = None,
    patience: int = 25,
) -> StalignResult:
    """Fit a deformation mapping ``query`` onto ``ref``.

    Parameters
    ----------
    ref, query
        ``(N, 2)`` / ``(M, 2)`` reference and query point clouds in ``(x, y)``
        order; the query is aligned onto the reference. Both are plain in-memory
        arrays -- extracting them from an ``AnnData`` / ``SpatialData`` is the
        caller's responsibility.
    landmarks_ref, landmarks_query
        Optional corresponding ``(x, y)`` landmark arrays used to initialise the
        affine, matched by row order. Must be provided together.
    initial_affine
        Optional homogeneous ``(3, 3)`` affine in public ``(x, y)`` coordinates.
        Mutually exclusive with landmark initialisation.
    initial_velocity, velocity_grid
        Optional continuation state. The velocity has shape ``(nt, rows, columns, 2)``;
        its components and the two grid axes use the solver's row-column convention.
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
    muA, muB
        Optional per-channel artifact and background means. ``None`` estimates the
        corresponding mean during fitting, matching upstream STalign's default.
    tol, patience
        Stop once the objective's relative improvement over the last ``patience``
        iterations falls below ``tol``. ``tol=None`` (default) always runs ``niter``.

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
    # Import the JAX-backed solver only after `requires` has checked the dependency,
    # so callers without JAX get a clean ImportError rather than a confusing failure
    # from a module-level `import jax`.
    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype, lddmm, transform_points_row_col
    from ._stalign_impl._helpers import affine_from_points, rasterize_cloud, validate_points

    if (landmarks_ref is None) != (landmarks_query is None):
        raise ValueError("Expected both landmark arrays to be provided together.")
    if initial_affine is not None and landmarks_ref is not None:
        raise ValueError("`initial_affine` is mutually exclusive with landmark initialisation.")

    # The solver runs internally in row-col (y, x); inputs are (x, y) -- swap at the boundary.
    source_rc = validate_points(query, name="query")[:, ::-1]
    target_rc = validate_points(ref, name="ref")[:, ::-1]
    source_grid, source_image = rasterize_cloud(source_rc, dx=dx, blur=blur, expand=raster_expand)
    target_grid, target_image = rasterize_cloud(target_rc, dx=dx, blur=blur, expand=raster_expand)

    dtype = jax_dtype()
    if initial_affine is not None:
        affine_xy = jnp.asarray(initial_affine, dtype=dtype)
        if affine_xy.shape != (3, 3):
            raise ValueError(f"Expected `initial_affine` to have shape (3, 3), found {affine_xy.shape}.")
        swap = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
        affine_rc = swap @ affine_xy @ swap
        linear, translation = affine_rc[:2, :2], affine_rc[:2, 2]
        src_lm = tgt_lm = None
    elif landmarks_ref is None:
        linear, translation = jnp.eye(2, dtype=dtype), jnp.zeros(2, dtype=dtype)
        src_lm = tgt_lm = None
    else:
        src_lm = validate_points(landmarks_query, name="landmarks_query")[:, ::-1]
        tgt_lm = validate_points(landmarks_ref, name="landmarks_ref")[:, ::-1]
        linear_np, translation_np = affine_from_points(src_lm, tgt_lm)
        linear, translation = jnp.asarray(linear_np, dtype=dtype), jnp.asarray(translation_np, dtype=dtype)

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        initial_velocity=initial_velocity,
        velocity_grid=velocity_grid,
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
        muA=muA,
        muB=muB,
        tol=tol,
        patience=patience,
    )
    aligned_rc = transform_points_row_col(result["xv"], result["v"], result["A"], source_rc, direction="forward")
    return StalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=aligned_rc[:, ::-1],
        match_weights=result["WM"],
        artifact_weights=result["WA"],
        background_weights=result["WB"],
        energies=result["energies"],
        n_iter=int(result["n_iter"]),
        # No raster axes: the grids here are the internal density rasters at `dx`
        # resolution, not a frame any real image lives on. Offering `warp_image` off them
        # would quietly resample the caller's image onto a coarse, unrelated grid.
    )


@requires("jax")
def fit_stalign_image(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    ref_scale: tuple[float, float] = (1.0, 1.0),
    query_scale: tuple[float, float] = (1.0, 1.0),
    ref_axes: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    query_axes: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    initial_affine: npt.ArrayLike | None = None,
    initial_velocity: npt.ArrayLike | None = None,
    velocity_grid: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
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
    muA: npt.ArrayLike | None = None,
    muB: npt.ArrayLike | None = None,
    tol: float | None = None,
    patience: int = 25,
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
    ref_axes, query_axes
        Optional explicit physical row and column axes. Both pairs must be supplied;
        they are mutually exclusive with non-unit ``ref_scale``/``query_scale``.
    initial_affine
        Optional homogeneous ``(3, 3)`` affine in public ``(x, y)`` coordinates.
    initial_velocity, velocity_grid
        Optional continuation state in the solver's row-column convention.
    a, p, expand, nt, niter, diffeo_start
        LDDMM controls, as in :func:`fit_stalign_obs`. Note ``a`` is a length in the *same*
        units as ``ref_scale`` -- the default of 20 suits pixel units, where
        :func:`fit_stalign_obs`'s 500 would exceed most images. ``diffeo_start`` defaults to
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
    muA, muB
        Optional per-channel artifact and background means. ``None`` estimates the
        corresponding mean during fitting, matching upstream STalign's default.
    tol, patience
        Stop once the objective's relative improvement over the last ``patience``
        iterations falls below ``tol``. ``tol=None`` (default) always runs ``niter``.

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

    if (query_axes is None) != (ref_axes is None):
        raise ValueError("Expected both `query_axes` and `ref_axes` to be provided together.")

    def explicit_axes(value: tuple[npt.ArrayLike, npt.ArrayLike], image: JaxArray, name: str):
        resolved = (jnp.asarray(value[0], dtype=dtype), jnp.asarray(value[1], dtype=dtype))
        expected = image.shape[1:]
        if resolved[0].ndim != 1 or resolved[1].ndim != 1 or tuple(map(len, resolved)) != expected:
            raise ValueError(f"Expected `{name}` lengths {expected}, found {tuple(map(len, resolved))}.")
        if len(resolved[0]) < 2 or len(resolved[1]) < 2:
            raise ValueError(f"Expected each `{name}` axis to contain at least two coordinates.")
        return resolved

    if query_axes is None:
        source_grid = axes(source_image, query_scale)
        target_grid = axes(target_image, ref_scale)
    else:
        if query_scale != (1.0, 1.0) or ref_scale != (1.0, 1.0):
            raise ValueError("Explicit axes are mutually exclusive with non-unit image scales.")
        source_grid = explicit_axes(query_axes, source_image, "query_axes")
        target_grid = explicit_axes(ref_axes, target_image, "ref_axes")

    if initial_affine is None:
        linear, translation = jnp.eye(2, dtype=dtype), jnp.zeros(2, dtype=dtype)
    else:
        affine_xy = jnp.asarray(initial_affine, dtype=dtype)
        if affine_xy.shape != (3, 3):
            raise ValueError(f"Expected `initial_affine` to have shape (3, 3), found {affine_xy.shape}.")
        swap = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=dtype)
        affine_rc = swap @ affine_xy @ swap
        linear, translation = affine_rc[:2, :2], affine_rc[:2, 2]

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        initial_velocity=initial_velocity,
        velocity_grid=velocity_grid,
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
        muA=muA,
        muB=muB,
        tol=tol,
        patience=patience,
    )
    return StalignResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        aligned_points=jnp.zeros((0, 2), dtype=dtype),
        query_axes=source_grid,
        ref_axes=target_grid,
        match_weights=result["WM"],
        artifact_weights=result["WA"],
        background_weights=result["WB"],
        energies=result["energies"],
        n_iter=int(result["n_iter"]),
    )
