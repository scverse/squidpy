"""STalign estimator: JAX LDDMM registration, at rank 2 and rank 3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Unpack

import numpy.typing as npt

if TYPE_CHECKING:
    import jax

    JaxArray = jax.Array
else:
    # Bound at runtime, not just under TYPE_CHECKING: `sphinx_autodoc_typehints` calls
    # `get_type_hints` on the result dataclasses, and jax is an optional extra that must
    # not be imported here. `Any` is the placeholder that keeps that resolvable.
    JaxArray = Any


class StalignImageSolverKwargs(TypedDict, total=False):
    """The LDDMM controls :func:`~squidpy.experimental.tl.align_stalign_image` takes.

    Also the shared set the other two entry points vary from:
    :class:`~squidpy.experimental.tl.StalignObsSolverKwargs` extends it with rasterization
    knobs, and :class:`~squidpy.experimental.tl.StalignVolumeSolverKwargs` drops the two
    keys that mean nothing at rank 3. The image path neither adds nor removes anything,
    which is why this set carries its name.

    Every key is optional. Defaults differ between the point-cloud and image solvers
    where noted below, since a kernel width in cell coordinates is not one in pixels.

    - ``initial_affine`` -- homogeneous ``(3, 3)`` affine in ``(x, y)`` to start from.
    - ``initial_velocity``, ``velocity_grid`` -- continuation state from a prior fit,
      in the solver's row-column convention.
    - ``a``, ``p``, ``expand``, ``nt``, ``niter``, ``diffeo_start`` -- LDDMM controls:
      kernel width, regularisation power, velocity-grid padding, integration steps,
      iterations, and the iteration the diffeomorphic part starts updating.
      Defaults obs/image: ``a`` 500/20, ``niter`` 5000/200, ``diffeo_start`` 0/100.
    - ``epL``, ``epT``, ``epV`` -- gradient-descent step sizes for the linear part,
      translation, and velocity field. Default ``epV`` obs/image: 2e3/1.0. All three are
      **scale dependent**, so coordinates in other units need them rescaled; ``epV`` is
      the one to reach for first -- too large and the deformation overwhelms the affine.
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


class StalignVolumeSolverKwargs(TypedDict, total=False):
    """LDDMM solver tuning accepted by :func:`~squidpy.experimental.tl.align_stalign_volume`.

    :class:`~squidpy.experimental.tl.StalignImageSolverKwargs` minus two keys: ``sigmaP``,
    because the rank-3 path has no point-matching energy and it would be a knob that does
    nothing, and ``initial_affine``, which is a named ``(4, 4)`` argument on
    :func:`~squidpy.experimental.tl.align_stalign_volume` rather than a solver key. Five
    defaults differ from the 2D case -- ``expand`` 1.25, ``epL`` 1e-6, ``epT`` 1e1,
    ``epV`` 1e3 and ``sigmaR`` 1e6. That last one is the only one that is not upstream's:
    ``LDDMM_3D_to_slice`` declares ``sigmaR=1e8``, which weights the regulariser so weakly that
    the velocity field grows unchecked and the reported objective climbs after its minimum.

    - ``initial_velocity``, ``velocity_grid`` -- continuation state from a prior fit, in
      the solver's ``(z, y, x)`` convention.
    - ``a``, ``p``, ``expand``, ``nt``, ``niter``, ``diffeo_start`` -- LDDMM controls:
      kernel width, regularisation power, velocity-grid padding, integration steps,
      iterations, and the iteration the diffeomorphic part starts updating.
    - ``epL``, ``epT``, ``epV`` -- gradient-descent step sizes for the linear part,
      translation, and velocity field.
    - ``sigmaM``, ``sigmaB``, ``sigmaA``, ``sigmaR`` -- noise scales for the matching,
      background, artifact and regularisation terms.
    - ``muA``, ``muB`` -- optional fixed per-channel artifact/background means, one entry
      per *section* channel; ``None`` estimates them during fitting.
    - ``tol``, ``patience`` -- early stopping on relative objective improvement;
      ``tol=None`` (default) always runs ``niter``.
    """

    initial_velocity: npt.ArrayLike
    velocity_grid: tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]
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
    muA: npt.ArrayLike | None
    muB: npt.ArrayLike | None
    tol: float | None
    patience: int


class StalignObsSolverKwargs(StalignImageSolverKwargs, total=False):
    """:class:`~squidpy.experimental.tl.StalignImageSolverKwargs` plus the rasterization knobs.

    The point-cloud path rasterizes both clouds into density images and then runs the same
    solver, so it accepts everything the image path does and three keys more.

    - ``dx`` -- grid spacing of the density rasters (default 30).
    - ``blur`` -- Gaussian blur scale(s) applied to the rasters (default (2, 1, 0.5)).
    - ``raster_expand`` -- field-of-view padding factor (default 1.1).
    """

    dx: float
    blur: float | Sequence[float]
    raster_expand: float


#: Shared LDDMM defaults. Every key here is forwarded to :func:`._stalign_impl._core.lddmm`,
#: which keeps the parameter names but carries no defaults of its own -- so these values
#: exist in exactly one place. Annotating the dict with the TypedDict makes the type
#: checker verify every value against its key.
_SOLVER_DEFAULTS: StalignImageSolverKwargs = {
    "a": 500.0,
    "p": 2.0,
    "expand": 2.0,
    "nt": 3,
    "niter": 5000,
    "diffeo_start": 0,
    "epL": 2e-8,
    "epT": 2e-1,
    "epV": 2e3,
    "sigmaM": 1.0,
    "sigmaB": 2.0,
    "sigmaA": 5.0,
    "sigmaR": 5e5,
    "sigmaP": 2e1,
    "muA": None,
    "muB": None,
    "tol": None,
    "patience": 25,
}

#: Point-cloud case: the shared solver defaults plus the rasterization knobs.
_OBS_DEFAULTS: StalignObsSolverKwargs = {
    **_SOLVER_DEFAULTS,
    "dx": 30.0,
    "blur": (2.0, 1.0, 0.5),
    "raster_expand": 1.1,
}

#: Image case: a kernel width of 500 would exceed most images, and starting the
#: diffeomorphic part halfway lets the affine settle before the velocity field can
#: absorb what is really a translation and fit it worse than the affine would have.
_IMAGE_DEFAULTS: StalignImageSolverKwargs = {
    **_SOLVER_DEFAULTS,
    "a": 20.0,
    "niter": 200,
    "diffeo_start": 100,
    "epV": 1.0,
}

#: Slice case: upstream's ``LDDMM_3D_to_slice`` defaults, which differ from the 2D
#: ``LDDMM``'s in five places. ``sigmaP`` is carried only because ``lddmm`` requires it;
#: with no landmarks the point term is identically zero, and the public
#: :class:`StalignVolumeSolverKwargs` deliberately does not expose it.
_VOLUME_DEFAULTS: StalignImageSolverKwargs = {
    **_SOLVER_DEFAULTS,
    "expand": 1.25,
    "epL": 1e-6,
    "epT": 1e1,
    "epV": 1e3,
    # Upstream's rank-3 default is 1e8. Against *this* regularisation energy -- the coherent
    # one, transforming every spatial axis -- 1e8 leaves the term at ~1e-5 of the objective on
    # a real volume-to-section fit, i.e. switched off, and the fit runs unregularised: rms |v|
    # 20.7 against 9.4 for upstream, and thin cortical laminae (VISam1-6a, AUDd6a, PERI5) lose
    # their last cell and vanish from the annotation. 1e8 is the right number for upstream's
    # line, which inflates the same field 3460-8880x by dropping one axis from the transform
    # and one size from the Parseval normalisation; it is the wrong number for a term without
    # that inflation. 1e6 restores the weight upstream gets by accident.
    "sigmaR": 1e6,
}

#: Keys the fit functions consume themselves rather than forwarding to the solver:
#: the rasterization knobs, and the affine that becomes the solver's `L`/`T`.
_CONSUMED_KEYS = frozenset({"dx", "blur", "raster_expand", "initial_affine"})

_JAX_REQUIRED = 'STalign alignment requires JAX: `pip install "squidpy[jax]"`.'


@dataclass(slots=True)
class Stalign2DResult:
    """A fitted STalign diffeomorphism whose reference frame is a plane.

    The rank-2 counterpart to :class:`Stalign3DResult`: both carry the same fitted
    fields, and the reference frame's dimensionality is what separates them --
    :meth:`transform` returns ``(N, 2)`` here and ``(N, 3)`` there.

    :meth:`transform` works in ``(x, y)``, and unlike
    :class:`Stalign3DResult`'s it takes a ``direction``: at rank 2 both images are
    flat, so mapping the reference back into the query frame is equally meaningful.
    """

    affine: JaxArray
    velocity: JaxArray
    velocity_grid: tuple[JaxArray, JaxArray]
    #: The fitted query cloud already mapped into the reference frame. ``None`` for image
    #: fits, where there is no cloud -- an empty array would read as "zero points aligned".
    aligned_points: JaxArray | None = None
    #: Row-col axes of the query and reference rasters the fit ran on, when it ran on
    #: images. ``None`` for point-cloud fits, where no raster survives the call.
    query_axes: tuple[JaxArray, JaxArray] | None = None
    ref_axes: tuple[JaxArray, JaxArray] | None = None
    match_weights: JaxArray | None = None
    artifact_weights: JaxArray | None = None
    background_weights: JaxArray | None = None
    energies: JaxArray | None = None
    n_iter: int | None = None

    @property
    def affine_xyz(self) -> JaxArray:
        """The affine part as a ``(3, 3)`` in public ``(x, y)`` order.

        :attr:`affine` is in the solver's row-column order; this is the same matrix in the
        order the public API speaks, registerable as a SpatialData
        :class:`~spatialdata.transformations.Affine`. The velocity field is not expressible
        that way, so this is the coarse part of the fit only.
        """
        from ._stalign_impl._core import reverse_axes

        swap = reverse_axes(2)
        return swap @ self.affine @ swap

    def deformation_grid(
        self,
        *,
        direction: Literal["forward", "backward"] = "forward",
        query_axes: tuple[JaxArray, JaxArray] | None = None,
        ref_axes: tuple[JaxArray, JaxArray] | None = None,
    ) -> JaxArray:
        """The dense row-column coordinate transform, shape ``(2, rows, columns)``.

        ``direction="backward"`` evaluates the fixed image's grid in the moving image's
        frame -- at rank 2 that is the reference grid in the query frame.
        """
        from ._stalign_impl._core import transform_grid_row_col

        if direction not in {"forward", "backward"}:
            raise ValueError(f"Expected `direction` to be 'forward' or 'backward', found {direction!r}.")
        source_axes = query_axes if query_axes is not None else self.query_axes
        target_axes = ref_axes if ref_axes is not None else self.ref_axes
        if source_axes is None or target_axes is None:
            raise ValueError(
                "This result was fitted on point clouds and carries no raster axes. "
                "Pass both `query_axes=` and `ref_axes=`, or fit with "
                "`align_stalign_image`."
            )
        # Forward evaluates the query grid in the reference frame; backward the reverse.
        axes = source_axes if direction == "forward" else target_axes
        return transform_grid_row_col(axes, self.velocity_grid, self.velocity, self.affine, direction=direction)

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
        from ._stalign_impl._core import interp
        from ._stalign_impl._helpers import as_chw

        arr = as_chw(image, name="image")
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
        return interp(sampling_axes, arr, grid)

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


@dataclass(slots=True)
class Stalign3DResult:
    """A fitted STalign registration whose reference frame is a volume.

    The rank-3 counterpart to :class:`Stalign2DResult`, placing a flat section's cells
    in a 3D reference.

    :meth:`transform` takes ``(x, y)`` section coordinates to ``(x, y, z)`` reference
    coordinates. Unlike :class:`Stalign2DResult`'s it has no ``direction``: the section is
    the fixed image and it is flat, so only section-into-volume is meaningful. Pair it with
    :func:`~squidpy.experimental.im.sample_volume` to read a reference volume at the
    mapped points.
    """

    #: Homogeneous ``(4, 4)`` affine in the solver's ``(z, y, x)`` array order.
    affine: JaxArray
    velocity: JaxArray
    velocity_grid: tuple[JaxArray, JaxArray, JaxArray]
    #: Physical ``(z, y, x)`` axes of the reference volume the fit ran on.
    ref_axes: tuple[JaxArray, JaxArray, JaxArray]
    #: Physical ``(y, x)`` axes of the section the fit ran on.
    query_axes: tuple[JaxArray, JaxArray]
    match_weights: JaxArray | None = None
    artifact_weights: JaxArray | None = None
    background_weights: JaxArray | None = None
    energies: JaxArray | None = None
    n_iter: int | None = None

    @property
    def affine_xyz(self) -> JaxArray:
        """The affine part as a ``(4, 4)`` in public ``(x, y, z)`` order.

        Registerable as a SpatialData :class:`~spatialdata.transformations.Affine` with
        differing input and output axes. The velocity field is not expressible that way --
        SpatialData transformations top out at affine -- so this is the coarse part of the
        fit only, and :meth:`transform` remains the faithful map.
        """
        from ._stalign_impl._core import reverse_axes

        swap = reverse_axes(3)
        return swap @ self.affine @ swap

    def deformation_grid(
        self,
        *,
        direction: Literal["forward", "backward"] = "backward",
        ref_axes: Sequence[npt.ArrayLike] | None = None,
        query_axes: Sequence[npt.ArrayLike] | None = None,
    ) -> JaxArray:
        """The dense ``(z, y, x)`` coordinate transform, shape ``(3, *grid)``.

        Same contract as :meth:`Stalign2DResult.deformation_grid`: ``"backward"`` evaluates
        the fixed image's grid in the moving image's frame, and it is the *same* call on
        the *same* fitted ``affine``/``velocity``/``velocity_grid`` that the objective
        samples through -- not an approximation for plotting. Given the same axes it agrees
        with the internal transform exactly.

        Which element is which flips at rank 3, though: the **volume** is the moving image,
        warped onto the fixed section. So ``"backward"`` (the default here, and the
        direction :meth:`transform` uses) evaluates the *section's* grid in volume
        coordinates, shape ``(3, 1, rows, columns)`` -- the section is lifted onto the
        ``z = 0`` plane, hence the length-1 ``z``. ``"forward"`` evaluates the volume's own
        grid in the section frame, shape ``(3, *volume_shape)``.

        Parameters
        ----------
        direction
            ``"backward"`` (default) for the section-into-volume map, ``"forward"`` for its
            inverse.
        ref_axes, query_axes
            Override the volume's ``(z, y, x)`` and the section's ``(y, x)`` axes. Default
            to the ones the fit ran on.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_grid_row_col

        if direction not in {"forward", "backward"}:
            raise ValueError(f"Expected `direction` to be 'forward' or 'backward', found {direction!r}.")
        volume_axes = tuple(self.ref_axes if ref_axes is None else ref_axes)
        section_axes = tuple(self.query_axes if query_axes is None else query_axes)
        if len(volume_axes) != 3 or len(section_axes) != 2:
            raise ValueError(
                f"Expected 3 reference axes and 2 query axes, found {len(volume_axes)} and {len(section_axes)}."
            )
        # The same lift `fit_stalign_volume` applies: a single-sample z axis at the origin
        # turns the flat section into something the rank-3 solver can address.
        lifted = (jnp.zeros(1, dtype=jax_dtype()), *section_axes)
        axes = volume_axes if direction == "forward" else lifted
        return transform_grid_row_col(axes, self.velocity_grid, self.velocity, self.affine, direction=direction)

    def transform(self, points: npt.ArrayLike) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` section points to ``(N, 3)`` ``(x, y, z)`` reference points.

        This owns both halves of the convention change: the lift of a flat section onto
        the ``z = 0`` plane, and the reversal between the caller's ``(x, y, z)`` and the
        solver's ``(z, y, x)``. Evaluated at each point rather than at the nearest raster
        cell, so it does not quantise to the fit's grid.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_points_row_col

        pts = jnp.asarray(points, dtype=jax_dtype())
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) `(x, y)` array, found shape {pts.shape}.")
        # The section is the fixed image, so mapping it into the reference is the
        # *backward* direction -- the same map the objective samples the volume through.
        lifted = jnp.stack((jnp.zeros(pts.shape[0], dtype=pts.dtype), pts[:, 1], pts[:, 0]), axis=1)
        transformed = transform_points_row_col(
            self.velocity_grid, self.velocity, self.affine, lifted, direction="backward"
        )
        return transformed[:, ::-1]


# Why the full 3D deformation rather than an affine plane plus an in-plane 2D fit: on a
# MERFISH-into-Allen-CCF run the diffeomorphism bends the fitted surface 2-5 voxels away
# from its affine plane for the outer 5% of cells -- one to two cortical layers. The
# `initial_*` arguments are therefore an initialisation, not the answer.
def fit_stalign_volume(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    ref_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    query_scale: tuple[float, float] = (1.0, 1.0),
    ref_axes: Sequence[npt.ArrayLike] | None = None,
    query_axes: Sequence[npt.ArrayLike] | None = None,
    initial_slice: int | None = None,
    initial_rotation: float = 0.0,
    initial_scale: float = 1.0,
    initial_affine: npt.ArrayLike | None = None,
    **solver_kwargs: Unpack[StalignVolumeSolverKwargs],
) -> Stalign3DResult:
    """Fit a single 2D section into a 3D reference volume, array-in / array-out.

    Internal: :func:`~squidpy.experimental.tl.align_stalign_volume` is the container-aware
    entry point and carries the user-facing documentation.

    Parameters
    ----------
    ref
        Reference volume, channels-first ``(c, z, y, x)``; a bare ``(z, y, x)`` array is
        promoted. Need not match ``query``'s channel count.
    query
        The section, channels-first ``(c, y, x)``; a bare ``(y, x)`` array is promoted.
    ref_scale, query_scale
        Physical size of one voxel/pixel, ``(z, y, x)`` and ``(y, x)``. Builds centred
        axes when the matching ``*_axes`` is not given.
    ref_axes, query_axes
        Explicit physical axes, ``(z, y, x)`` and ``(y, x)``, resolved per side.
    initial_slice
        Index along the reference's first axis to centre the section on; sets the
        translation's out-of-plane component. ``None`` centres on the middle.
    initial_rotation, initial_scale
        In-plane rotation (**radians**) and uniform scale of the initial affine.
    initial_affine
        Homogeneous ``(4, 4)`` affine in ``(x, y, z)`` order, replacing the three
        ``initial_*`` arguments above and mutually exclusive with them.
    solver_kwargs
        See :class:`StalignVolumeSolverKwargs`.

    Returns
    -------
    A :class:`Stalign3DResult`.
    """
    _require_jax()

    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype, lddmm
    from ._stalign_impl._helpers import affine_xy_to_rc, as_chw, resolve_axes

    opts: dict[str, Any] = _VOLUME_DEFAULTS | solver_kwargs
    dtype = jax_dtype()

    target_image = as_chw(query, name="query", ndim=2)
    source_image = as_chw(ref, name="ref", ndim=3)
    if source_image.shape[1] < 2:
        # A `(c, y, x)` section passed as the reference reads as a one-voxel-deep volume,
        # which is the likely way to arrive here. Named explicitly because there is no
        # out-of-plane information in it to fit -- `align_stalign_image` is the 2D path.
        raise ValueError(
            f"Expected `ref` to be a volume with at least two samples along `z`, found depth "
            f"{source_image.shape[1]}. A single plane carries no out-of-plane information; use "
            f"`align_stalign_image` to register two 2D images."
        )

    # The reference is the moving image: it is the volume that gets warped onto the
    # section, so it plays LDDMM's `I`/`xI` role and the section plays `J`/`xJ`.
    source_grid = resolve_axes(ref_axes, ref_scale, source_image.shape[1:], "ref_axes")
    section_grid = resolve_axes(query_axes, query_scale, target_image.shape[1:], "query_axes")
    # Upstream's whole 3D-to-slice special case: give the section a single-sample z axis
    # at the origin and a length-1 z extent, and the rank-3 solver does the rest.
    target_grid = (jnp.zeros(1, dtype=dtype), *section_grid)
    target_image = target_image[:, None]

    if initial_affine is not None:
        if initial_slice is not None or initial_rotation != 0.0 or initial_scale != 1.0:
            raise ValueError(
                "`initial_affine` replaces the `initial_slice` / `initial_rotation` / "
                "`initial_scale` construction, so they are mutually exclusive."
            )
        linear, translation = affine_xy_to_rc(initial_affine, ndim=3)
    else:
        slice_index = source_image.shape[1] // 2 if initial_slice is None else initial_slice
        if not -source_image.shape[1] <= slice_index < source_image.shape[1]:
            raise ValueError(
                f"`initial_slice={initial_slice}` is outside the reference's first axis "
                f"of length {source_image.shape[1]}."
            )
        cos, sin = jnp.cos(initial_rotation), jnp.sin(initial_rotation)
        # Rotation about the out-of-plane axis, then a uniform scale, in `(z, y, x)`.
        linear = initial_scale * jnp.array(
            [[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]],
            dtype=dtype,
        )
        translation = jnp.asarray(
            [
                -source_grid[0][slice_index],
                jnp.mean(target_grid[1]),
                jnp.mean(target_grid[2]),
            ],
            dtype=dtype,
        )

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        **{key: value for key, value in opts.items() if key not in _CONSUMED_KEYS},
    )
    return Stalign3DResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        ref_axes=source_grid,
        query_axes=section_grid,
        match_weights=result["WM"],
        artifact_weights=result["WA"],
        background_weights=result["WB"],
        energies=result["energies"],
        n_iter=int(result["n_iter"]),
    )


def _require_jax() -> None:
    """Fail with an actionable message rather than a bare ImportError on the optional extra.

    Called at the top of each fit function: JAX is imported inside them, not at module
    scope, so ``import squidpy.experimental`` stays cheap and installable without it.
    """
    try:
        import jax  # noqa: F401
    except ImportError as e:
        raise ImportError(_JAX_REQUIRED) from e


def _initial_affine_and_landmarks(
    landmarks_ref: npt.ArrayLike | None,
    landmarks_query: npt.ArrayLike | None,
    initial_affine: npt.ArrayLike | None,
) -> tuple[JaxArray, JaxArray, JaxArray | None, JaxArray | None]:
    """Resolve the starting affine and the row-col landmark pair the point term uses.

    Shared by the point-cloud and image paths: they differ in what they rasterize, not in
    the landmark contract. Landmarks are ``(x, y)``, matched by row order, and in the same
    units as the fit's coordinates -- cell coordinates for a point-cloud fit, the images'
    physical axes for an image fit.

    The two initialisers are not exclusive. Landmarks play two roles: they always
    contribute the point-matching term the solver weights by ``sigmaP``, and they *also*
    derive the starting affine when ``initial_affine`` is absent. Passing both is how you
    keep the matching term while pinning the start yourself.
    """
    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype
    from ._stalign_impl._helpers import affine_from_points, affine_xy_to_rc, validate_points

    if (landmarks_ref is None) != (landmarks_query is None):
        raise ValueError("Expected both landmark arrays to be provided together.")

    dtype = jax_dtype()
    source_landmarks = target_landmarks = None
    if landmarks_ref is not None:
        # The solver runs in row-col (y, x); landmarks arrive as (x, y) like every other
        # public coordinate, so they swap at the same boundary the clouds do.
        source_landmarks = validate_points(landmarks_query, name="landmarks_query")[:, ::-1]
        target_landmarks = validate_points(landmarks_ref, name="landmarks_ref")[:, ::-1]

    if initial_affine is not None:
        linear, translation = affine_xy_to_rc(initial_affine)
    elif source_landmarks is not None:
        linear_np, translation_np = affine_from_points(source_landmarks, target_landmarks)
        linear = jnp.asarray(linear_np, dtype=dtype)
        translation = jnp.asarray(translation_np, dtype=dtype)
    else:
        linear, translation = jnp.eye(2, dtype=dtype), jnp.zeros(2, dtype=dtype)
    return linear, translation, source_landmarks, target_landmarks


def fit_stalign_obs(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    landmarks_ref: npt.ArrayLike | None = None,
    landmarks_query: npt.ArrayLike | None = None,
    **solver_kwargs: Unpack[StalignObsSolverKwargs],
) -> Stalign2DResult:
    """Fit a deformation mapping the ``query`` cloud onto the ``ref`` cloud.

    Internal: :func:`~squidpy.experimental.tl.align_stalign_obs` is the container-aware
    entry point and carries the user-facing documentation.

    Parameters
    ----------
    ref, query
        ``(N, 2)`` / ``(M, 2)`` point clouds in ``(x, y)`` order; the query is aligned
        onto the reference.
    landmarks_ref, landmarks_query
        Paired ``(x, y)`` landmark arrays initialising the affine, matched by row order.
        Must be given together, and are mutually exclusive with ``initial_affine``.
    solver_kwargs
        See :class:`StalignObsSolverKwargs`.

    Returns
    -------
    A :class:`Stalign2DResult`; its ``aligned_points`` is ``query`` already mapped.
    """
    _require_jax()

    from ._stalign_impl._core import lddmm, transform_points_row_col
    from ._stalign_impl._helpers import rasterize_cloud, validate_points

    opts = _OBS_DEFAULTS | solver_kwargs
    linear, translation, src_lm, tgt_lm = _initial_affine_and_landmarks(
        landmarks_ref, landmarks_query, opts.get("initial_affine")
    )

    # The solver runs internally in row-col (y, x); inputs are (x, y) -- swap at the boundary.
    source_rc = validate_points(query, name="query")[:, ::-1]
    target_rc = validate_points(ref, name="ref")[:, ::-1]
    raster = {"dx": opts["dx"], "blur": opts["blur"], "expand": opts["raster_expand"]}
    source_grid, source_image = rasterize_cloud(source_rc, **raster)
    target_grid, target_image = rasterize_cloud(target_rc, **raster)

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        points_source=src_lm,
        points_target=tgt_lm,
        **{key: value for key, value in opts.items() if key not in _CONSUMED_KEYS},
    )
    aligned_rc = transform_points_row_col(result["xv"], result["v"], result["A"], source_rc, direction="forward")
    return Stalign2DResult(
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


def fit_stalign_image(
    ref: npt.ArrayLike,
    query: npt.ArrayLike,
    *,
    ref_scale: tuple[float, float] = (1.0, 1.0),
    query_scale: tuple[float, float] = (1.0, 1.0),
    ref_axes: Sequence[npt.ArrayLike] | None = None,
    query_axes: Sequence[npt.ArrayLike] | None = None,
    landmarks_ref: npt.ArrayLike | None = None,
    landmarks_query: npt.ArrayLike | None = None,
    **solver_kwargs: Unpack[StalignImageSolverKwargs],
) -> Stalign2DResult:
    """Fit a deformation mapping the ``query`` image onto the ``ref`` image.

    Internal: :func:`~squidpy.experimental.tl.align_stalign_image` is the container-aware
    entry point and carries the user-facing documentation.

    Parameters
    ----------
    ref, query
        Channels-first ``(c, y, x)`` rasters; a bare ``(y, x)`` array is promoted. They
        need not share a shape nor a channel count.
    ref_scale, query_scale
        Physical size of one pixel as ``(y, x)``, defaulting to pixel units. Builds
        centred axes when the matching ``*_axes`` is not given.
    ref_axes, query_axes
        Explicit physical row/column axes, resolved per side: either may be given alone,
        and each is mutually exclusive with a non-unit scale on *its own* side only.
    landmarks_ref, landmarks_query
        Paired ``(x, y)`` landmark arrays in the *images'* physical units, matched by row
        order. See :func:`_initial_affine_and_landmarks` for how they combine with
        ``initial_affine``.
    solver_kwargs
        See :class:`StalignImageSolverKwargs`.

    Returns
    -------
    A :class:`Stalign2DResult`.
    """
    _require_jax()

    from ._stalign_impl._core import lddmm
    from ._stalign_impl._helpers import as_chw, resolve_axes

    opts = _IMAGE_DEFAULTS | solver_kwargs
    linear, translation, src_lm, tgt_lm = _initial_affine_and_landmarks(
        landmarks_ref, landmarks_query, opts.get("initial_affine")
    )

    source_image = as_chw(query, name="query")
    target_image = as_chw(ref, name="ref")

    # Explicit axes and a non-unit scale are mutually exclusive within a side; see
    # `resolve_axes`, which both ranks share so the rule cannot differ between them.
    source_grid = resolve_axes(query_axes, query_scale, source_image.shape[1:], "query_axes")
    target_grid = resolve_axes(ref_axes, ref_scale, target_image.shape[1:], "ref_axes")

    result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        points_source=src_lm,
        points_target=tgt_lm,
        **{key: value for key, value in opts.items() if key not in _CONSUMED_KEYS},
    )
    return Stalign2DResult(
        affine=result["A"],
        velocity=result["v"],
        velocity_grid=result["xv"],
        query_axes=source_grid,
        ref_axes=target_grid,
        match_weights=result["WM"],
        artifact_weights=result["WA"],
        background_weights=result["WB"],
        energies=result["energies"],
        n_iter=int(result["n_iter"]),
    )
