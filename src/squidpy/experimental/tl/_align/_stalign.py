"""STalign estimator: JAX LDDMM registration, at rank 2 and rank 3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, TypedDict, Unpack

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import jax
    from anndata import AnnData
    from spatialdata import SpatialData

    JaxArray = jax.Array
else:
    # Bound at runtime, not just under TYPE_CHECKING: `sphinx_autodoc_typehints` calls
    # `get_type_hints` on the fit classes, and jax is an optional extra that must not be
    # imported here. `Any` is the placeholder that keeps that resolvable.
    JaxArray = Any


class StalignImageSolverKwargs(TypedDict, total=False):
    """The LDDMM controls :func:`~squidpy.experimental.tl.stalign_align_image` takes.

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
    """LDDMM solver tuning accepted by :func:`~squidpy.experimental.tl.stalign_align_volume`.

    :class:`~squidpy.experimental.tl.StalignImageSolverKwargs` minus two keys: ``sigmaP``,
    because the rank-3 path has no point-matching energy and it would be a knob that does
    nothing, and ``initial_affine``, which is a named ``(4, 4)`` argument on
    :func:`~squidpy.experimental.tl.stalign_align_volume` rather than a solver key. Five
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


def _check_direction(value: object) -> None:
    """Reject a ``direction`` that is neither of the two the solver understands."""
    if value not in {"forward", "backward"}:
        raise ValueError(f"Expected `direction` to be 'forward' or 'backward', found {value!r}.")


@dataclass(frozen=True, kw_only=True)
class StalignFit:
    """A fitted STalign diffeomorphism.

    One concrete subclass per ``stalign_align_*`` entry point. What a fit can do follows
    from what it was fitted from, so the frame-dependent operations live on the subclasses
    that carry a frame rather than raising at runtime on the ones that do not.

    """

    #: Homogeneous affine in the solver's row-column order -- ``(3, 3)`` at rank 2,
    #: ``(4, 4)`` at rank 3.
    affine: JaxArray
    #: The fitted velocity field, row-column.
    velocity: JaxArray
    #: The axes the velocity field lives on, row-column.
    velocity_grid: tuple[JaxArray, ...]
    #: Raster-shaped -- the mixture model's per-pixel match posterior.
    match_weights: JaxArray | None = None
    #: Raster-shaped -- the mixture model's per-pixel artifact posterior.
    artifact_weights: JaxArray | None = None
    #: Raster-shaped -- the mixture model's per-pixel background posterior.
    background_weights: JaxArray | None = None
    #: ``(niter,)`` -- the objective trace; slice it with :attr:`n_iter`.
    energies: JaxArray | None = None
    #: ``int`` -- the iteration the fit stopped at.
    n_iter: int | None = None

    #: Dimensionality of the reference frame the fit maps into.
    rank: ClassVar[Literal[2, 3]] = 2
    #: Which entry point produced the fit. The discriminant :meth:`from_uns` decodes on,
    #: because ``rank`` alone does not separate an obs fit from an image fit.
    kind: ClassVar[Literal["obs", "image", "volume"]]

    def transform_points(
        self,
        points: npt.ArrayLike,
        *,
        direction: Literal["forward", "backward"] = "forward",
    ) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` points through the fit.

        Evaluated per point, not at the nearest raster cell, so it does not quantise to the
        fit's grid. ``"forward"`` maps the query into the reference frame, ``"backward"``
        the reverse -- both meaningful at rank 2, where the two images are flat.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_points_row_col

        _check_direction(direction)
        pts = jnp.asarray(points, dtype=jax_dtype())
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) `(x, y)` array, found shape {pts.shape}.")
        # The solver runs in row-col; swap at both boundaries.
        transformed_rc = transform_points_row_col(
            self.velocity_grid, self.velocity, self.affine, pts[:, ::-1], direction=direction
        )
        return transformed_rc[:, ::-1]

    def transform(
        self,
        data: AnnData | SpatialData,
        *,
        key_added: str = "spatial_aligned",
        spatial_key: str = "spatial",
        table_key: str | None = None,
        coordinate_system: str | None = None,
        inplace: bool = True,
    ) -> np.ndarray | None:
        """Write the fit's transformed coordinates into a container's ``obsm``.

        The container-level counterpart of :meth:`transform_points`: read
        ``obsm[spatial_key]``, map it through the fit, store it under ``obsm[key_added]``.
        A rank-2 fit writes ``(N, 2)`` coordinates in the reference frame; a rank-3 fit
        writes ``(N, 3)`` ``(x, y, z)`` positions in the reference volume.

        Parameters
        ----------
        data
            The container holding the query coordinates.
        key_added
            ``obsm`` key to write the transformed coordinates under.
        spatial_key
            ``obsm`` key holding the coordinates to map.
        table_key
            For a :class:`~spatialdata.SpatialData`, which table to read and write.
        coordinate_system
            The frame ``spatial_key`` is expected to sit in. ``None`` (default) uses the
            one the fit's own units came from, which is the frame that actually has to
            match; pass a value only to check against a different one.
        inplace
            ``True`` (default) writes ``obsm[key_added]`` and returns ``None``. ``False``
            leaves ``data`` untouched and returns the transformed coordinates.

        Returns
        -------
        ``None``, or the ``(N, 2)`` / ``(N, 3)`` transformed coordinates when
        ``inplace=False``.
        """
        from ._api import apply_fit_to_container

        return apply_fit_to_container(
            self,
            data,
            key_added=key_added,
            spatial_key=spatial_key,
            table_key=table_key,
            coordinate_system=coordinate_system,
            inplace=inplace,
        )

    def to_uns(
        self,
        data: AnnData | SpatialData,
        *,
        key: str = "stalign",
        table_key: str | None = None,
    ) -> None:
        """Store the fit in a table's ``uns``, in a form that survives a write.

        ``uns`` is the only place either container has for a fit: a diffeomorphism is not a
        SpatialData transformation. Not automatic, because a rank-3 velocity field runs to
        hundreds of megabytes.
        """
        from ._api import store_fit_on_container

        store_fit_on_container(self, data, key=key, table_key=table_key)

    @classmethod
    def from_uns(
        cls,
        data: AnnData | SpatialData,
        *,
        key: str = "stalign",
        table_key: str | None = None,
    ) -> StalignFit:
        """Decode a fit written by :meth:`to_uns`, as numpy arrays.

        Returns whichever class the stored ``kind`` names, whatever class this is called on.
        """
        from ._api import load_fit_from_container

        return load_fit_from_container(data, key=key, table_key=table_key)


@dataclass(frozen=True, kw_only=True)
class StalignObsFit(StalignFit):
    """A fit from :func:`~squidpy.experimental.tl.stalign_align_obs`.

    Both clouds are rasterised into density images at ``dx`` and those are what the fit
    ran on -- not a frame any real image lives on, so no raster axes survive and there is
    no ``deformation_grid`` or ``warp_image`` to resample the wrong grid with.
    """

    velocity_grid: tuple[JaxArray, JaxArray]
    #: ``(N, 2)`` ``(x, y)`` -- the fitted query cloud already mapped into the reference frame.
    aligned_points: JaxArray | None = None

    rank: ClassVar[Literal[2]] = 2
    kind: ClassVar[Literal["obs"]] = "obs"


@dataclass(frozen=True, kw_only=True)
class StalignImageFit(StalignFit):
    """A fit from :func:`~squidpy.experimental.tl.stalign_align_image`.

    Both sides are real images, so both frames survive and the frame-dependent operations
    need no axes from the caller.
    """

    velocity_grid: tuple[JaxArray, JaxArray]
    #: Row-column physical axes of the query raster the fit ran on.
    query_axes: tuple[JaxArray, JaxArray]
    #: Row-column physical axes of the reference raster the fit ran on.
    ref_axes: tuple[JaxArray, JaxArray]
    #: The query coordinate system the fit's units came from. :meth:`StalignFit.transform`
    #: checks the coordinates it is given against this frame.
    coordinate_system: str = "global"

    rank: ClassVar[Literal[2]] = 2
    kind: ClassVar[Literal["image"]] = "image"

    def deformation_grid(
        self,
        *,
        direction: Literal["forward", "backward"] = "forward",
        query_axes: Sequence[npt.ArrayLike] | None = None,
        ref_axes: Sequence[npt.ArrayLike] | None = None,
    ) -> JaxArray:
        """The dense row-column coordinate transform of the fit, shape ``(2, *grid)``.

        The *same* call on the *same* fitted ``affine``/``velocity``/``velocity_grid`` that
        the objective samples through -- not an approximation for plotting. Given the same
        axes it agrees with the internal transform exactly.

        ``"forward"`` evaluates the query grid in the reference frame; ``"backward"`` the
        reverse. ``query_axes`` / ``ref_axes`` override the axes the fit ran on, for
        evaluating the transform on a raster other than the fitted one.
        """
        from ._stalign_impl._core import transform_grid_row_col

        _check_direction(direction)
        source_axes = self.query_axes if query_axes is None else query_axes
        target_axes = self.ref_axes if ref_axes is None else ref_axes
        axes = source_axes if direction == "forward" else target_axes
        return transform_grid_row_col(axes, self.velocity_grid, self.velocity, self.affine, direction=direction)

    def warp_image(
        self,
        image: npt.ArrayLike,
        *,
        direction: Literal["forward", "backward"] = "forward",
        query_axes: Sequence[npt.ArrayLike] | None = None,
        ref_axes: Sequence[npt.ArrayLike] | None = None,
    ) -> JaxArray:
        """Resample an image through the fit.

        ``"forward"`` resamples the query image onto the reference's grid; ``"backward"``
        the reference onto the query's. Resampling a *forward* map means sampling through
        the *backward* grid, which is why the two are crossed below.

        ``query_axes`` / ``ref_axes`` override the axes the fit ran on -- for warping an
        image of the same scene at a different resolution than the fit used.
        """
        from ._stalign_impl._core import interp
        from ._stalign_impl._helpers import as_chw

        _check_direction(direction)
        arr = as_chw(image, name="image")
        source_axes = self.query_axes if query_axes is None else query_axes
        target_axes = self.ref_axes if ref_axes is None else ref_axes
        grid = self.deformation_grid(
            direction="backward" if direction == "forward" else "forward",
            query_axes=source_axes,
            ref_axes=target_axes,
        )
        sampling_axes = source_axes if direction == "forward" else target_axes
        return interp(sampling_axes, arr, grid)


@dataclass(frozen=True, kw_only=True)
class StalignVolumeFit(StalignFit):
    """A fit from :func:`~squidpy.experimental.tl.stalign_align_volume`.

    Places a flat section in a 3D reference. Pair :meth:`transform_points` with
    :func:`~squidpy.experimental.im.sample_volume` to read a reference volume at the mapped
    points. There is no ``warp_image``: the reference is a volume and the section is a plane
    through it, so there is no image to resample.

    """

    velocity_grid: tuple[JaxArray, JaxArray, JaxArray]
    #: Physical ``(z, y, x)`` axes of the reference volume the fit ran on.
    ref_axes: tuple[JaxArray, JaxArray, JaxArray]
    #: Physical ``(y, x)`` axes of the section the fit ran on.
    query_axes: tuple[JaxArray, JaxArray]
    #: The query coordinate system the fit's units came from. :meth:`StalignFit.transform`
    #: checks the coordinates it is given against this frame.
    coordinate_system: str = "global"

    rank: ClassVar[Literal[3]] = 3
    kind: ClassVar[Literal["volume"]] = "volume"

    def transform_points(
        self,
        points: npt.ArrayLike,
        *,
        direction: Literal["forward", "backward"] | None = None,
    ) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` section points to ``(N, 3)`` ``(x, y, z)`` in the volume.

        ``direction`` is accepted only to keep the signature compatible with
        :meth:`StalignFit.transform_points` and must be left at ``None``: the section is the
        fixed image and it is flat, so only the section-into-volume map is defined.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_points_row_col

        if direction is not None:
            raise ValueError(
                "`direction` does not apply at rank 3: the section is the fixed image and it "
                "is flat, so only the section-into-volume map is defined."
            )
        pts = jnp.asarray(points, dtype=jax_dtype())
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) `(x, y)` array, found shape {pts.shape}.")
        # This owns both halves of the convention change: the lift of a flat section onto the
        # `z = 0` plane, and the reversal between the caller's `(x, y, z)` and the solver's
        # `(z, y, x)`. The section is the fixed image, so mapping it into the reference is the
        # *backward* direction -- the same map the objective samples the volume through.
        lifted = jnp.stack((jnp.zeros(pts.shape[0], dtype=pts.dtype), pts[:, 1], pts[:, 0]), axis=1)
        transformed = transform_points_row_col(
            self.velocity_grid, self.velocity, self.affine, lifted, direction="backward"
        )
        return transformed[:, ::-1]

    def deformation_grid(
        self,
        *,
        direction: Literal["forward", "backward"] = "backward",
        query_axes: Sequence[npt.ArrayLike] | None = None,
        ref_axes: Sequence[npt.ArrayLike] | None = None,
    ) -> JaxArray:
        """The dense row-column coordinate transform of the fit, shape ``(3, *grid)``.

        Which element is fixed flips with the rank, so the natural direction does too: the
        *volume* is the moving image here, so ``"backward"`` -- evaluating the section's
        lifted grid in the volume's frame -- is the default and the one the objective
        samples through.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_grid_row_col

        _check_direction(direction)
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
) -> StalignVolumeFit:
    """Fit a single 2D section into a 3D reference volume, array-in / array-out.

    Internal: :func:`~squidpy.experimental.tl.stalign_align_volume` is the container-aware
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
    A :class:`StalignVolumeFit`.
    """
    _require_jax()

    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype, lddmm
    from ._stalign_impl._helpers import affine_xy_to_rc, as_chw, resolve_axes

    # `dict[str, Any]`, not the TypedDict: merging two different TypedDicts with `|` is
    # not an operation the type system defines, and the values are heterogeneous anyway.
    opts: dict[str, Any] = _VOLUME_DEFAULTS | solver_kwargs
    dtype = jax_dtype()

    target_image = as_chw(query, name="query", ndim=2)
    source_image = as_chw(ref, name="ref", ndim=3)
    if source_image.shape[1] < 2:
        # A `(c, y, x)` section passed as the reference reads as a one-voxel-deep volume,
        # which is the likely way to arrive here. Named explicitly because there is no
        # out-of-plane information in it to fit -- `stalign_align_image` is the 2D path.
        raise ValueError(
            f"Expected `ref` to be a volume with at least two samples along `z`, found depth "
            f"{source_image.shape[1]}. A single plane carries no out-of-plane information; use "
            f"`stalign_align_image` to register two 2D images."
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
        # TODO: this construction assumes the reference volume's in-plane axes are centred
        # on the origin, which holds for `centred_axes` (the `*_scale` path) but never for
        # `_element_axes` -- so the public `stalign_align_volume` starts the section at the
        # volume's in-plane *corner*, half the extent away in y and x. `initial_scale` is
        # wrong for the same reason: it scales `linear` without the translation
        # compensating, moving the selected slice. The fix is
        # `T = centre_section - linear @ centre_volume`. Deferred: it changes the numerics
        # of every rank-3 fit, so it wants its own commit and a regression test asserting
        # placement rather than shape.
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

    fit_result = lddmm(
        source_grid,
        source_image,
        target_grid,
        target_image,
        L=linear,
        T=translation,
        **{key: value for key, value in opts.items() if key not in _CONSUMED_KEYS},
    )
    return StalignVolumeFit(
        affine=fit_result["A"],
        velocity=fit_result["v"],
        velocity_grid=fit_result["xv"],
        # Indexed rather than passed whole: `resolve_axes` is rank-agnostic and returns a
        # variadic tuple, while the class documents the exact arity its rank implies.
        ref_axes=(source_grid[0], source_grid[1], source_grid[2]),
        query_axes=(section_grid[0], section_grid[1]),
        match_weights=fit_result["WM"],
        artifact_weights=fit_result["WA"],
        background_weights=fit_result["WB"],
        energies=fit_result["energies"],
        n_iter=int(fit_result["n_iter"]),
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
) -> StalignObsFit:
    """Fit a deformation mapping the ``query`` cloud onto the ``ref`` cloud.

    Internal: :func:`~squidpy.experimental.tl.stalign_align_obs` is the container-aware
    entry point and carries the user-facing documentation.

    Parameters
    ----------
    ref, query
        ``(N, 2)`` / ``(M, 2)`` point clouds in ``(x, y)`` order; the query is aligned
        onto the reference.
    landmarks_ref, landmarks_query
        Paired ``(x, y)`` landmark arrays initialising the affine, matched by row order.
        Must be given together. Not exclusive with ``initial_affine``: landmarks always
        contribute the point-matching term, and additionally derive the starting affine
        when ``initial_affine`` is absent.
    solver_kwargs
        See :class:`StalignObsSolverKwargs`.

    Returns
    -------
    A :class:`StalignObsFit`; its ``aligned_points`` is ``query`` already mapped.
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

    fit_result = lddmm(
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
    aligned_rc = transform_points_row_col(
        fit_result["xv"], fit_result["v"], fit_result["A"], source_rc, direction="forward"
    )
    return StalignObsFit(
        affine=fit_result["A"],
        velocity=fit_result["v"],
        velocity_grid=fit_result["xv"],
        aligned_points=aligned_rc[:, ::-1],
        match_weights=fit_result["WM"],
        artifact_weights=fit_result["WA"],
        background_weights=fit_result["WB"],
        energies=fit_result["energies"],
        n_iter=int(fit_result["n_iter"]),
        # No raster axes: the grids here are the internal density rasters at `dx`
        # resolution, not a frame any real image lives on. Offering `warp_image`
        # off them would quietly resample the caller's image onto a coarse, unrelated grid.
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
) -> StalignImageFit:
    """Fit a deformation mapping the ``query`` image onto the ``ref`` image.

    Internal: :func:`~squidpy.experimental.tl.stalign_align_image` is the container-aware
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
    A :class:`StalignImageFit`.
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

    fit_result = lddmm(
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
    return StalignImageFit(
        affine=fit_result["A"],
        velocity=fit_result["v"],
        velocity_grid=fit_result["xv"],
        query_axes=(source_grid[0], source_grid[1]),
        ref_axes=(target_grid[0], target_grid[1]),
        match_weights=fit_result["WM"],
        artifact_weights=fit_result["WA"],
        background_weights=fit_result["WB"],
        energies=fit_result["energies"],
        n_iter=int(fit_result["n_iter"]),
    )
