"""STalign estimator: JAX LDDMM registration, at rank 2 and rank 3."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Unpack

import numpy as np
from fast_array_utils.types import HasArrayNamespace as Array

from squidpy._params import resolve_params
from squidpy.types import (
    StalignImageParams,
    StalignObsParams,
    StalignVolumeParams,
)

if TYPE_CHECKING:
    import jax
    from anndata import AnnData
    from spatialdata import SpatialData

    JaxArray = jax.Array
else:
    JaxArray = Any


#: Keys the fit functions consume themselves rather than forward to the solver.
_CONSUMED_KEYS = frozenset({"dx", "blur", "raster_expand", "initial_affine"})

_JAX_REQUIRED = 'STalign alignment requires JAX: `pip install "squidpy[jax]"`.'


@dataclass(frozen=True, kw_only=True)
class StalignFit:
    """A fitted STalign diffeomorphism.

    The base of :class:`~squidpy.experimental.tl.StalignObsFit`,
    :class:`~squidpy.experimental.tl.StalignImageFit` and
    :class:`~squidpy.experimental.tl.StalignVolumeFit`, one per ``stalign_align_*`` entry
    point; frame-dependent operations live on the subclasses that carry a frame. Use it to
    annotate any fit, and for the :meth:`from_uns` classmethod.
    """

    #: Homogeneous affine in the solver's row-column order, ``(3, 3)`` at rank 2,
    #: ``(4, 4)`` at rank 3.
    affine: JaxArray
    #: The fitted velocity field, row-column.
    velocity: JaxArray
    #: The axes the velocity field lives on, row-column.
    velocity_grid: tuple[JaxArray, ...]
    #: The mixture model's per-pixel match posterior, raster-shaped.
    match_weights: JaxArray | None = None
    #: The mixture model's per-pixel artifact posterior, raster-shaped.
    artifact_weights: JaxArray | None = None
    #: The mixture model's per-pixel background posterior, raster-shaped.
    background_weights: JaxArray | None = None
    #: The objective trace, ``(niter,)``; slice it with :attr:`n_iter`.
    energies: JaxArray | None = None
    #: The iteration the fit stopped at.
    n_iter: int | None = None

    #: Dimensionality of the reference frame the fit maps into.
    rank: ClassVar[Literal[2, 3]] = 2
    #: Which entry point produced the fit; what :meth:`from_uns` decodes on.
    kind: ClassVar[Literal["obs", "image", "volume"]]

    def transform_points(self, points: Array) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` query points into the reference frame.

        Evaluated per point, not at the nearest raster cell, so it does not quantise to the
        fit's grid.
        """
        return self._map_points(points, direction="forward")

    def _map_points(self, points: Array, *, direction: Literal["forward", "backward"]) -> JaxArray:
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_points_row_col

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
        inplace
            ``True`` (default) writes ``obsm[key_added]`` and returns ``None``. ``False``
            leaves ``data`` untouched and returns the transformed coordinates.

        Returns
        -------
        ``None``, or the ``(N, 2)`` / ``(N, 3)`` transformed coordinates when
        ``inplace=False``.
        """
        from spatialdata import SpatialData

        from ._api import _assert_table_coords_share_frame, _resolve_table, _write_coords

        fitted_in = getattr(self, "coordinate_system", None)  # only the raster fits record one
        if isinstance(data, SpatialData) and fitted_in is not None:
            _assert_table_coords_share_frame(
                data,
                adata=_resolve_table(data, table_key, side="query"),
                spatial_key=spatial_key,
                coordinate_system=fitted_in,
            )
        return _write_coords(
            data,
            table_key,
            spatial_key,
            key_added,
            transform=self.transform_points,
            spatial_key_name="spatial_key",
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
        from ._api import _resolve_table
        from ._io import fit_to_uns

        fit_to_uns(self, _resolve_table(data, table_key, side="query"), key)

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
        from ._api import _resolve_table
        from ._io import fit_from_uns

        return fit_from_uns(_resolve_table(data, table_key, side="query"), key)


@dataclass(frozen=True, kw_only=True)
class StalignObsFit(StalignFit):
    """A fit from :func:`~squidpy.experimental.tl.stalign_align_obs`.

    Both clouds are rasterised into density images at ``dx`` and those are what the fit ran
    on. That is not a frame any real image lives on, so no raster axes survive and there is
    no ``deformation_grid`` or ``warp_image``.
    """

    velocity_grid: tuple[JaxArray, JaxArray]
    #: The fitted query cloud, ``(N, 2)`` ``(x, y)``, already mapped into the reference frame.
    aligned_points: JaxArray | None = None

    rank: ClassVar[Literal[2]] = 2
    kind: ClassVar[Literal["obs"]] = "obs"

    def inverse_transform_points(self, points: Array) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` reference points into the query frame."""
        return self._map_points(points, direction="backward")


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

    def inverse_transform_points(self, points: Array) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` reference points into the query frame."""
        return self._map_points(points, direction="backward")

    def deformation_grid(
        self,
        *,
        query_axes: Sequence[Array] | None = None,
        ref_axes: Sequence[Array] | None = None,
    ) -> JaxArray:
        """The query grid evaluated in the reference frame, row-column, shape ``(2, *grid)``.

        The same transform the objective uses, not an approximation for plotting.
        ``query_axes`` / ``ref_axes`` override the axes the fit ran on, for a raster other
        than the fitted one.
        """
        return self._grid("forward", query_axes, ref_axes)

    def warp_image(
        self,
        image: Array,
        *,
        query_axes: Sequence[Array] | None = None,
        ref_axes: Sequence[Array] | None = None,
    ) -> JaxArray:
        """Resample a query image onto the reference grid.

        ``query_axes`` / ``ref_axes`` override the axes the fit ran on: for an image of the
        same scene at a different resolution than the fit used.
        """
        return self._warp(image, "forward", query_axes, ref_axes)

    def inverse_warp_image(
        self,
        image: Array,
        *,
        query_axes: Sequence[Array] | None = None,
        ref_axes: Sequence[Array] | None = None,
    ) -> JaxArray:
        """Resample a reference image onto the query grid; see :meth:`warp_image`."""
        return self._warp(image, "backward", query_axes, ref_axes)

    def _grid(
        self,
        direction: Literal["forward", "backward"],
        query_axes: Sequence[Array] | None,
        ref_axes: Sequence[Array] | None,
    ) -> JaxArray:
        from ._stalign_impl._core import transform_grid_row_col

        source_axes = self.query_axes if query_axes is None else query_axes
        target_axes = self.ref_axes if ref_axes is None else ref_axes
        axes = source_axes if direction == "forward" else target_axes
        return transform_grid_row_col(axes, self.velocity_grid, self.velocity, self.affine, direction=direction)

    def _warp(
        self,
        image: Array,
        direction: Literal["forward", "backward"],
        query_axes: Sequence[Array] | None,
        ref_axes: Sequence[Array] | None,
    ) -> JaxArray:
        from ._stalign_impl._core import interp
        from ._stalign_impl._helpers import as_chw

        arr = as_chw(image, name="image")
        source_axes = self.query_axes if query_axes is None else query_axes
        target_axes = self.ref_axes if ref_axes is None else ref_axes
        # pulling an image one way samples it through the grid of the other
        grid = self._grid("backward" if direction == "forward" else "forward", source_axes, target_axes)
        return interp(source_axes if direction == "forward" else target_axes, arr, grid)


@dataclass(frozen=True, kw_only=True)
class StalignVolumeFit(StalignFit):
    """A fit from :func:`~squidpy.experimental.tl.stalign_align_volume`, at rank 3.

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

    def transform_points(self, points: Array) -> JaxArray:
        """Map ``(N, 2)`` ``(x, y)`` section points to ``(N, 3)`` ``(x, y, z)`` in the volume."""
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_points_row_col

        pts = jnp.asarray(points, dtype=jax_dtype())
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected an (N, 2) `(x, y)` array, found shape {pts.shape}.")
        # the volume is the solver's moving image, so section -> volume is its backward map
        lifted = jnp.stack((jnp.zeros(pts.shape[0], dtype=pts.dtype), pts[:, 1], pts[:, 0]), axis=1)
        transformed = transform_points_row_col(
            self.velocity_grid, self.velocity, self.affine, lifted, direction="backward"
        )
        return transformed[:, ::-1]

    def deformation_grid(self, *, query_axes: Sequence[Array] | None = None) -> JaxArray:
        """The section grid, lifted to ``z = 0``, evaluated in the volume: row-column, ``(3, 1, *grid)``.

        The same transform the objective uses. ``query_axes`` overrides the section axes the
        fit ran on.
        """
        import jax.numpy as jnp

        from ._stalign_impl._core import jax_dtype, transform_grid_row_col

        section_axes = tuple(self.query_axes if query_axes is None else query_axes)
        if len(section_axes) != 2:
            raise ValueError(f"Expected 2 query axes, found {len(section_axes)}.")
        lifted = (jnp.zeros(1, dtype=jax_dtype()), *section_axes)
        return transform_grid_row_col(lifted, self.velocity_grid, self.velocity, self.affine, direction="backward")


def fit_stalign_volume(
    ref: Array,
    query: Array,
    *,
    ref_axes: Sequence[Array] | None = None,
    query_axes: Sequence[Array] | None = None,
    initial_slice: int | None = None,
    initial_rotation: float = 0.0,
    initial_scale: float = 1.0,
    initial_affine: Array | None = None,
    **solver_params: Unpack[StalignVolumeParams],
) -> StalignVolumeFit:
    """Array-in / array-out core of :func:`~squidpy.experimental.tl.stalign_align_volume`, which documents it."""
    _require_jax()

    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype, lddmm
    from ._stalign_impl._helpers import affine_xy_to_rc, as_chw, resolve_axes

    opts = resolve_params(solver_params, StalignVolumeParams, arg_name="solver_params")
    dtype = jax_dtype()

    target_image = as_chw(query, name="query", ndim=2)
    source_image = as_chw(ref, name="ref", ndim=3)
    if source_image.shape[1] < 2:
        raise ValueError(
            f"Expected `ref` to be a volume with at least two samples along `z`, found depth "
            f"{source_image.shape[1]}. A single plane carries no out-of-plane information; use "
            f"`stalign_align_image` to register two 2D images."
        )

    # the volume is the moving image (LDDMM's I/xI), the section the fixed one (J/xJ)
    source_grid = resolve_axes(ref_axes, source_image.shape[1:], "ref_axes")
    section_grid = resolve_axes(query_axes, target_image.shape[1:], "query_axes")
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
        # send the volume's centre at `initial_slice` to the section's centre
        centre_volume = jnp.asarray(
            [source_grid[0][slice_index], jnp.mean(source_grid[1]), jnp.mean(source_grid[2])],
            dtype=dtype,
        )
        centre_section = jnp.asarray(
            [target_grid[0][0], jnp.mean(target_grid[1]), jnp.mean(target_grid[2])],
            dtype=dtype,
        )
        translation = centre_section - linear @ centre_volume

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
        **fit_result,
        ref_axes=(source_grid[0], source_grid[1], source_grid[2]),
        query_axes=(section_grid[0], section_grid[1]),
    )


def _require_jax() -> None:
    """Fail with an actionable message rather than a bare ImportError on the optional extra."""
    try:
        import jax  # noqa: F401
    except ImportError as e:
        raise ImportError(_JAX_REQUIRED) from e


def _initial_affine_and_landmarks(
    landmarks_ref: Array | None,
    landmarks_query: Array | None,
    initial_affine: Array | None,
) -> tuple[JaxArray, JaxArray, JaxArray | None, JaxArray | None]:
    """Resolve the starting affine and the row-col landmark pair the point term uses.

    Landmarks are ``(x, y)``, matched by row order, and in the same units as the fit's
    coordinates: cell coordinates for a point-cloud fit, the images' physical axes for an
    image fit.

    The two initialisers are not exclusive. Landmarks play two roles: they always
    contribute the point-matching term the solver weights by ``sigmaP``, and they *also*
    derive the starting affine when ``initial_affine`` is absent. Passing both is how you
    keep the matching term while pinning the start yourself.

    The derived start is translation-only for fewer than 3 pairs, and refuses collinear
    landmarks (a line leaves the affine undetermined); ``initial_affine`` bypasses both.
    """
    import jax.numpy as jnp

    from ._stalign_impl._core import jax_dtype
    from ._stalign_impl._helpers import affine_from_points, affine_xy_to_rc, validate_points

    if (landmarks_ref is None) != (landmarks_query is None):
        raise ValueError("Expected both landmark arrays to be provided together.")

    dtype = jax_dtype()
    source_landmarks = target_landmarks = None
    if landmarks_ref is not None:
        # landmarks arrive as (x, y); the solver runs in row-col
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
    ref: Array,
    query: Array,
    *,
    landmarks_ref: Array | None = None,
    landmarks_query: Array | None = None,
    **solver_params: Unpack[StalignObsParams],
) -> StalignObsFit:
    """Array-in / array-out core of :func:`~squidpy.experimental.tl.stalign_align_obs`, which documents it."""
    _require_jax()

    from ._stalign_impl._core import lddmm, transform_points_row_col
    from ._stalign_impl._helpers import rasterize_cloud, validate_points

    opts = resolve_params(solver_params, StalignObsParams, arg_name="solver_params")
    linear, translation, src_lm, tgt_lm = _initial_affine_and_landmarks(
        landmarks_ref, landmarks_query, opts.get("initial_affine")
    )

    # The solver runs internally in row-col (y, x); inputs are (x, y): swap at the boundary.
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
        fit_result["velocity_grid"], fit_result["velocity"], fit_result["affine"], source_rc, direction="forward"
    )
    return StalignObsFit(
        **fit_result,
        aligned_points=aligned_rc[:, ::-1],
    )


def fit_stalign_image(
    ref: Array,
    query: Array,
    *,
    ref_axes: Sequence[Array] | None = None,
    query_axes: Sequence[Array] | None = None,
    landmarks_ref: Array | None = None,
    landmarks_query: Array | None = None,
    **solver_params: Unpack[StalignImageParams],
) -> StalignImageFit:
    """Array-in / array-out core of :func:`~squidpy.experimental.tl.stalign_align_image`, which documents it."""
    _require_jax()

    from ._stalign_impl._core import lddmm
    from ._stalign_impl._helpers import as_chw, resolve_axes

    opts = resolve_params(solver_params, StalignImageParams, arg_name="solver_params")
    linear, translation, src_lm, tgt_lm = _initial_affine_and_landmarks(
        landmarks_ref, landmarks_query, opts.get("initial_affine")
    )

    source_image = as_chw(query, name="query")
    target_image = as_chw(ref, name="ref")

    source_grid = resolve_axes(query_axes, source_image.shape[1:], "query_axes")
    target_grid = resolve_axes(ref_axes, target_image.shape[1:], "ref_axes")

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
        **fit_result,
        query_axes=(source_grid[0], source_grid[1]),
        ref_axes=(target_grid[0], target_grid[1]),
    )
