"""The public alignment functions, built on the array-in / array-out estimators.

Thin orchestrators: resolve the ``*_key`` arguments to in-memory arrays and call the
estimator. The estimators themselves never see a container; SpatialData transformation
write-back lives in :mod:`._io`.

Fitting and writing are separate calls for STalign. A diffeomorphism has no SpatialData
representation, so the fit cannot live in a container: it is the return value, and
:func:`stalign_apply_transform` and :func:`stalign_apply_warp` take one and write. Keeping
it also keeps what only a fit can do -- warp an image, evaluate the dense deformation, map
points the fit never saw. :func:`stalign_store_fit` puts one in ``uns`` when it is worth
keeping, and the ``stalign_apply_*`` functions accept that key in place of the fit.
:func:`align_landmarks` fits and writes in one call, which its result being an affine, and
so representable, makes honest.

The ``stalign_apply_*`` functions take ``inplace``, with the meaning scanpy gives it:
``inplace=False`` hands back what would have been written instead of writing it. A
function that returns a fit takes no such flag -- there is nothing to write yet, and
``copy`` in scanpy's sense (operate on a duplicated container) is a caller's ``.copy()``
away.

``key_added`` always names a write target, defaulting to a conventional key the way
scanpy's does -- it is never the switch for whether to write. That is ``inplace``'s job,
and one flag with one meaning beats two spellings of the same thing.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Literal, Unpack

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from ._io import fit_from_uns, fit_to_uns, writeback_affine_sdata
from ._landmark import apply_affine, fit_affine, fit_similarity
from ._stalign import (
    StalignImageSolverKwargs,
    StalignObsSolverKwargs,
    StalignVolumeSolverKwargs,
    fit_stalign_image,
    fit_stalign_obs,
    fit_stalign_volume,
    stalign_transform_points,
    stalign_warp_image,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ._stalign import Stalign2DResult, Stalign3DResult, StalignResult

__all__ = [
    "align_landmarks",
    "stalign_align_image",
    "stalign_align_obs",
    "stalign_align_volume",
    "stalign_apply_transform",
    "stalign_apply_warp",
    "stalign_store_fit",
]


def _resolve_pair(value: str | tuple[str | None, str | None], *, name: str) -> tuple[str | None, str | None]:
    """Normalise a single key or a ``(ref, query)`` pair to a pair."""
    if isinstance(value, str):
        return value, value
    if isinstance(value, tuple | list) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(f"`{name}` must be a single key or a `(ref, query)` pair, got {value!r}.")


def _resolve_table(container: AnnData | SpatialData, table_key: str | None, *, side: str) -> AnnData:
    """Resolve a container plus ``table_key`` to the AnnData holding the data."""
    if isinstance(container, AnnData):
        if table_key is not None:
            raise ValueError(
                f"`table_key` was given for the {side}, but the {side} is an AnnData, which has no tables. "
                f"Pass `table_key=None` for it (use a `(ref, query)` pair to address mixed containers)."
            )
        return container
    if not isinstance(container, SpatialData):
        raise TypeError(f"Expected the {side} to be an AnnData or SpatialData, got {type(container).__name__}.")
    if table_key is None:
        raise ValueError(
            f"The {side} is a SpatialData, which may hold several tables; pass `table_key` to say "
            f"which one holds the data. Available: {sorted(container.tables)}."
        )
    if table_key not in container.tables:
        raise KeyError(
            f"`table_key={table_key!r}`: no such table in the {side}. Available: {sorted(container.tables)}."
        )
    return container.tables[table_key]


def _read_coords(adata: AnnData, key: str, *, side: str, name: str) -> np.ndarray:
    """Read a validated ``(N, 2)`` coordinate array from ``obsm``."""
    if key not in adata.obsm:
        raise KeyError(f"`{name}={key!r}`: no `obsm[{key!r}]` on the {side}. Available: {sorted(adata.obsm)}.")
    coords = np.asarray(adata.obsm[key])
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"`{name}={key!r}` on the {side} must be an (N, 2) array, found shape {coords.shape}.")
    return coords


def _as_chw(value: npt.ArrayLike, *, what: str, ndim: int = 2) -> np.ndarray:
    """Promote an unchannelled array to a single-channel one.

    ``ndim`` is the number of *spatial* axes: 2 for a section, 3 for a reference volume.
    """
    array = np.asarray(value)
    if array.ndim == ndim:
        array = array[None]
    if array.ndim != ndim + 1:
        spatial = ", ".join("zyx"[-ndim:])
        raise ValueError(f"{what} must be a ({spatial}) or (c, {spatial}) image, found shape {array.shape}.")
    return array


def _read_image(container: SpatialData, key: str, *, side: str, ndim: int = 2) -> np.ndarray:
    """Read a channels-first array from a SpatialData image element."""
    if not isinstance(container, SpatialData):
        raise TypeError(f"`image_key` names a SpatialData image, but the {side} is a {type(container).__name__}.")
    if key not in container.images:
        raise KeyError(f"`image_key={key!r}`: no such image in the {side}. Available: {sorted(container.images)}.")

    from squidpy.experimental.im._utils import get_element_data

    element = get_element_data(container.images[key], "scale0", "image", key)
    return _as_chw(element.data, what=f"`image_key={key!r}`", ndim=ndim)


def _write_coords(
    container: AnnData | SpatialData,
    table_key: str | None,
    spatial_key: str,
    key_added: str,
    *,
    transform: Callable[[np.ndarray], npt.ArrayLike],
    spatial_key_name: str,
    inplace: bool = True,
) -> np.ndarray | None:
    """Transform ``obsm[spatial_key]`` and write it to ``obsm[key_added]`` on the query."""
    adata = _resolve_table(container, table_key, side="query")
    coords = _read_coords(adata, spatial_key, side="query", name=spatial_key_name)
    transformed = np.asarray(transform(coords))
    if not inplace:
        return transformed
    adata.obsm[key_added] = transformed
    return None


def _query_of(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    *,
    ref_address: tuple[str | None, ...],
    query_address: tuple[str | None, ...],
    key_name: str,
) -> AnnData | SpatialData:
    """The container holding the query sample, validating the single-SpatialData form."""
    if data_query is not None:
        return data_query
    if not isinstance(data_ref, SpatialData):
        raise ValueError("`data_query` is required unless `data_ref` is a SpatialData holding both samples.")
    if ref_address == query_address:
        raise ValueError(
            f"With a single SpatialData, pass `{key_name}=(ref, query)` to say which of its elements holds each sample."
        )
    return data_ref


def stalign_align_obs(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    spatial_key: str | tuple[str, str] = "spatial",
    table_key: str | tuple[str | None, str | None] | None = None,
    landmarks_ref: npt.ArrayLike | None = None,
    landmarks_query: npt.ArrayLike | None = None,
    **solver_kwargs: Unpack[StalignObsSolverKwargs],
) -> Stalign2DResult:
    """Align a query point cloud onto a reference with STalign (diffeomorphic LDDMM).

    Fits and returns; nothing is written. Pass the fit to
    :func:`~squidpy.experimental.tl.stalign_apply_transform` to write transformed
    coordinates into a container.

    Parameters
    ----------
    data_ref, data_query
        The reference and query samples, each an :class:`~anndata.AnnData` or a
        :class:`~spatialdata.SpatialData`. Leave ``data_query=None`` with ``data_ref``
        a SpatialData holding both samples, distinguished by a ``table_key`` pair.
    spatial_key
        ``obsm`` key holding the ``(N, 2)`` spatial coordinates, or a ``(ref, query)``
        pair when the two samples key their coordinates differently.
    table_key
        For SpatialData input, which table holds the coordinates. A single key applies
        to both sides; a ``(ref, query)`` pair addresses each side separately (entries
        may be ``None`` for an AnnData side).
    landmarks_ref, landmarks_query
        Optional paired ``(x, y)`` landmark arrays (matched by row order) used to
        initialise the affine.
    solver_kwargs
        LDDMM solver tuning; see
        :class:`~squidpy.experimental.tl.StalignObsSolverKwargs` for the accepted
        keys, their meaning, and their defaults.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.Stalign2DResult`.
    """
    ref_spatial, query_spatial = _resolve_pair(spatial_key, name="spatial_key")
    ref_table, query_table = (None, None) if table_key is None else _resolve_pair(table_key, name="table_key")
    query_container = _query_of(
        data_ref,
        data_query,
        ref_address=(ref_table, ref_spatial),
        query_address=(query_table, query_spatial),
        key_name="table_key",
    )

    ref_adata = _resolve_table(data_ref, ref_table, side="reference")
    query_adata = _resolve_table(query_container, query_table, side="query")

    return fit_stalign_obs(
        ref=_read_coords(ref_adata, ref_spatial, side="reference", name="spatial_key"),
        query=_read_coords(query_adata, query_spatial, side="query", name="spatial_key"),
        landmarks_ref=landmarks_ref,
        landmarks_query=landmarks_query,
        **solver_kwargs,
    )


def stalign_align_image(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    *,
    image_key: str | tuple[str, str],
    ref_coordinate_system: str = "global",
    query_coordinate_system: str = "global",
    landmarks_ref: npt.ArrayLike | None = None,
    landmarks_query: npt.ArrayLike | None = None,
    **solver_kwargs: Unpack[StalignImageSolverKwargs],
) -> Stalign2DResult:
    """Align a query image onto a reference image with STalign (diffeomorphic LDDMM).

    Fits and returns; nothing is written. Pass the fit to
    :func:`~squidpy.experimental.tl.stalign_apply_warp` to materialise the aligned image,
    or to :func:`~squidpy.experimental.tl.stalign_apply_transform` to place a table's
    cells in the reference frame.

    Parameters
    ----------
    sdata_ref, sdata_query
        The :class:`~spatialdata.SpatialData` objects holding the reference and query
        images. Leave ``sdata_query=None`` with ``sdata_ref`` holding both images,
        distinguished by an ``image_key`` pair.
    image_key
        Name of the image element, or a ``(ref, query)`` pair.
    ref_coordinate_system, query_coordinate_system
        Coordinate systems to read each element's physical axes in. The scale and
        translation the elements carry supply the units, so two images at different
        resolutions need nothing restated -- and nothing can be restated to contradict
        the container.
    landmarks_ref, landmarks_query
        Optional paired ``(x, y)`` landmark arrays (matched by row order), in the units of
        the corresponding ``*_coordinate_system`` -- the same units the elements' own
        transformations supply, not pixel indices. They contribute the point-matching term
        the solver weights by ``sigmaP``, and derive the starting affine unless
        ``initial_affine`` is given, in which case that wins and the matching term stays.
    solver_kwargs
        LDDMM solver tuning; see
        :class:`~squidpy.experimental.tl.StalignImageSolverKwargs` for the accepted
        keys, their meaning, and their defaults. ``a``, ``epL``, ``epT`` and ``epV`` are
        lengths and step sizes in the units the elements carry, and their defaults are
        tuned for pixel-sized units -- an element scaled to microns needs them rescaled.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.Stalign2DResult`.
    """
    ref_image, query_image = _resolve_pair(image_key, name="image_key")
    query_container = _query_of(
        sdata_ref, sdata_query, ref_address=(ref_image,), query_address=(query_image,), key_name="image_key"
    )

    ref_array = _read_image(sdata_ref, ref_image, side="reference")
    query_array = _read_image(query_container, query_image, side="query")
    return fit_stalign_image(
        ref=ref_array,
        query=query_array,
        ref_axes=_element_axes(
            sdata_ref, ref_image, ref_array, coordinate_system=ref_coordinate_system, side="reference"
        ),
        query_axes=_element_axes(
            query_container, query_image, query_array, coordinate_system=query_coordinate_system, side="query"
        ),
        landmarks_ref=landmarks_ref,
        landmarks_query=landmarks_query,
        **solver_kwargs,
    )


def _element_axes(
    container: SpatialData,
    key: str,
    array: np.ndarray,
    *,
    coordinate_system: str,
    side: str,
) -> list[np.ndarray]:
    """Physical axes of an image element, read off its transformation.

    The scale and translation the element carries into ``coordinate_system`` are what put
    it in physical units, so the fit reads them rather than taking a ``*_scale`` argument
    that could disagree with the container. Axis order matches the array's spatial axes.
    """
    from spatialdata.transformations import get_transformation

    spatial = ("z", "y", "x")[-(array.ndim - 1) :]
    transformation = get_transformation(container.images[key], to_coordinate_system=coordinate_system)
    matrix = np.asarray(transformation.to_affine_matrix(input_axes=spatial, output_axes=spatial), dtype=float)

    off_diagonal = matrix[: len(spatial), : len(spatial)] - np.diag(np.diag(matrix[: len(spatial), : len(spatial)]))
    if np.any(np.abs(off_diagonal) > 1e-9):
        raise ValueError(
            f"`image_key={key!r}` on the {side} carries a rotation or shear into "
            f"{coordinate_system!r}, which cannot be expressed as per-axis coordinates. "
            f"Pass the axes explicitly, or move the rotation into the fit's initialisation."
        )
    scale = np.diag(matrix[: len(spatial), : len(spatial)])
    offset = matrix[: len(spatial), -1]
    return [
        np.arange(size, dtype=float) * step + shift
        for size, step, shift in zip(array.shape[1:], scale, offset, strict=True)
    ]


def _assert_table_coords_share_frame(
    sdata: SpatialData, adata: AnnData, spatial_key: str, *, coordinate_system: str
) -> None:
    """Refuse to transform ``obsm`` coordinates that are not in ``coordinate_system``.

    The fit's units come from the image element's transformation, so the coordinates it is
    applied to have to be in that same system. A table's ``obsm`` sits in the intrinsic
    frame of the element it annotates, which only coincides when that element's transform
    into ``coordinate_system`` is the identity. Checked rather than silently applied: the
    result is plausible reference coordinates that are simply wrong, with nothing to
    reveal it.
    """
    from spatialdata.transformations import get_transformation

    region = adata.uns.get("spatialdata_attrs", {}).get("region")
    regions = [region] if isinstance(region, str) else list(region or [])
    for name in regions:
        if name not in sdata:
            continue
        matrix = np.asarray(
            get_transformation(sdata[name], to_coordinate_system=coordinate_system).to_affine_matrix(
                input_axes=("x", "y"), output_axes=("x", "y")
            ),
            dtype=float,
        )
        if not np.allclose(matrix, np.eye(3)):
            raise ValueError(
                f"`spatial_key={spatial_key!r}` lives in the intrinsic frame of element {name!r}, which "
                f"carries a non-identity transformation into {coordinate_system!r} -- the coordinate system "
                f"the fit's units come from. Transforming it would silently produce wrong reference "
                f"coordinates. Store the coordinates in {coordinate_system!r} units, or apply "
                f"`stalign_transform_points` to coordinates you have placed there yourself."
            )


def stalign_align_volume(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    *,
    image_key: str | tuple[str, str],
    ref_coordinate_system: str = "global",
    query_coordinate_system: str = "global",
    initial_slice: int | None = None,
    initial_rotation: float = 0.0,
    initial_scale: float = 1.0,
    initial_affine: npt.ArrayLike | None = None,
    **solver_kwargs: Unpack[StalignVolumeSolverKwargs],
) -> Stalign3DResult:
    """Place a 2D section into a 3D reference volume with STalign (diffeomorphic LDDMM).

    The plane of a physical section is unknown and generally not exactly coronal, so this
    fits the full 3D deformation rather than an oblique plane plus an in-plane 2D fit;
    ``initial_slice`` / ``initial_rotation`` / ``initial_scale`` are an initialisation, not
    the answer.

    Fits and returns; nothing is written. Pass the fit to
    :func:`~squidpy.experimental.tl.stalign_apply_transform` to give every cell real
    ``(x, y, z)`` reference coordinates, or to
    :func:`~squidpy.experimental.tl.stalign_transform_points` and
    :func:`~squidpy.experimental.im.sample_volume` to read the reference volume where the
    cells land.

    Parameters
    ----------
    sdata_ref, sdata_query
        The :class:`~spatialdata.SpatialData` objects holding the reference volume and the
        section. Leave ``sdata_query=None`` with ``sdata_ref`` holding both, distinguished
        by an ``image_key`` pair.
    image_key
        Name of the image element, or a ``(ref, query)`` pair. The reference must be a
        ``(c, z, y, x)`` volume and the query a ``(c, y, x)`` section -- for cells rather
        than an image, rasterize them first with
        :func:`~squidpy.experimental.im.rasterize_points`.
    ref_coordinate_system, query_coordinate_system
        Coordinate systems to read each element's physical axes in. The scale and
        translation the elements carry supply the units, so nothing has to be restated.
    initial_slice
        Index along the reference's ``z`` axis to centre the section on. ``None`` centres
        on the middle of the volume.
    initial_rotation
        In-plane rotation of the initial affine, in **radians**.
    initial_scale
        Uniform scale of the initial affine. A little under 1 is a common start.
    initial_affine
        Homogeneous ``(4, 4)`` affine in ``(x, y, z)``, replacing the three ``initial_*``
        arguments above. The escape hatch when the initialisation needs to be exact.
    solver_kwargs
        LDDMM solver tuning; see
        :class:`~squidpy.experimental.tl.StalignVolumeSolverKwargs`.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.Stalign3DResult`.
    """
    ref_image, query_image = _resolve_pair(image_key, name="image_key")
    query_container = _query_of(
        sdata_ref, sdata_query, ref_address=(ref_image,), query_address=(query_image,), key_name="image_key"
    )

    ref_array = _read_image(sdata_ref, ref_image, side="reference", ndim=3)
    query_array = _read_image(query_container, query_image, side="query")

    return fit_stalign_volume(
        ref=ref_array,
        query=query_array,
        ref_axes=_element_axes(
            sdata_ref, ref_image, ref_array, coordinate_system=ref_coordinate_system, side="reference"
        ),
        query_axes=_element_axes(
            query_container, query_image, query_array, coordinate_system=query_coordinate_system, side="query"
        ),
        initial_slice=initial_slice,
        initial_rotation=initial_rotation,
        initial_scale=initial_scale,
        initial_affine=initial_affine,
        **solver_kwargs,
    )


def stalign_apply_transform(
    fit_result: StalignResult | str,
    data: AnnData | SpatialData,
    *,
    key_added: str = "spatial_aligned",
    spatial_key: str = "spatial",
    table_key: str | None = None,
    coordinate_system: str = "global",
    inplace: bool = True,
) -> np.ndarray | None:
    """Write a fit's transformed coordinates into a container's ``obsm``.

    The container-level counterpart of
    :func:`~squidpy.experimental.tl.stalign_transform_points`: read ``obsm[spatial_key]``,
    map it through ``fit_result``, store it under ``obsm[key_added]``. A rank-2 fit writes
    ``(N, 2)`` coordinates in the reference frame; a rank-3 fit writes ``(N, 3)``
    ``(x, y, z)`` positions in the reference volume.

    Parameters
    ----------
    fit_result
        A fit from any of the ``stalign_align_*`` functions, or the ``uns`` key of one
        stored by :func:`~squidpy.experimental.tl.stalign_store_fit`.
    data
        The container to write into -- the query side the fit was given.
    key_added
        ``obsm`` key to write the transformed coordinates to, ``"spatial_aligned"`` by
        default.
    spatial_key
        ``obsm`` key holding the ``(N, 2)`` coordinates to transform.
    table_key
        For a :class:`~spatialdata.SpatialData`, which table holds them.
    coordinate_system
        The coordinate system the fit's units came from. Consulted only when ``fit_result`` was
        fitted on image elements, whose transformations supplied those units; a
        point-cloud fit's units are the coordinates' own and nothing can disagree.
    inplace
        ``True`` (default) writes ``obsm[key_added]`` and returns ``None``. ``False``
        leaves ``data`` untouched and returns the transformed coordinates, which saves
        digging ``spatial_key`` out of the container to pass to
        :func:`~squidpy.experimental.tl.stalign_transform_points` by hand.

    Returns
    -------
    ``None``, or the ``(N, 2)`` / ``(N, 3)`` transformed coordinates when
    ``inplace=False``.
    """
    if isinstance(fit_result, str):
        fit_result = fit_from_uns(_resolve_table(data, table_key, side="query"), fit_result)
    if isinstance(data, SpatialData) and "query_axes" in fit_result:
        # A fit carrying raster axes took its units from an image element's transformation,
        # so the coordinates it is applied to have to sit in that same frame.
        _assert_table_coords_share_frame(
            data,
            _resolve_table(data, table_key, side="query"),
            spatial_key,
            coordinate_system=coordinate_system,
        )
    return _write_coords(
        data,
        table_key,
        spatial_key,
        key_added,
        transform=functools.partial(stalign_transform_points, fit_result),
        spatial_key_name="spatial_key",
        inplace=inplace,
    )


def stalign_store_fit(
    fit_result: StalignResult,
    data: AnnData | SpatialData,
    *,
    key: str = "stalign",
    table_key: str | None = None,
) -> None:
    """Store a fit in a table's ``uns``, in a form that survives a write.

    A fit is what re-applies an alignment, and none of the other write paths keep it: a
    diffeomorphism is not a SpatialData transformation, so ``obsm`` coordinates and a warped
    element are both products of the fit rather than the fit itself. ``uns`` is the only
    place either container has for it.

    Not automatic on the ``stalign_align_*`` functions, because a rank-3 velocity field runs
    to hundreds of megabytes. Call this when the fit is worth keeping; the
    ``stalign_apply_*`` functions then take ``key`` in place of the fit.

    Parameters
    ----------
    fit_result
        A fit from any of the ``stalign_align_*`` functions.
    data
        The container to store it on.
    key
        ``uns`` key to write to, ``"stalign"`` by default.
    table_key
        For a :class:`~spatialdata.SpatialData`, which table's ``uns`` to use.
    """
    fit_to_uns(fit_result, _resolve_table(data, table_key, side="query"), key)


def stalign_apply_warp(
    fit_result: Stalign2DResult | str,
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    *,
    image_key: str | tuple[str, str],
    key_added: str | None = None,
    table_key: str | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """Materialise a fit's warped query image as a new element on the query.

    The container-level counterpart of
    :func:`~squidpy.experimental.tl.stalign_warp_image`. The fitted diffeomorphism cannot
    be expressed as a SpatialData transformation -- the available types are affine at most
    -- so the aligned image is written as a new element rather than registered: resampled
    onto the reference's pixel grid, and inheriting the reference element's placement.

    Parameters
    ----------
    fit_result
        A rank-2 fit from :func:`~squidpy.experimental.tl.stalign_align_image`, or the
        ``uns`` key of one stored by :func:`~squidpy.experimental.tl.stalign_store_fit`.
    sdata_ref, sdata_query
        The containers holding the reference and query images, as the fit was given them.
        The reference is what places the output element, so it is needed here too. Leave
        ``sdata_query=None`` when one container holds both, with an ``image_key`` pair.
    image_key
        Name of the image element, or the ``(ref, query)`` pair the fit ran on.
    key_added
        Image element name on the query to write the warped image under. ``None``
        (default) uses the query element's own name with an ``_aligned`` suffix -- a
        conventional key, not a request to skip the write, which is ``inplace=False``.
    table_key
        Which table's ``uns`` to read ``fit_result`` from, when it is a key.
    inplace
        ``True`` (default) writes the element and returns ``None``. ``False`` leaves
        ``sdata_query`` untouched and returns the warped ``(c, y, x)`` array.

    Returns
    -------
    ``None``, or the warped image when ``inplace=False``.
    """
    from spatialdata.models import Image2DModel
    from spatialdata.transformations import get_transformation

    ref_image, query_image = _resolve_pair(image_key, name="image_key")
    query_container = _query_of(
        sdata_ref, sdata_query, ref_address=(ref_image,), query_address=(query_image,), key_name="image_key"
    )
    if isinstance(fit_result, str):
        stored = fit_from_uns(_resolve_table(query_container, table_key, side="query"), fit_result)
        if stored["rank"] != 2:
            raise ValueError(
                f"`stalign_apply_warp` needs a rank-2 fit, `uns[{fit_result!r}]` holds rank {stored['rank']}."
            )
        fit_result = stored
    query_array = _read_image(query_container, query_image, side="query")
    added = f"{query_image}_aligned" if key_added is None else key_added
    # `np.asarray` because `stalign_warp_image` returns a JAX array, which the parser rejects.
    warped = np.asarray(stalign_warp_image(fit_result, query_array))
    if not inplace:
        return warped
    # A forward warp resamples the query onto the *reference's* grid, so the new element
    # is placed by the reference's own transformations -- reconstructing a scale and
    # translation from the fitted axes would be the same numbers, spelled less reliably.
    query_container.images[added] = Image2DModel.parse(
        warped,
        dims=("c", "y", "x"),
        transformations=dict(get_transformation(sdata_ref.images[ref_image], get_all=True)),
    )
    return None


def align_landmarks(
    data_ref: AnnData | SpatialData | npt.ArrayLike,
    data_query: AnnData | SpatialData | npt.ArrayLike | None = None,
    *,
    landmark_key: str | tuple[str, str] | None = None,
    fit: Literal["similarity", "affine"] = "similarity",
    table_key: str | tuple[str | None, str | None] | None = None,
    spatial_key: str | None = None,
    key_added: str | None = None,
    target_coordinate_system: str | None = None,
) -> npt.NDArray[np.float64] | None:
    """Align a query sample onto a reference from paired landmarks (closed-form affine).

    Parameters
    ----------
    data_ref, data_query
        The reference and query samples, each an :class:`~anndata.AnnData` or a
        :class:`~spatialdata.SpatialData`. Leave ``data_query=None`` with ``data_ref``
        a SpatialData holding both samples' landmarks, distinguished by a
        ``landmark_key`` pair.

        Both may instead be the ``(N, 2)`` landmark arrays themselves, matched by row order --
        the same form ``landmarks_ref`` / ``landmarks_query`` take on
        :func:`~squidpy.experimental.tl.stalign_align_obs`. Landmarks are correspondences
        *between* two samples rather than observations *of* one, so they do not always have a
        container to live in: thirteen of them cannot sit in the ``obsm`` of a sample with
        eighty thousand cells. Given arrays, this returns the affine, and every argument that
        addresses a container must be left unset.
    landmark_key
        Where the ``(N, 2)`` landmark correspondences live (matched by row order), or a
        ``(ref, query)`` pair. Required for container input, rejected for arrays. On an AnnData -- or a SpatialData with ``table_key`` --
        this is an ``obsm`` key; on a SpatialData without ``table_key`` it names a
        shapes element, the layout napari-spatialdata writes when landmarks are picked
        interactively.
    fit
        ``"similarity"`` (default) fits 4 degrees of freedom (rotation + uniform scale
        + translation); ``"affine"`` fits all 6 (adding non-uniform scale and shear).
        The more constrained fit cannot shear a sample that should not be sheared.
    table_key
        For SpatialData input, read the landmarks from this table's ``obsm`` instead
        of a shapes element. A single key applies to both sides; a ``(ref, query)``
        pair addresses each side separately.
    spatial_key
        ``obsm`` key of the query array to transform when ``key_added`` is given. The
        landmarks are correspondences rather than the data, so what moves has to be
        named explicitly.
    key_added
        ``obsm`` key on the query to write the transformed ``spatial_key`` coordinates
        to. Mutually exclusive with ``target_coordinate_system``.
    target_coordinate_system
        Register the fitted affine as a SpatialData transformation into this
        coordinate system instead of materialising anything: every element registered
        to the query's coordinate system inherits the alignment. Requires the
        landmarks to come from shapes elements, and refuses when the reference sits in
        the same coordinate system of the same object (it would be dragged along).

    Returns
    -------
    The fitted homogeneous ``(3, 3)`` affine in ``(x, y)`` when neither ``key_added``
    nor ``target_coordinate_system`` is given -- directly usable as a
    :class:`~spatialdata.transformations.Affine`; otherwise ``None``, having written into
    the query container itself. Copy it first if the original must survive.
    """
    if fit not in {"similarity", "affine"}:
        raise ValueError(f"Unknown `fit={fit!r}`. Expected one of affine, similarity.")
    fit_fn = fit_similarity if fit == "similarity" else fit_affine
    if key_added is not None and target_coordinate_system is not None:
        raise ValueError(
            "`key_added` and `target_coordinate_system` are mutually exclusive: the first materialises "
            "transformed coordinates, the second registers the fit as a transformation."
        )
    if spatial_key is not None and key_added is None:
        raise ValueError("`spatial_key` says what `key_added` transforms, so it needs `key_added` to be set.")

    if not isinstance(data_ref, AnnData | SpatialData):
        # The landmarks themselves, not containers holding them. There is no key to address
        # and nothing to write into, so this returns the matrix and refuses the arguments that
        # only mean something for a container rather than silently ignoring them.
        # `fit_similarity` / `fit_affine` validate the pair, so nothing is re-checked here.
        if data_query is None:
            raise ValueError(
                "`data_ref` is an array of landmarks, so `data_query` must be the matching "
                "array of query landmarks rather than `None`."
            )
        for name, value in (
            ("landmark_key", landmark_key),
            ("table_key", table_key),
            ("spatial_key", spatial_key),
            ("key_added", key_added),
            ("target_coordinate_system", target_coordinate_system),
        ):
            if value is not None:
                raise ValueError(f"`{name}` addresses a container, and landmark arrays have none.")
        return fit_fn(data_ref, data_query)

    if landmark_key is None:
        raise ValueError(
            "`landmark_key` says where the landmarks live on a container. Pass the landmark "
            "arrays directly as `data_ref` and `data_query` if they are not stored on one."
        )

    ref_lm_key, query_lm_key = _resolve_pair(landmark_key, name="landmark_key")
    ref_table, query_table = (None, None) if table_key is None else _resolve_pair(table_key, name="table_key")
    query_container = _query_of(
        data_ref,
        data_query,
        ref_address=(ref_table, ref_lm_key),
        query_address=(query_table, query_lm_key),
        key_name="landmark_key",
    )

    ref_lm = _read_landmarks(data_ref, ref_lm_key, ref_table, side="reference")
    query_lm = _read_landmarks(query_container, query_lm_key, query_table, side="query")

    if target_coordinate_system is not None:
        return _register_transformation(
            fit_fn,
            ref_lm,
            query_lm,
            data_ref=data_ref,
            query_container=query_container,
            ref_lm_key=ref_lm_key,
            query_lm_key=query_lm_key,
            ref_table=ref_table,
            query_table=query_table,
            target_coordinate_system=target_coordinate_system,
        )

    matrix = fit_fn(ref_lm, query_lm)
    if key_added is None:
        return matrix
    if spatial_key is None:
        raise ValueError(
            "`key_added` needs `spatial_key` when aligning by landmarks: the landmarks are "
            "correspondences, so they do not say which array to transform. Pass e.g. "
            '`spatial_key="spatial"`, or use `target_coordinate_system=...` on a SpatialData to '
            "move every element in the query's coordinate system at once."
        )
    return _write_coords(
        query_container,
        query_table,
        spatial_key,
        key_added,
        transform=functools.partial(apply_affine, matrix),
        spatial_key_name="spatial_key",
    )


def _read_landmarks(
    container: AnnData | SpatialData,
    landmark_key: str | None,
    table_key: str | None,
    *,
    side: str,
) -> np.ndarray:
    """Read ``(N, 2)`` ``(x, y)`` landmarks from an ``obsm`` key or a shapes element."""
    if isinstance(container, AnnData) or table_key is not None:
        adata = _resolve_table(container, table_key, side=side)
        return _read_coords(adata, landmark_key, side=side, name="landmark_key")
    if not isinstance(container, SpatialData):
        raise TypeError(f"Expected the {side} to be an AnnData or SpatialData, got {type(container).__name__}.")
    if landmark_key not in container.shapes:
        raise KeyError(
            f"`landmark_key={landmark_key!r}`: no such shapes element in the {side}. "
            f"Available: {sorted(container.shapes)}. To read landmarks from a table's `obsm` "
            f"instead, pass `table_key`."
        )
    geometry = container.shapes[landmark_key].geometry
    return np.column_stack([geometry.x.to_numpy(), geometry.y.to_numpy()])


def _register_transformation(
    fit_fn: Callable[..., np.ndarray],
    ref_lm: np.ndarray,
    query_lm: np.ndarray,
    *,
    data_ref: AnnData | SpatialData,
    query_container: AnnData | SpatialData,
    ref_lm_key: str | None,
    query_lm_key: str | None,
    ref_table: str | None,
    query_table: str | None,
    target_coordinate_system: str,
) -> None:
    """Register an affine fit into a coordinate system instead of materialising it."""
    if not isinstance(query_container, SpatialData):
        raise TypeError("`target_coordinate_system` registers a transformation, which only a SpatialData has.")
    if query_table is not None:
        raise ValueError(
            "`target_coordinate_system` needs the query landmarks in a shapes element, but "
            "`table_key` reads them from a table, which has no coordinate system of its own. "
            "Store the landmarks as a shapes element, or write to `key_added` to move only the "
            "query's coordinates."
        )

    moving_cs = _coordinate_system_of(query_container, query_lm_key, side="query")
    # Registering moves *everything* in `moving_cs`. If the reference sits in that same
    # coordinate system of the same object, it would be dragged along with the query --
    # silently producing a wrong answer rather than failing.
    if (
        data_ref is query_container
        and ref_table is None
        and _coordinate_system_of(data_ref, ref_lm_key, side="reference") == moving_cs
    ):
        raise ValueError(
            f"The reference and query are both in coordinate system {moving_cs!r}, so registering "
            f"the fit there would move the reference too. Put each sample in its own coordinate "
            f"system (what napari-spatialdata does when landmarks are picked per sample), or "
            f"write to `key_added` to move only the query."
        )

    writeback_affine_sdata(
        fit_fn(ref_lm, query_lm),
        query_container,
        moving_cs=moving_cs,
        target_cs=target_coordinate_system,
    )
    return None


def _coordinate_system_of(sdata: SpatialData, element: str, *, side: str) -> str:
    """The coordinate system the shapes element is annotated in.

    Everything registered to it moves with the fit, so it has to be unambiguous.
    Reading it off the element rather than taking it as an argument keeps the call site
    to the ``*_key`` arguments, and it is the same element the user picked the
    landmarks on.
    """
    from spatialdata.transformations import get_transformation

    systems = sorted(get_transformation(sdata.shapes[element], get_all=True))
    if len(systems) != 1:
        raise ValueError(
            f"`landmark_key={element!r}` on the {side} is registered to {len(systems)} coordinate "
            f"systems ({', '.join(systems)}), so which one the alignment should move is ambiguous. "
            f"Register it to exactly one."
        )
    return systems[0]
