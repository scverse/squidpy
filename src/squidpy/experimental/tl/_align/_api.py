"""The public alignment functions, built on the array-in / array-out estimators."""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING, Any, Literal, Unpack

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from squidpy._validators import normalize_choice
from squidpy.types import StalignImageParams, StalignObsParams, StalignVolumeParams

from ._io import read_coords, resolve_table, write_coords, writeback_affine_sdata
from ._landmark import apply_affine, fit_landmarks
from ._stalign import (
    StalignFit,
    StalignImageFit,
    StalignVolumeFit,
    fit_stalign_image,
    fit_stalign_obs,
    fit_stalign_volume,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    import xarray as xr
    from fast_array_utils.types import HasArrayNamespace as Array

    from ._stalign import StalignObsFit

__all__ = [
    "align_landmarks",
    "stalign_align_image",
    "stalign_align_obs",
    "stalign_align_volume",
]


def _resolve_pair(value: str | tuple[str | None, str | None], *, name: str) -> tuple[str | None, str | None]:
    """Normalise a single key or a ``(ref, query)`` pair to a pair."""
    if isinstance(value, str):
        return value, value
    if isinstance(value, tuple | list) and len(value) == 2:
        return value[0], value[1]
    raise ValueError(f"`{name}` must be a single key or a `(ref, query)` pair, got {value!r}.")


def _read_image(container: SpatialData, key: str, *, side: str, ndim: int, scale: str) -> xr.DataArray:
    """The image element's DataArray at ``scale``."""
    if not isinstance(container, SpatialData):
        raise TypeError(f"`image_key` names a SpatialData image, but the {side} is a {type(container).__name__}.")
    if key not in container.images:
        raise KeyError(f"`image_key={key!r}`: no such image in the {side}. Available: {sorted(container.images)}.")

    from squidpy.experimental.im._utils import get_element_data

    node = container.images[key]
    levels = list(node.keys()) if hasattr(node, "keys") else ["scale0"]
    if scale not in levels:
        raise KeyError(f"`scale={scale!r}`: no such level for `image_key={key!r}` on the {side}. Available: {levels}.")
    element = get_element_data(node, scale, "image", key)
    if ndim == 3 and "z" not in element.dims:
        raise ValueError(
            f"`image_key={key!r}` on the {side} is a 2D image, but this needs a volume with a `z` axis. "
            f"Use `stalign_align_image` to register two 2D images."
        )
    if ndim == 2 and "z" in element.dims:
        raise ValueError(f"`image_key={key!r}` on the {side} is a volume, but this needs a 2D image.")
    return element


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
    landmarks_ref: Array | None = None,
    landmarks_query: Array | None = None,
    **solver_params: Unpack[StalignObsParams],
) -> StalignObsFit:
    """Align a query point cloud onto a reference with STalign (diffeomorphic LDDMM).

    Fits and returns; nothing is written. Call
    :meth:`~squidpy.experimental.tl.StalignFit.transform` on the fit to write transformed
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
        Optional paired ``(x, y)`` landmark arrays (matched by row order). They contribute
        the point-matching term the solver weights by ``sigmaP``, and derive the starting
        affine unless ``initial_affine`` is given, in which case that wins and the matching
        term stays. The derived start is translation-only for fewer than 3 pairs, and 3 or
        more must not be collinear.
    solver_params
        LDDMM solver tuning; see
        :class:`~squidpy.types.StalignObsParams` for the accepted
        keys, their meaning, and their defaults.

    Returns
    -------
    The fit. Its ``aligned_points`` is ``query`` already mapped into the reference frame;
    it carries no raster axes, the point path having no image frame to keep.
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

    ref_adata = resolve_table(data_ref, ref_table, side="reference")
    query_adata = resolve_table(query_container, query_table, side="query")

    return fit_stalign_obs(
        ref=read_coords(ref_adata, key=ref_spatial, side="reference", name="spatial_key"),
        query=read_coords(query_adata, key=query_spatial, side="query", name="spatial_key"),
        landmarks_ref=landmarks_ref,
        landmarks_query=landmarks_query,
        **solver_params,
    )


def stalign_align_image(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    *,
    image_key: str | tuple[str, str],
    ref_coordinate_system: str = "global",
    query_coordinate_system: str = "global",
    scale: str | tuple[str, str] = "scale0",
    landmarks_ref: Array | None = None,
    landmarks_query: Array | None = None,
    **solver_params: Unpack[StalignImageParams],
) -> StalignImageFit:
    """Align a query image onto a reference image with STalign (diffeomorphic LDDMM).

    Fits and returns; nothing is written. Call
    :meth:`~squidpy.experimental.tl.StalignImageFit.warp_image` on the fit to resample the
    aligned image, or :meth:`~squidpy.experimental.tl.StalignFit.transform` to place a
    table's cells in the reference frame.

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
        resolutions need nothing restated: and nothing can be restated to contradict
        the container.
    scale
        Multiscale level to fit on, or a ``(ref, query)`` pair. A coarser level fits faster;
        its physical axes come from that level's own transformation, so the units stay right.
    landmarks_ref, landmarks_query
        Optional paired ``(x, y)`` landmark arrays (matched by row order), in the units of
        the corresponding ``*_coordinate_system``: the same units the elements' own
        transformations supply, not pixel indices. They contribute the point-matching term
        the solver weights by ``sigmaP``, and derive the starting affine unless
        ``initial_affine`` is given, in which case that wins and the matching term stays.
        The derived start is translation-only for fewer than 3 pairs, and 3 or more must
        not be collinear.
    solver_params
        LDDMM solver tuning; see
        :class:`~squidpy.types.StalignImageParams` for the accepted
        keys, their meaning, and their defaults.

    Returns
    -------
    The fit, carrying both images' physical axes, so warping and the dense deformation
    need no axes from the caller.
    """
    return _fit_raster(
        fit_stalign_image,
        sdata_ref,
        sdata_query,
        image_key,
        ref_coordinate_system=ref_coordinate_system,
        query_coordinate_system=query_coordinate_system,
        scale=scale,
        ref_ndim=2,
        landmarks_ref=landmarks_ref,
        landmarks_query=landmarks_query,
        **solver_params,
    )


def _fit_raster[F: StalignFit](
    fit_fn: Callable[..., F],
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None,
    image_key: str | tuple[str, str],
    *,
    ref_coordinate_system: str,
    query_coordinate_system: str,
    scale: str | tuple[str, str],
    ref_ndim: int,
    **fit_kwargs: Any,
) -> F:
    """Read both image elements, fit them on their physical axes and stamp the query frame on the fit."""
    ref_image, query_image = _resolve_pair(image_key, name="image_key")
    ref_scale, query_scale = _resolve_pair(scale, name="scale")
    query_container = _query_of(
        sdata_ref, sdata_query, ref_address=(ref_image,), query_address=(query_image,), key_name="image_key"
    )

    ref = _read_image(sdata_ref, ref_image, side="reference", ndim=ref_ndim, scale=ref_scale)
    query = _read_image(query_container, query_image, side="query", ndim=2, scale=query_scale)
    return dataclasses.replace(
        fit_fn(
            ref=np.asarray(ref.data),
            query=np.asarray(query.data),
            ref_axes=_element_axes(ref, ref_image, coordinate_system=ref_coordinate_system, side="reference"),
            query_axes=_element_axes(query, query_image, coordinate_system=query_coordinate_system, side="query"),
            **fit_kwargs,
        ),
        coordinate_system=query_coordinate_system,
    )


def _element_axes(element: xr.DataArray, key: str, *, coordinate_system: str, side: str) -> list[np.ndarray]:
    """Physical axes of one image level, read off its own transformation, in array axis order."""
    from spatialdata.transformations import get_transformation

    spatial = tuple(dim for dim in element.dims if dim in ("z", "y", "x"))
    transformation = get_transformation(element, to_coordinate_system=coordinate_system)
    matrix = np.asarray(transformation.to_affine_matrix(input_axes=spatial, output_axes=spatial), dtype=float)

    off_diagonal = matrix[: len(spatial), : len(spatial)] - np.diag(np.diag(matrix[: len(spatial), : len(spatial)]))
    if np.any(np.abs(off_diagonal) > 1e-9):
        raise ValueError(
            f"`image_key={key!r}` on the {side} carries a rotation or shear into "
            f"{coordinate_system!r}, which cannot be expressed as per-axis coordinates."
        )
    scale = np.diag(matrix[: len(spatial), : len(spatial)])
    offset = matrix[: len(spatial), -1]
    return [
        np.arange(element.sizes[dim], dtype=float) * step + shift
        for dim, step, shift in zip(spatial, scale, offset, strict=True)
    ]


def stalign_align_volume(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    *,
    image_key: str | tuple[str, str],
    ref_coordinate_system: str = "global",
    query_coordinate_system: str = "global",
    scale: str | tuple[str, str] = "scale0",
    initial_slice: int | None = None,
    initial_rotation: float = 0.0,
    initial_scale: float = 1.0,
    initial_affine: Array | None = None,
    **solver_params: Unpack[StalignVolumeParams],
) -> StalignVolumeFit:
    """Place a 2D section into a 3D reference volume with STalign (diffeomorphic LDDMM).

    The plane of a physical section is unknown and generally not exactly coronal, so this
    fits the full 3D deformation rather than an oblique plane plus an in-plane 2D fit;
    ``initial_slice`` / ``initial_rotation`` / ``initial_scale`` are an initialisation, not
    the answer.

    Fits and returns; nothing is written. Call
    :meth:`~squidpy.experimental.tl.StalignFit.transform` on the fit to give every cell real
    ``(x, y, z)`` reference coordinates, or pair
    :meth:`~squidpy.experimental.tl.StalignFit.transform_points` with
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
        ``(c, z, y, x)`` volume and the query a ``(c, y, x)`` section: for cells rather
        than an image, rasterize them first with
        :func:`~squidpy.experimental.im.rasterize_points`.
    ref_coordinate_system, query_coordinate_system
        Coordinate systems to read each element's physical axes in. The scale and
        translation the elements carry supply the units, so nothing has to be restated.
    scale
        Multiscale level to fit on, or a ``(ref, query)`` pair. A coarser level fits faster;
        its physical axes come from that level's own transformation, so the units stay right.
    initial_slice
        Index along the reference's ``z`` axis to centre the section on. ``None`` centres
        on the middle of the volume.
    initial_rotation
        In-plane rotation of the initial affine, in **radians**.
    initial_scale
        Uniform scale of the initial affine. A little under 1 is a common start.
    initial_affine
        Homogeneous ``(4, 4)`` affine in ``(x, y, z)``, replacing the three ``initial_*``
        arguments above, mapping query section points into the reference volume like
        :meth:`~squidpy.experimental.tl.StalignVolumeFit.transform_points` does. The escape
        hatch when the initialisation needs to be exact.
    solver_params
        LDDMM solver tuning; see
        :class:`~squidpy.types.StalignVolumeParams`.

    Returns
    -------
    The fit, carrying the reference volume's ``(z, y, x)`` axes and the section's ``(y, x)``.
    Maps section points into the volume; there is no image to warp at rank 3.
    """
    return _fit_raster(
        fit_stalign_volume,
        sdata_ref,
        sdata_query,
        image_key,
        ref_coordinate_system=ref_coordinate_system,
        query_coordinate_system=query_coordinate_system,
        scale=scale,
        ref_ndim=3,
        initial_slice=initial_slice,
        initial_rotation=initial_rotation,
        initial_scale=initial_scale,
        initial_affine=initial_affine,
        **solver_params,
    )


def align_landmarks(
    data_ref: AnnData | SpatialData | Array,
    data_query: AnnData | SpatialData | Array | None = None,
    *,
    landmark_key: str | tuple[str, str] | None = None,
    method: Literal["similarity", "affine"] = "similarity",
    table_key: str | tuple[str | None, str | None] | None = None,
    spatial_key: str = "spatial",
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

        Both may instead be the ``(N, 2)`` landmark arrays themselves, matched by row order:
        the same form ``landmarks_ref`` / ``landmarks_query`` take on
        :func:`~squidpy.experimental.tl.stalign_align_obs`. Given arrays, this returns the
        affine, and every argument that addresses a container must be left unset.
    landmark_key
        Where the ``(N, 2)`` landmark correspondences live (matched by row order), or a
        ``(ref, query)`` pair. Required for container input, rejected for arrays. On an AnnData (or a SpatialData with ``table_key``)
        this is an ``obsm`` key; on a SpatialData without ``table_key`` it names a
        shapes element, the layout napari-spatialdata writes when landmarks are picked
        interactively.
    method
        ``"similarity"`` (default) fits 4 degrees of freedom (rotation + uniform scale
        + translation, never a reflection) from 2 or more distinct landmarks, a line
        included; ``"affine"`` fits all 6 (adding non-uniform scale and shear) and needs 3
        or more that are not collinear. The more constrained fit cannot shear a sample that
        should not be sheared. napari-spatialdata's similarity may reflect, so for mirrored
        landmarks the two can disagree.
    table_key
        For SpatialData input, read the landmarks from this table's ``obsm`` instead
        of a shapes element. A single key applies to both sides; a ``(ref, query)``
        pair addresses each side separately.
    spatial_key
        ``obsm`` key of the query array to transform, ``"spatial"`` by default. The
        landmarks are correspondences rather than the data, so this names what actually
        moves. Read only when ``key_added`` is given.
    key_added
        ``obsm`` key on the query to write the transformed ``spatial_key`` coordinates
        to. Given it, the affine is applied and written instead of returned; mutually
        exclusive with ``target_coordinate_system``.
    target_coordinate_system
        Register the fitted affine as a SpatialData transformation into this
        coordinate system instead of materialising anything: every element registered
        to the query's coordinate system inherits the alignment. Requires the
        landmarks to come from shapes elements, and refuses when the reference sits in
        the same coordinate system of the same object (it would be dragged along).

    Returns
    -------
    The fitted homogeneous ``(3, 3)`` affine in ``(x, y)`` when neither ``key_added``
    nor ``target_coordinate_system`` is given: directly usable as a
    :class:`~spatialdata.transformations.Affine`; otherwise ``None``, having written into
    the query container itself. Copy it first if the original must survive.
    """
    method = normalize_choice(method, ("similarity", "affine"), name="method")
    fit_fn = functools.partial(fit_landmarks, method=method)
    if key_added is not None and target_coordinate_system is not None:
        raise ValueError(
            "`key_added` and `target_coordinate_system` are mutually exclusive: the first materialises "
            "transformed coordinates, the second registers the fit as a transformation."
        )

    if not isinstance(data_ref, AnnData | SpatialData):
        if data_query is None:
            raise ValueError(
                "`data_ref` is an array of landmarks, so `data_query` must be the matching "
                "array of query landmarks rather than `None`."
            )
        for name, value in (
            ("landmark_key", landmark_key),
            ("table_key", table_key),
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

    if key_added is not None and isinstance(query_container, SpatialData) and query_table is None:
        raise ValueError(
            "`key_added` writes into a table's `obsm`, so it needs `table_key`, which also reads the "
            "landmarks from that table. With the landmarks in shapes elements, use "
            "`target_coordinate_system` to register the fit instead."
        )
    ref_lm = _read_landmarks(data_ref, ref_lm_key, ref_table, side="reference")
    query_lm = _read_landmarks(query_container, query_lm_key, query_table, side="query")

    if target_coordinate_system is not None:
        # Register the fit into a coordinate system instead of materialising it.
        if not isinstance(query_container, SpatialData):
            raise TypeError("`target_coordinate_system` registers a transformation, which only a SpatialData has.")
        if query_table is not None:
            raise ValueError(
                "`target_coordinate_system` needs the query landmarks in a shapes element, but "
                "`table_key` reads them from a table, which has no coordinate system of its own. "
                "Store the landmarks as a shapes element, or write to `key_added` to move only the "
                "query's coordinates."
            )

        moving_cs = _coordinate_system_of(query_container, element=query_lm_key, side="query")
        # registering moves everything in `moving_cs`, a reference there included
        if data_ref is query_container and (
            _table_regions_in(data_ref, ref_table, moving_cs)
            if ref_table is not None
            else _coordinate_system_of(data_ref, element=ref_lm_key, side="reference") == moving_cs
        ):
            raise ValueError(
                f"The reference and query are both in coordinate system {moving_cs!r}, so registering "
                f"the fit there would move the reference too. Put each sample in its own coordinate "
                f"system (what napari-spatialdata does when landmarks are picked per sample), or "
                f"read the landmarks from a table with `table_key` and write to `key_added` to move "
                f"only the query's coordinates."
            )

        writeback_affine_sdata(
            fit_fn(ref_lm, query_lm),
            query_container,
            moving_cs=moving_cs,
            target_cs=target_coordinate_system,
        )
        return None

    matrix = fit_fn(ref_lm, query_lm)
    if key_added is None:
        return matrix
    return write_coords(
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
        adata = resolve_table(container, table_key, side=side)
        return read_coords(adata, key=landmark_key, side=side, name="landmark_key")
    if not isinstance(container, SpatialData):
        raise TypeError(f"Expected the {side} to be an AnnData or SpatialData, got {type(container).__name__}.")
    if landmark_key not in container.shapes:
        raise KeyError(
            f"`landmark_key={landmark_key!r}`: no such shapes element in the {side}. "
            f"Available: {sorted(container.shapes)}. To read landmarks from a table's `obsm` "
            f"instead, pass `table_key`."
        )
    from spatialdata import transform

    coordinate_system = _coordinate_system_of(container, element=landmark_key, side=side)
    geometry = transform(container.shapes[landmark_key], to_coordinate_system=coordinate_system).geometry
    if not (geometry.geom_type == "Point").all():
        raise ValueError(
            f"`landmark_key={landmark_key!r}` on the {side} holds {sorted(set(geometry.geom_type))} geometries; "
            f"landmarks must be points."
        )
    return np.column_stack([geometry.x.to_numpy(), geometry.y.to_numpy()])


def _table_regions_in(sdata: SpatialData, table_key: str, coordinate_system: str) -> bool:
    """Whether any element the table annotates is registered to ``coordinate_system``."""
    from spatialdata.transformations import get_transformation

    region = sdata.tables[table_key].uns.get("spatialdata_attrs", {}).get("region")
    regions = [region] if isinstance(region, str) else list(region or [])
    return any(name in sdata and coordinate_system in get_transformation(sdata[name], get_all=True) for name in regions)


def _coordinate_system_of(sdata: SpatialData, *, element: str, side: str) -> str:
    """The one coordinate system the shapes element is annotated in; raises if ambiguous."""
    from spatialdata.transformations import get_transformation

    systems = sorted(get_transformation(sdata.shapes[element], get_all=True))
    if len(systems) != 1:
        raise ValueError(
            f"`landmark_key={element!r}` on the {side} is registered to {len(systems)} coordinate "
            f"systems ({', '.join(systems)}), so which one the alignment should move is ambiguous. "
            f"Register it to exactly one."
        )
    return systems[0]
