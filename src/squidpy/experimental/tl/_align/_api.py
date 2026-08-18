"""The public alignment functions, built on the array-in / array-out estimators.

Thin orchestrators, one per method: resolve the ``*_key`` arguments to in-memory arrays,
call the estimator, write the result back. The estimators themselves never see a
container; SpatialData transformation write-back lives in :mod:`._io`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Unpack

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from ._io import shallow_copy_sdata, writeback_affine_sdata
from ._landmark import fit_affine, fit_similarity
from ._stalign import (
    StalignObsSolverKwargs,
    StalignSliceSolverKwargs,
    StalignSolverKwargs,
    fit_stalign_image,
    fit_stalign_obs,
    fit_stalign_slice,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from ._landmark import AffineFitResult
    from ._stalign import StalignResult, StalignSliceResult

__all__ = ["align_landmarks", "align_stalign_image", "align_stalign_obs", "align_stalign_slice"]


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

    element = container.images[key]
    # Multiscale images are a DataTree; the full-resolution level is the first scale.
    if not hasattr(element, "dims"):
        element = next(iter(element.values()))
        element = element[next(iter(element.data_vars))]

    return _as_chw(element.data, what=f"`image_key={key!r}`", ndim=ndim)


def _copy_for_write(container: AnnData | SpatialData, table_key: str | None) -> AnnData | SpatialData:
    """Duplicate just enough of ``container`` that writing into its table leaves it untouched."""
    if isinstance(container, AnnData):
        return container.copy()
    out = shallow_copy_sdata(container)
    # `shallow_copy_sdata` shares element objects with the original, so the one table
    # we are about to write through still has to be duplicated.
    if table_key is not None and table_key in out.tables:
        out.tables[table_key] = out.tables[table_key].copy()
    return out


def _write_coords(
    container: AnnData | SpatialData,
    table_key: str | None,
    spatial_key: str,
    key_added: str,
    *,
    transform: Callable[[np.ndarray], npt.ArrayLike],
    spatial_key_name: str,
    inplace: bool,
) -> AnnData | SpatialData | None:
    """Transform ``obsm[spatial_key]`` and write it to ``obsm[key_added]`` on the query."""
    target = container if inplace else _copy_for_write(container, table_key)
    adata = _resolve_table(target, table_key, side="query")
    coords = _read_coords(adata, spatial_key, side="query", name=spatial_key_name)
    adata.obsm[key_added] = np.asarray(transform(coords))
    return None if inplace else target


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


def check_inplace(inplace: bool, *targets: str | None, names: str) -> None:
    if inplace and all(target is None for target in targets):
        raise ValueError(
            f"`inplace=True` has nothing to write: pass {names} to say what to write. "
            f"Without it the fit is returned and nothing is modified."
        )


def align_stalign_obs(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    spatial_key: str | tuple[str, str] = "spatial",
    table_key: str | tuple[str | None, str | None] | None = None,
    key_added: str | None = None,
    inplace: bool = False,
    landmarks_ref: npt.ArrayLike | None = None,
    landmarks_query: npt.ArrayLike | None = None,
    **solver_kwargs: Unpack[StalignObsSolverKwargs],
) -> StalignResult | AnnData | SpatialData | None:
    """Align a query point cloud onto a reference with STalign (diffeomorphic LDDMM).

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
    key_added
        ``obsm`` key on the query to write the transformed coordinates to. ``None``
        (default) writes nothing and returns the fitted alignment instead -- fitting is
        expensive and usually worth inspecting before it overwrites anything.
    inplace
        Whether to write into the query container itself. ``False`` (default) writes
        into a copy and returns it, leaving the input untouched. Needs ``key_added``:
        ``inplace=True`` with nothing to write raises rather than being ignored.
    landmarks_ref, landmarks_query
        Optional paired ``(x, y)`` landmark arrays (matched by row order) used to
        initialise the affine.
    solver_kwargs
        LDDMM solver tuning; see
        :class:`~squidpy.experimental.tl.StalignObsSolverKwargs` for the accepted
        keys, their meaning, and their defaults.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.StalignResult` when
    ``key_added`` is ``None``; otherwise the modified copy, or ``None`` when
    ``inplace=True``.
    """
    check_inplace(inplace, key_added, names="`key_added`")
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

    result = fit_stalign_obs(
        ref=_read_coords(ref_adata, ref_spatial, side="reference", name="spatial_key"),
        query=_read_coords(query_adata, query_spatial, side="query", name="spatial_key"),
        landmarks_ref=landmarks_ref,
        landmarks_query=landmarks_query,
        **solver_kwargs,
    )
    if key_added is None:
        return result
    return _write_coords(
        query_container,
        query_table,
        query_spatial,
        key_added,
        transform=result.transform,
        spatial_key_name="spatial_key",
        inplace=inplace,
    )


def align_stalign_image(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    *,
    image_key: str | tuple[str, str],
    key_added: str | None = None,
    inplace: bool = False,
    ref_scale: tuple[float, float] = (1.0, 1.0),
    query_scale: tuple[float, float] = (1.0, 1.0),
    ref_axes: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    query_axes: tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    **solver_kwargs: Unpack[StalignSolverKwargs],
) -> StalignResult | SpatialData | None:
    """Align a query image onto a reference image with STalign (diffeomorphic LDDMM).

    Parameters
    ----------
    sdata_ref, sdata_query
        The :class:`~spatialdata.SpatialData` objects holding the reference and query
        images. Leave ``sdata_query=None`` with ``sdata_ref`` holding both images,
        distinguished by an ``image_key`` pair.
    image_key
        Name of the image element, or a ``(ref, query)`` pair.
    key_added
        Image element name on the query to materialise the warped image under. The
        fitted diffeomorphism cannot be expressed as a SpatialData transformation --
        the available types are affine at most -- so the aligned image is written as a
        new element. ``None`` (default) writes nothing and returns the fitted
        alignment instead.
    inplace
        Whether to write into ``sdata_query`` itself. ``False`` (default) writes into a
        copy and returns it, leaving the input untouched. Needs ``key_added``:
        ``inplace=True`` with nothing to write raises rather than being ignored.
    ref_scale, query_scale
        Physical size of one pixel as ``(y, x)``; pass these when the two images have
        different resolutions.
    ref_axes, query_axes
        Optional explicit physical row/column axes; mutually exclusive with non-unit
        scales.
    solver_kwargs
        LDDMM solver tuning; see
        :class:`~squidpy.experimental.tl.StalignSolverKwargs` for the accepted
        keys, their meaning, and their defaults.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.StalignResult` when
    ``key_added`` is ``None``; otherwise the modified copy, or ``None`` when
    ``inplace=True``.
    """
    check_inplace(inplace, key_added, names="`key_added`")
    ref_image, query_image = _resolve_pair(image_key, name="image_key")
    query_container = _query_of(
        sdata_ref, sdata_query, ref_address=(ref_image,), query_address=(query_image,), key_name="image_key"
    )

    query_array = _read_image(query_container, query_image, side="query")
    result = fit_stalign_image(
        ref=_read_image(sdata_ref, ref_image, side="reference"),
        query=query_array,
        ref_scale=ref_scale,
        query_scale=query_scale,
        ref_axes=ref_axes,
        query_axes=query_axes,
        **solver_kwargs,
    )
    if key_added is None:
        return result

    from spatialdata.models import Image2DModel

    target = query_container if inplace else shallow_copy_sdata(query_container)
    warped = _as_chw(result.warp_image(query_array), what=f"`key_added={key_added!r}`")
    target.images[key_added] = Image2DModel.parse(warped, dims=("c", "y", "x"))
    return None if inplace else target


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


def align_stalign_slice(
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
    table_key: str | None = None,
    spatial_key: str = "spatial",
    key_added: str | None = None,
    inplace: bool = False,
    **solver_kwargs: Unpack[StalignSliceSolverKwargs],
) -> StalignSliceResult | SpatialData | None:
    """Place a 2D section into a 3D reference volume with STalign (diffeomorphic LDDMM).

    The plane of a physical section is unknown and generally not exactly coronal, so this
    fits the full 3D deformation rather than an oblique plane plus an in-plane 2D fit;
    ``initial_slice`` / ``initial_rotation`` / ``initial_scale`` are an initialisation, not
    the answer. Every cell in ``spatial_key`` then gets real reference coordinates.

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
        Uniform scale of the initial affine. Upstream's notebooks start at 0.9.
    initial_affine
        Homogeneous ``(4, 4)`` affine in ``(x, y, z)``, replacing the three ``initial_*``
        arguments above. The escape hatch when the initialisation needs to be exact.
    table_key
        Which of the query's tables holds the cell coordinates to transform. Required when
        ``key_added`` is given.
    spatial_key
        ``obsm`` key on the query table holding the ``(N, 2)`` section coordinates.
    key_added
        ``obsm`` key on the query table to write the ``(N, 3)`` reference coordinates to.
        ``None`` (default) writes nothing and returns the fit instead -- this is an
        expensive fit and usually worth inspecting first.
    inplace
        Whether to write into ``sdata_query`` itself. ``False`` (default) writes into a
        copy and returns it. Needs ``key_added``.
    solver_kwargs
        LDDMM solver tuning; see
        :class:`~squidpy.experimental.tl.StalignSliceSolverKwargs`.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.StalignSliceResult` when ``key_added`` is
    ``None``; otherwise the modified copy, or ``None`` when ``inplace=True``.

    Notes
    -----
    The reference volume typically carries two channels -- a normalised intensity volume
    and its centred square -- against a single-channel section; the contrast transform
    fits one to the other, so the channel counts need not match.

    ``muA`` / ``muB`` take one entry per *section* channel. Upstream's published notebook
    passes three against a single-channel target and relies on broadcasting, which sums
    three identical terms and makes its effective artifact scale ``sigmaA / sqrt(3)``
    rather than ``sigmaA``. squidpy validates the length instead, so ``muA=[3.0]`` here is
    **not** the same fit as that notebook's ``muA=[3, 3, 3]``.
    """
    check_inplace(inplace, key_added, names="`key_added`")
    ref_image, query_image = _resolve_pair(image_key, name="image_key")
    query_container = _query_of(
        sdata_ref, sdata_query, ref_address=(ref_image,), query_address=(query_image,), key_name="image_key"
    )
    if key_added is not None and table_key is None:
        raise ValueError(
            "`key_added` writes the reference coordinates into a table's `obsm`, so "
            "`table_key` is needed to say which table holds the cells."
        )

    ref_array = _read_image(sdata_ref, ref_image, side="reference", ndim=3)
    query_array = _read_image(query_container, query_image, side="query")

    result = fit_stalign_slice(
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
    if key_added is None:
        return result
    return _write_coords(
        query_container,
        table_key,
        spatial_key,
        key_added,
        transform=result.transform,
        spatial_key_name="spatial_key",
        inplace=inplace,
    )


def align_landmarks(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    landmark_key: str | tuple[str, str],
    fit: Literal["similarity", "affine"] = "similarity",
    table_key: str | tuple[str | None, str | None] | None = None,
    spatial_key: str | None = None,
    key_added: str | None = None,
    target_coordinate_system: str | None = None,
    inplace: bool = False,
) -> AffineFitResult | AnnData | SpatialData | None:
    """Align a query sample onto a reference from paired landmarks (closed-form affine).

    Parameters
    ----------
    data_ref, data_query
        The reference and query samples, each an :class:`~anndata.AnnData` or a
        :class:`~spatialdata.SpatialData`. Leave ``data_query=None`` with ``data_ref``
        a SpatialData holding both samples' landmarks, distinguished by a
        ``landmark_key`` pair.
    landmark_key
        Where the ``(N, 2)`` landmark correspondences live (matched by row order), or a
        ``(ref, query)`` pair. On an AnnData -- or a SpatialData with ``table_key`` --
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
    inplace
        Whether to write into the query container itself. ``False`` (default) writes
        into a copy and returns it, leaving the input untouched. Needs one of
        ``key_added`` or ``target_coordinate_system``: ``inplace=True`` with nothing to
        write raises rather than being ignored.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.AffineFitResult`
    when neither ``key_added`` nor ``target_coordinate_system`` is given; otherwise
    the modified copy, or ``None`` when ``inplace=True``.
    """
    check_inplace(inplace, key_added, target_coordinate_system, names="`key_added` or `target_coordinate_system`")
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
            inplace=inplace,
        )

    result = fit_fn(ref_lm, query_lm)
    if key_added is None:
        return result
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
        transform=result.transform,
        spatial_key_name="spatial_key",
        inplace=inplace,
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
    fit_fn: Callable[..., AffineFitResult],
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
    inplace: bool,
) -> SpatialData | None:
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

    result = fit_fn(ref_lm, query_lm, source_cs=moving_cs, target_cs=target_coordinate_system)
    return writeback_affine_sdata(
        result,
        query_container,
        inplace=inplace,
        moving_cs=moving_cs,
        target_cs=target_coordinate_system,
    )


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
