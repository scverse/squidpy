"""Public alignment functions built on the :mod:`squidpy.experimental.methods` core.

These are thin orchestrators: resolve inputs to in-memory arrays, dispatch to a
fit-core estimator, write the result back. All container I/O and write-back live
in :mod:`._io`; the estimators themselves never see a container.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from squidpy.experimental.methods import ALIGN_IMAGES, ALIGN_LANDMARKS, ALIGN_SAMPLES
from squidpy.experimental.tl._align._io import shallow_copy_sdata, writeback_affine_sdata
from squidpy.experimental.tl._align._paths import DataPath, parse_path, read_path, write_path

if TYPE_CHECKING:
    from squidpy.experimental.methods import AlignResult, Registry

__all__ = ["align", "align_by_landmarks"]

F = TypeVar("F", bound="Callable[..., Any]")


def _resolve_in(in_: str | tuple[str, str]) -> tuple[DataPath, DataPath]:
    """Normalise ``in_`` to a ``(ref_path, query_path)`` pair."""
    if isinstance(in_, str):
        path = parse_path(in_, name="in_")
        return path, path
    if not (isinstance(in_, tuple | list) and len(in_) == 2):
        raise ValueError(f"`in_` must be a path or a (ref_path, query_path) pair, got {in_!r}.")
    return parse_path(in_[0], name="in_[0]"), parse_path(in_[1], name="in_[1]")


def _copy_for_write(container: AnnData | SpatialData, path: DataPath) -> AnnData | SpatialData:
    """Duplicate just enough of ``container`` that writing at ``path`` leaves it untouched."""
    if isinstance(container, AnnData):
        return container.copy()

    out = shallow_copy_sdata(container)
    # `shallow_copy_sdata` shares element objects with the original, so the one element
    # we are about to write through still has to be duplicated.
    if path.modality == "points" and path.element in out.tables:
        out.tables[path.element] = out.tables[path.element].copy()
    return out


def _methods_rst(registry: Registry, indent: str = " " * 8) -> str:
    """Render a registry's methods as a reST list linking to each implementation."""
    items = [f"- ``{key}`` -- :func:`~{(fn := registry.get(key)).__module__}.{fn.__name__}`" for key in registry.keys()]
    return ("\n" + indent).join(items)


def _document_methods(**registries: Registry) -> Callable[[F], F]:
    """Fill ``{<name>}`` docstring placeholders with each registry's method list.

    First-party and deterministic -- the registries are fully populated by import
    time, so this only templates known content (nothing from optional packages).
    ``str.replace`` (not ``str.format``) leaves other ``{...}`` in the docstring
    untouched.
    """

    def decorator(fn: F) -> F:
        if fn.__doc__:
            for token, registry in registries.items():
                fn.__doc__ = fn.__doc__.replace("{" + token + "}", _methods_rst(registry))
        return fn

    return decorator


@_document_methods(align_samples_methods=ALIGN_SAMPLES, align_images_methods=ALIGN_IMAGES)
def align(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    in_: str | tuple[str, str],
    out: str | None = None,
    method: str = "stalign",
    copy: bool = False,
    **method_kwargs: Any,
) -> AlignResult | AnnData | SpatialData | None:
    """Align a query sample onto a reference sample.

    Parameters
    ----------
    data_ref, data_query
        Both :class:`~anndata.AnnData`, or both :class:`~spatialdata.SpatialData`, or
        ``data_ref`` a SpatialData with ``data_query=None`` to align two of its own
        elements (distinguished by passing a pair to ``in_``).
    in_
        Where to read from. One path applied to both containers, or a
        ``(ref_path, query_path)`` pair. Accepted forms:

        - ``"obsm/spatial"`` -- an AnnData ``obsm`` key
        - ``"tables/slice1/obsm/spatial"`` -- an ``obsm`` key of a SpatialData table
        - ``"images/he"`` -- a SpatialData image

        The path also selects what is aligned: an ``obsm`` path fits on point clouds,
        an ``images`` path fits on image intensities.
    out
        Where to write the aligned query, as a path into ``data_query``. ``None``
        (default) writes nothing and returns the fitted alignment instead -- fitting is
        expensive and usually worth inspecting before it overwrites anything.

        For an ``obsm`` path the transformed coordinates are written. For an ``images``
        path the warped image is materialised: a diffeomorphism cannot be expressed as a
        SpatialData transformation, so it cannot be registered lazily.
    method
        Fitting method. For ``obsm`` paths, one of the ``align_samples`` family:

        {align_samples_methods}

        For ``images`` paths, one of the ``align_images`` family:

        {align_images_methods}
    copy
        Write into a copy of the query container and return it, instead of mutating in
        place. Ignored when ``out`` is ``None``.
    method_kwargs
        Solver arguments, forwarded flat to the chosen ``method``.

    Returns
    -------
    The fitted :class:`~squidpy.experimental.tl.AlignResult` when ``out`` is ``None``;
    the modified copy when ``copy=True``; otherwise ``None``.
    """
    ref_path, query_path = _resolve_in(in_)
    if ref_path.modality != query_path.modality:
        raise ValueError(
            f"`in_` mixes modalities: {ref_path.raw!r} is {ref_path.modality}, "
            f"{query_path.raw!r} is {query_path.modality}. Both must address the same kind of data."
        )

    ref_container = data_ref
    query_container = data_ref if data_query is None else data_query
    if data_query is None and not isinstance(data_ref, SpatialData):
        raise ValueError("`data_query` is required unless `data_ref` is a SpatialData holding both elements.")

    ref_array = read_path(ref_container, ref_path, name="in_")
    query_array = read_path(query_container, query_path, name="in_")

    registry = ALIGN_IMAGES if ref_path.modality == "image" else ALIGN_SAMPLES
    result = registry.get(method)(ref=ref_array, query=query_array, **method_kwargs)

    if out is None:
        return result

    out_path = parse_path(out, name="out")
    if out_path.modality != query_path.modality:
        raise ValueError(
            f"`out={out!r}` is {out_path.modality} but `in_` is {query_path.modality}; "
            f"alignment does not convert between the two."
        )

    target = _copy_for_write(query_container, out_path) if copy else query_container
    if out_path.modality == "points":
        value = np.asarray(result.transform(query_array))
    else:
        value = np.asarray(result.warp_image(query_array))
    write_path(target, out_path, value)
    return target if copy else None


@_document_methods(align_landmarks_methods=ALIGN_LANDMARKS)
def align_by_landmarks(
    ref: np.ndarray | Sequence[tuple[float, float]],
    query: np.ndarray | Sequence[tuple[float, float]],
    *,
    method: Literal["similarity", "affine"] = "similarity",
    data: AnnData | SpatialData | None = None,
    in_: str | None = None,
    out: str | None = None,
    copy: bool = False,
    cs_ref: str | None = None,
    cs_query: str | None = None,
) -> AlignResult | AnnData | SpatialData | None:
    """Align by a closed-form fit on pre-paired landmarks.

    Kept separate from :func:`align` because the two write fundamentally different
    things: a similarity or affine fit *is* representable as a SpatialData
    transformation, so it can be registered lazily on a whole coordinate system, whereas
    :func:`align`'s diffeomorphism has to be materialised.

    Parameters
    ----------
    ref, query
        Equal-length ``(N, 2)`` ``(x, y)`` landmark arrays (``N >= 3``), paired by
        row order. No automatic correspondence matching is performed.
    method
        Fitting method in the ``align_landmarks`` family. See each implementation
        for its method-specific arguments:

        {align_landmarks_methods}
    data
        Container to write the alignment into. Required unless ``out`` is ``None``.
    in_
        For an AnnData ``data``, the path of the coordinates to transform, e.g.
        ``"obsm/spatial"``. Unused when registering a transformation on a SpatialData.
    out
        Where to write. ``None`` (default) writes nothing and returns the fitted affine.
        For an AnnData ``data``, a path such as ``"obsm/aligned"``. For a SpatialData,
        the affine is registered rather than materialised, so ``out`` is ignored and
        ``cs_ref`` / ``cs_query`` select the coordinate systems instead.
    copy
        Write into a copy of ``data`` and return it, instead of mutating in place.
    cs_ref, cs_query
        Coordinate-system names. For a SpatialData ``data`` the fitted affine is
        registered on every element living in ``cs_query``, mapping it into ``cs_ref``.

    Returns
    -------
    The fitted affine when ``out`` is ``None`` and ``data`` is ``None``; the modified
    copy when ``copy=True``; otherwise ``None``.
    """
    result = ALIGN_LANDMARKS.get(method)(ref=ref, query=query, source_cs=cs_query, target_cs=cs_ref)

    if data is None:
        if out is not None:
            raise ValueError("`data` is required when `out` is given.")
        return result

    if isinstance(data, SpatialData):
        # A similarity/affine transform is representable, so register it across the
        # coordinate system rather than resampling anything.
        return writeback_affine_sdata(
            result,
            data,
            output_mode="copy" if copy else "inplace",
            moving_cs=cs_query,
            target_cs=cs_ref,
        )

    if not isinstance(data, AnnData):
        raise TypeError(f"`data` must be AnnData or SpatialData, got {type(data).__name__}.")

    if out is None:
        return result
    in_path = parse_path(in_ or "obsm/spatial", name="in_")
    out_path = parse_path(out, name="out")
    if in_path.modality != "points" or out_path.modality != "points":
        raise ValueError("`align_by_landmarks` reads and writes point coordinates; use an `obsm/...` path.")

    target = data.copy() if copy else data
    coords = read_path(target, in_path, name="in_")
    write_path(target, out_path, np.asarray(result.transform(coords)))
    return target if copy else None
