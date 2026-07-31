"""The public alignment function, built on the :mod:`squidpy.experimental.methods` core.

A thin orchestrator: resolve ``in_`` to in-memory arrays, dispatch to a fit-core
estimator, write the result back at ``out``. Path resolution lives in :mod:`._paths` and
transformation write-back in :mod:`._io`; the estimators themselves never see a container.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from squidpy.experimental.methods import ALIGN
from squidpy.experimental.methods.registry import MODALITIES
from squidpy.experimental.tl._align._io import shallow_copy_sdata, writeback_affine_sdata
from squidpy.experimental.tl._align._paths import DataPath, parse_path, read_path, write_path

if TYPE_CHECKING:
    from squidpy.experimental.methods import AlignResult, Modality, Registry

__all__ = ["align"]

F = TypeVar("F", bound="Callable[..., Any]")

#: Method used when ``method`` is left unset, per modality. Point clouds and rasters both
#: go to the diffeomorphic solver; landmarks default to the more constrained of the two
#: closed-form fits, since a similarity cannot shear a sample that should not be sheared.
DEFAULT_METHOD: dict[str, str] = {"obs": "stalign", "images": "stalign", "landmarks": "similarity"}


def _resolve_in(in_: str | tuple[str, str]) -> tuple[DataPath, DataPath]:
    """Normalise ``in_`` to a ``(ref_path, query_path)`` pair."""
    if isinstance(in_, str):
        path = parse_path(in_, name="in_")
        return path, path
    if not (isinstance(in_, tuple | list) and len(in_) == 2):
        raise ValueError(f"`in_` must be a path or a (ref_path, query_path) pair, got {in_!r}.")
    return parse_path(in_[0], name="in_[0]"), parse_path(in_[1], name="in_[1]")


def _check_path_suits(by: Modality, path: DataPath, *, name: str) -> None:
    """Reject a path whose contents the ``by`` slot could not consume.

    ``by`` says what drives the fit; the path only says where to read it from. Those are
    free to differ -- ``by="landmarks"`` with ``in_="obsm/lm"`` reads correspondences out
    of an ``obsm`` key, so no SpatialData is needed to hold four points. What cannot
    differ is the *shape*: raster slots need rasters and coordinate slots need ``(N, 2)``
    arrays, and catching that here beats a shape error from deep inside a solver.
    """
    wants_raster = by == "images"
    is_raster = path.modality == "images"
    if wants_raster and not is_raster:
        raise ValueError(f"`by='images'` needs an image path, but `{name}={path.raw!r}` reads coordinates.")
    if not wants_raster and is_raster:
        raise ValueError(f"`by={by!r}` needs an (N, 2) coordinate path, but `{name}={path.raw!r}` reads an image.")


def _copy_for_write(container: AnnData | SpatialData, path: DataPath) -> AnnData | SpatialData:
    """Duplicate just enough of ``container`` that writing at ``path`` leaves it untouched."""
    if isinstance(container, AnnData):
        return container.copy()

    out = shallow_copy_sdata(container)
    # `shallow_copy_sdata` shares element objects with the original, so the one element
    # we are about to write through still has to be duplicated.
    if path.modality == "obs" and path.element in out.tables:
        out.tables[path.element] = out.tables[path.element].copy()
    return out


def _methods_rst(registry: Registry, modality: Modality, indent: str = " " * 8) -> str:
    """Render the methods supporting ``modality`` as a reST list."""
    items = [
        f"- ``{key}`` -- :func:`~{(fn := registry.get(key).implementation(modality)).__module__}.{fn.__name__}`"
        for key in registry.supporting(modality)
    ]
    return ("\n" + indent).join(items)


def _document_methods(registry: Registry, **tokens: Modality) -> Callable[[F], F]:
    """Fill ``{<name>}`` docstring placeholders with each modality's method list.

    First-party and deterministic -- the registry is fully populated by import time, so
    this only templates known content (nothing from optional packages). ``str.replace``
    (not ``str.format``) leaves other ``{...}`` in the docstring untouched.
    """

    def decorator(fn: F) -> F:
        if fn.__doc__:
            for token, modality in tokens.items():
                fn.__doc__ = fn.__doc__.replace("{" + token + "}", _methods_rst(registry, modality))
        return fn

    return decorator


@_document_methods(ALIGN, obs_methods="obs", image_methods="images", landmark_methods="landmarks")
def align(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    in_: str | tuple[str, str],
    out: str | None = None,
    by: Modality = "obs",
    apply_to: str | None = None,
    method: str | None = None,
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
        - ``"shapes/landmarks"`` -- a shapes element, as napari-spatialdata writes landmarks

        ``in_`` says only *where* to read. What the arrays mean is ``by``'s job, so
        ``by="landmarks"`` can read correspondences straight out of an ``obsm`` key and
        needs no SpatialData just to hold a handful of points.
    out
        Where to write, as a path into ``data_query``. ``None`` (default) writes nothing
        and returns the fitted alignment instead -- fitting is expensive and usually worth
        inspecting before it overwrites anything.

        - an ``obsm`` path writes the transformed coordinates
        - an ``images`` path materialises the warped image
        - ``"cs/aligned"`` registers the fit as a transformation into that coordinate
          system, leaving the data untouched. Only available for methods whose fit is an
          affine; a diffeomorphism has no SpatialData transformation to be expressed as.
    by
        What drives the alignment:

        - ``"obs"`` (default) -- the point clouds themselves
        - ``"images"`` -- raster intensities
        - ``"landmarks"`` -- paired correspondences, matched by row order

        This also selects which of ``method``'s slots is used, so asking for a modality a
        method does not implement fails immediately and says what it does implement.
    apply_to
        Which array the fitted transform is applied to before writing. Defaults to ``in_``
        -- with ``by="obs"`` or ``"images"`` the thing you aligned is the thing you want
        moved. ``by="landmarks"`` is the exception: ``in_`` holds correspondences rather
        than data, so this must be given (or use ``out="cs/..."`` to move a whole
        coordinate system at once).
    method
        Which method to fit with. Defaults to ``"stalign"`` for ``by="obs"`` and
        ``by="images"``, and ``"similarity"`` for ``by="landmarks"``. Available per
        modality:

        ``by="obs"``:

        {obs_methods}

        ``by="images"``:

        {image_methods}

        ``by="landmarks"``:

        {landmark_methods}
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
    if by not in MODALITIES:
        raise ValueError(f"Unknown `by={by!r}`. Expected one of {', '.join(MODALITIES)}.")

    ref_path, query_path = _resolve_in(in_)
    if ref_path.modality != query_path.modality:
        raise ValueError(
            f"`in_` mixes modalities: {ref_path.raw!r} is {ref_path.modality}, "
            f"{query_path.raw!r} is {query_path.modality}. Both must address the same kind of data."
        )
    _check_path_suits(by, ref_path, name="in_")
    align_method = ALIGN.get(method if method is not None else DEFAULT_METHOD[by])

    query_container = data_ref if data_query is None else data_query
    if data_query is None and not isinstance(data_ref, SpatialData):
        raise ValueError("`data_query` is required unless `data_ref` is a SpatialData holding both elements.")

    ref_array = read_path(data_ref, ref_path, name="in_")
    query_array = read_path(query_container, query_path, name="in_")

    result = align_method.implementation(by)(ref=ref_array, query=query_array, **method_kwargs)

    if out is None:
        return result

    out_path = parse_path(out, name="out")
    if out_path.coordinate_system:
        return _register_transformation(result, data_ref, query_container, ref_path, query_path, out_path, copy=copy)

    source_path = _resolve_apply_to(apply_to, query_path, by, out_path)
    if out_path.modality != source_path.modality:
        raise ValueError(
            f"`out={out!r}` is {out_path.modality} but the data being transformed "
            f"({source_path.raw!r}) is {source_path.modality}; alignment does not convert between the two."
        )

    target = _copy_for_write(query_container, out_path) if copy else query_container
    if out_path.modality == "images":
        value = np.asarray(result.warp_image(read_path(target, source_path, name="apply_to")))
    else:
        value = np.asarray(result.transform(read_path(target, source_path, name="apply_to")))
    write_path(target, out_path, value)
    return target if copy else None


def _resolve_apply_to(
    apply_to: str | None,
    query_path: DataPath,
    by: Modality,
    out_path: DataPath,
) -> DataPath:
    """Which array the fitted transform is applied to.

    For an ``obs`` or ``images`` fit this is just ``in_`` -- the thing you aligned is the
    thing you want moved. A landmark fit is different: ``in_`` holds correspondences, not
    data, so the target has to be named. There is no default for that; guessing
    ``obsm/spatial`` would silently transform the wrong array in any dataset that happens
    to key its coordinates differently.
    """
    if apply_to is not None:
        return parse_path(apply_to, name="apply_to")
    if by == "landmarks":
        raise ValueError(
            f"`out={out_path.raw!r}` needs `apply_to` when aligning by landmarks: `in_` holds the "
            f"landmark correspondences, so it does not say which array to transform. Pass e.g. "
            f'`apply_to="obsm/spatial"`, or use `out="cs/<name>"` on a SpatialData to move every '
            f"element in the coordinate system at once."
        )
    return query_path


def _register_transformation(
    result: AlignResult,
    data_ref: AnnData | SpatialData,
    container: AnnData | SpatialData,
    ref_path: DataPath,
    query_path: DataPath,
    out_path: DataPath,
    *,
    copy: bool,
) -> SpatialData | None:
    """Register an affine fit into a coordinate system instead of materialising it."""
    if not isinstance(container, SpatialData):
        raise TypeError(f"`out={out_path.raw!r}` names a coordinate system, which only a SpatialData has.")
    if not hasattr(result, "matrix"):
        raise ValueError(
            f"`out={out_path.raw!r}` registers the fit as a transformation, but this method fits a "
            f"deformation that SpatialData has no transformation type for -- its transformations are "
            f"affine at most. Write to an `images/<name>` or `obsm/<key>` path to materialise it instead."
        )

    moving_cs = _coordinate_system_of(container, query_path, name="in_")
    # Registering moves *everything* in `moving_cs`. If the reference sits in that same
    # coordinate system of the same object, it would be dragged along with the query --
    # silently producing a wrong answer rather than failing.
    if data_ref is container and _coordinate_system_of(data_ref, ref_path, name="in_") == moving_cs:
        raise ValueError(
            f"The reference and query are both in coordinate system {moving_cs!r}, so registering "
            f"the fit there would move the reference too. Put each sample in its own coordinate "
            f"system (what napari-spatialdata does when landmarks are picked per sample), or write "
            f"to a data path with `apply_to` to move only the query."
        )

    return writeback_affine_sdata(
        result,
        container,
        output_mode="copy" if copy else "inplace",
        moving_cs=moving_cs,
        target_cs=out_path.element,
    )


#: Element collections whose members carry transformations. Tables do not: they annotate
#: elements rather than sitting in space themselves.
_SPATIAL_COLLECTIONS = ("shapes", "points", "images", "labels")


def _coordinate_system_of(sdata: SpatialData, path: DataPath, *, name: str) -> str:
    """The coordinate system the element at ``path`` is annotated in.

    Everything registered to it moves with the fit, so it has to be unambiguous. Reading
    it off the element rather than taking it as an argument keeps the call site to
    ``in_``/``out``, and it is the same element the user picked the landmarks on.
    """
    from spatialdata.transformations import get_transformation

    collection = path.raw.strip("/").split("/")[0]
    if collection not in _SPATIAL_COLLECTIONS:
        raise ValueError(
            f'`out="cs/..."` needs `{name}={path.raw!r}` to name a spatial element, but a table has no '
            f"coordinate system of its own. Store the landmarks as a shapes element, or write to a "
            f"data path with `apply_to` to move only the query's coordinates."
        )

    systems = sorted(get_transformation(getattr(sdata, collection)[path.element], get_all=True))
    if len(systems) != 1:
        raise ValueError(
            f"`{name}={path.raw!r}` is registered to {len(systems)} coordinate systems ({', '.join(systems)}), "
            f"so which one the alignment should move is ambiguous. Register it to exactly one."
        )
    return systems[0]
