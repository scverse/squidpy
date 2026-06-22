"""Public alignment functions built on the :mod:`squidpy.experimental.method_registry` core.

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

from squidpy._validators import assert_one_of
from squidpy.experimental.method_registry.align_landmarks import ALIGN_LANDMARKS
from squidpy.experimental.method_registry.align_samples import ALIGN_SAMPLES
from squidpy.experimental.tl._align._io import (
    get_coords,
    resolve_obs_pair,
    writeback_affine_sdata,
    writeback_obs,
)

if TYPE_CHECKING:
    from squidpy.experimental.method_registry import AlignResult, Registry

OUTPUT_MODES = ("object", "copy", "inplace")
ON_VALUES = ("obs", "image")

__all__ = ["align", "align_by_landmarks"]

F = TypeVar("F", bound="Callable[..., Any]")


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


@_document_methods(align_samples_methods=ALIGN_SAMPLES)
def align(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    *,
    method: str = "stalign",
    on: Literal["obs", "image"] = "obs",
    ref_key: str | None = None,
    query_key: str | None = None,
    spatial_key: str = "spatial",
    output_mode: Literal["object", "copy", "inplace"] = "object",
    key_added: str | None = None,
    **method_kwargs: Any,
) -> AlignResult | AnnData | SpatialData | None:
    """Align a query sample onto a reference sample.

    Parameters
    ----------
    data_ref, data_query
        Both :class:`~anndata.AnnData`, or both :class:`~spatialdata.SpatialData`,
        or ``data_ref`` a SpatialData with ``data_query=None`` to align two of its
        own tables (selected by ``ref_key`` / ``query_key``).
    method
        Fitting method in the ``align_samples`` family. See each implementation
        for its method-specific arguments:

        {align_samples_methods}
    on
        ``"obs"`` aligns the ``obsm`` point clouds. ``"image"`` is reserved and
        currently raises :class:`NotImplementedError`.
    ref_key, query_key
        Table keys, required (and only valid) for SpatialData inputs.
    spatial_key
        ``obsm`` key holding the ``(x, y)`` coordinates. Defaults to ``"spatial"``.
    output_mode
        - ``"object"`` (default) -- return the fitted :class:`~squidpy.experimental.tl.AlignResult`; nothing is written.
        - ``"inplace"`` -- write the aligned coordinates into the query and return ``None``.
        - ``"copy"`` -- write into a copy of the query and return the copy.
    key_added
        Destination ``obsm`` key for the aligned coordinates. If ``None`` it
        defaults to ``f"aligned_{spatial_key}"``; if that key already exists and
        ``key_added`` was not given explicitly, a :class:`ValueError` is raised
        (pass ``key_added`` to overwrite intentionally).
    method_kwargs
        Method-specific solver arguments, forwarded flat to the chosen
        ``method``'s implementation:

        {align_samples_methods}
    """
    assert_one_of(output_mode, OUTPUT_MODES, name="output_mode")
    assert_one_of(on, ON_VALUES, name="on")
    if on == "image":
        raise NotImplementedError("`align(on='image')` is not implemented yet; use `on='obs'`.")

    ref_adata, query_adata, container, element_key = resolve_obs_pair(data_ref, data_query, ref_key, query_key)
    ref_xy = get_coords(ref_adata, spatial_key)
    query_xy = get_coords(query_adata, spatial_key)

    result = ALIGN_SAMPLES.get(method)(ref=ref_xy, query=query_xy, **method_kwargs)

    return writeback_obs(
        result,
        output_mode=output_mode,
        query_adata=query_adata,
        container=container,
        element_key=element_key,
        spatial_key=spatial_key,
        key_added=key_added,
    )


@_document_methods(align_landmarks_methods=ALIGN_LANDMARKS)
def align_by_landmarks(
    ref: np.ndarray | Sequence[tuple[float, float]],
    query: np.ndarray | Sequence[tuple[float, float]],
    *,
    method: Literal["similarity", "affine"] = "similarity",
    data: AnnData | SpatialData | None = None,
    cs_name_ref: str | None = None,
    cs_name_query: str | None = None,
    spatial_key: str = "spatial",
    output_mode: Literal["object", "copy", "inplace"] = "object",
    key_added: str | None = None,
) -> AlignResult | AnnData | SpatialData | None:
    """Align by a closed-form fit on pre-paired landmarks.

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
        Target to write the alignment into. Required for ``output_mode`` other
        than ``"object"``.
    cs_name_ref, cs_name_query
        Coordinate-system names. For a SpatialData ``data`` the fitted affine is
        registered on every element in ``cs_name_query``, mapping into
        ``cs_name_ref``.
    spatial_key
        ``obsm`` key when ``data`` is an :class:`~anndata.AnnData`.
    output_mode
        See :func:`align`. ``"object"`` (default) returns the fitted
        :class:`~squidpy.experimental.tl.AlignResult`.
    key_added
        Destination ``obsm`` key when ``data`` is an AnnData (see :func:`align`).
    """
    assert_one_of(output_mode, OUTPUT_MODES, name="output_mode")

    result = ALIGN_LANDMARKS.get(method)(
        ref=ref,
        query=query,
        source_cs=cs_name_query,
        target_cs=cs_name_ref,
    )

    if output_mode == "object":
        return result
    if data is None:
        raise ValueError("`data` is required when `output_mode` is 'copy' or 'inplace'.")

    if isinstance(data, SpatialData):
        return writeback_affine_sdata(
            result, data, output_mode=output_mode, moving_cs=cs_name_query, target_cs=cs_name_ref
        )
    if isinstance(data, AnnData):
        return writeback_obs(
            result,
            output_mode=output_mode,
            query_adata=data,
            container=None,
            element_key=None,
            spatial_key=spatial_key,
            key_added=key_added,
        )
    raise TypeError(f"`data` must be AnnData or SpatialData, got {type(data).__name__}.")
