"""Public ``align_*`` orchestrators.

Each function is intentionally thin: resolve inputs, validate, dispatch to a
backend, write the result back.  All branching on argument shape lives in
:mod:`._io`; all backend selection lives in :mod:`._backends`; all validation
of "passed-but-unneeded" combinations lives in :mod:`._validation`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from squidpy.experimental.tl._align._backends import get_backend
from squidpy.experimental.tl._align._io import (
    apply_affine_to_cs,
    materialise_obs,
    resolve_element_pair,
    resolve_image_pair,
    resolve_obs_pair,
)
from squidpy.experimental.tl._align._types import AffineTransform, AlignPair, AlignResult
from squidpy.experimental.tl._align._validation import (
    ALLOWED_FLAVOURS_IMAGES,
    ALLOWED_FLAVOURS_OBS,
    ALLOWED_OUTPUT_MODES_NONOBS,
    ALLOWED_OUTPUT_MODES_OBS,
    validate_flavour,
    validate_key_added,
    validate_landmark_model,
    validate_landmarks,
    validate_output_mode,
    validate_required,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


def align_obs(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None = None,
    adata_ref_name: str | None = None,
    adata_query_name: str | None = None,
    flavour: Literal["stalign", "moscot"] = "stalign",
    *,
    output_mode: Literal["affine", "obs", "return"] = "affine",
    key_added: str | None = None,
    device: Literal["cpu", "gpu"] | None = None,
    inplace: bool = True,
    **flavour_kwargs: Any,
) -> AnnData | SpatialData | AlignResult | None:
    """Align two ``obs``-level point clouds (cells / spots).

    Parameters
    ----------
    data_ref, data_query
        Either both :class:`anndata.AnnData`, both :class:`spatialdata.SpatialData`,
        or ``data_ref`` a SpatialData and ``data_query=None`` (in which case
        ``adata_ref_name`` and ``adata_query_name`` select two different
        tables of the same SpatialData object).
    adata_ref_name, adata_query_name
        Required only when SpatialData inputs are used.  Passing them with
        AnnData inputs raises an educational :class:`ValueError`.
    flavour
        Backend to use.  ``'stalign'`` is the default LDDMM-based fit;
        ``'moscot'`` is OT-based.
    output_mode
        How to deliver the result:

        - ``'affine'`` — register the fitted affine on the query element via
          :func:`spatialdata.transformations.set_transformation`, so every
          element in the query coordinate system inherits the alignment.
          Requires the backend to produce an affine transform.
        - ``'obs'`` — bake the (possibly non-affine) fit into a new AnnData
          whose ``obsm['spatial']`` already lives in the reference coordinate
          system; for SpatialData inputs the new table is stored under
          ``key_added``.
        - ``'return'`` — return the raw :class:`AlignResult`; no writeback.
    key_added
        Required when ``output_mode='obs'`` and inputs are SpatialData.
        Rejected with any other ``output_mode``.
    device
        ``'cpu'``/``'gpu'`` to force a JAX device, or ``None`` to let JAX
        pick the default.  Only consulted by JAX-backed flavours.
    inplace
        If ``True``, mutate the query container; otherwise return a copy.
    **flavour_kwargs
        Backend-specific knobs forwarded as-is to the chosen backend.
    """
    validate_flavour(flavour, allowed=ALLOWED_FLAVOURS_OBS, op="align_obs")
    validate_output_mode(output_mode, allowed=ALLOWED_OUTPUT_MODES_OBS, op="align_obs")
    validate_key_added(key_added, output_mode)

    pair = resolve_obs_pair(data_ref, data_query, adata_ref_name, adata_query_name)
    backend = get_backend(flavour)
    result = backend.align_obs(pair, device=device, **flavour_kwargs)

    return _writeback(pair, result, output_mode=output_mode, key_added=key_added, inplace=inplace)


def align_images(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    img_ref_name: str | None = None,
    img_query_name: str | None = None,
    flavour: Literal["stalign"] = "stalign",
    *,
    scale_ref: str | Literal["auto"] = "auto",
    scale_query: str | Literal["auto"] = "auto",
    output_mode: Literal["affine", "return"] = "affine",
    device: Literal["cpu", "gpu"] | None = None,
    inplace: bool = True,
    **flavour_kwargs: Any,
) -> SpatialData | AlignResult | None:
    """Align two raster images living inside :class:`spatialdata.SpatialData`.

    Parameters
    ----------
    sdata_ref, sdata_query
        SpatialData containers.  Pass ``sdata_query=None`` to align two
        images of the same SpatialData against each other.
    img_ref_name, img_query_name
        Image element keys.
    flavour
        Only ``'stalign'`` is currently supported.
    scale_ref, scale_query
        Scale level for multi-scale image elements.  ``'auto'`` picks the
        coarsest level.  Single-scale images ignore this parameter.
    output_mode
        ``'affine'`` registers the fit on the query image element so all of
        its scales inherit the transformation; ``'return'`` returns the raw
        :class:`AlignResult`.
    device, inplace, flavour_kwargs
        See :func:`align_obs`.
    """
    validate_required(name="img_ref_name", value=img_ref_name, when="calling `align_images`")
    validate_required(name="img_query_name", value=img_query_name, when="calling `align_images`")
    validate_flavour(flavour, allowed=ALLOWED_FLAVOURS_IMAGES, op="align_images")
    validate_output_mode(output_mode, allowed=ALLOWED_OUTPUT_MODES_NONOBS, op="align_images")

    pair = resolve_image_pair(
        sdata_ref,
        sdata_query,
        img_ref_name,
        img_query_name,
        scale_ref=scale_ref,
        scale_query=scale_query,
    )
    backend = get_backend(flavour)
    result = backend.align_images(pair, device=device, **flavour_kwargs)

    return _writeback(pair, result, output_mode=output_mode, key_added=None, inplace=inplace)


def align_by_landmarks(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None = None,
    cs_name_ref: str | None = None,
    cs_name_query: str | None = None,
    scale_ref: str | None = None,
    scale_query: str | None = None,
    landmarks_ref: tuple[tuple[float, float], ...] | None = None,
    landmarks_query: tuple[tuple[float, float], ...] | None = None,
    *,
    model: Literal["similarity", "affine"] = "similarity",
    output_mode: Literal["affine", "return"] = "affine",
    inplace: bool = True,
) -> SpatialData | AlignResult | None:
    """Align by a closed-form fit on user-provided landmarks.

    Pure NumPy under the hood — JAX is **not** required for this path.

    Parameters
    ----------
    sdata_ref, sdata_query
        SpatialData containers.  Pass ``sdata_query=None`` to align two
        coordinate systems of the same SpatialData against each other.
    cs_name_ref, cs_name_query
        Coordinate system names.
    scale_ref, scale_query
        Optional scale identifiers used purely for landmark-extent
        validation: if you extracted your landmarks at a particular scale,
        passing the same scale here lets us catch the "wrong scale" footgun
        early.
    landmarks_ref, landmarks_query
        Equal-length sequences of ``(y, x)`` tuples.  ``model='similarity'``
        needs ≥ 2 pairs, ``model='affine'`` needs ≥ 3.
    model
        ``'similarity'`` (rotation + uniform scale + translation) or
        ``'affine'`` (full 6-parameter linear).
    output_mode, inplace
        See :func:`align_obs`.
    """
    validate_required(name="cs_name_ref", value=cs_name_ref, when="calling `align_by_landmarks`")
    validate_required(name="cs_name_query", value=cs_name_query, when="calling `align_by_landmarks`")
    validate_required(name="landmarks_ref", value=landmarks_ref, when="calling `align_by_landmarks`")
    validate_required(name="landmarks_query", value=landmarks_query, when="calling `align_by_landmarks`")

    validate_output_mode(output_mode, allowed=ALLOWED_OUTPUT_MODES_NONOBS, op="align_by_landmarks")
    validate_landmark_model(model)

    # We don't materialise extents here in the skeleton; backends / a future
    # PR can fill in the cs-extent lookup once we wire spatialdata.get_extent.
    ref_arr, query_arr = validate_landmarks(landmarks_ref, landmarks_query, model=model)

    pair = resolve_element_pair(sdata_ref, sdata_query, cs_name_ref, cs_name_query)

    from squidpy.experimental.tl._align._backends._landmark import fit_landmark_affine

    affine = fit_landmark_affine(
        ref_arr,
        query_arr,
        model=model,
        source_cs=cs_name_query,
        target_cs=cs_name_ref,
    )
    result = AlignResult(transform=affine, metadata={"flavour": "landmark", "model": model})

    return _writeback(pair, result, output_mode=output_mode, key_added=None, inplace=inplace)


# ---------------------------------------------------------------------------
# Internal: writeback dispatch
# ---------------------------------------------------------------------------


def _writeback(
    pair: AlignPair,
    result: AlignResult,
    *,
    output_mode: str,
    key_added: str | None,
    inplace: bool,
) -> AnnData | SpatialData | AlignResult | None:
    if output_mode == "return":
        return result

    if output_mode == "affine":
        if not isinstance(result.transform, AffineTransform):
            raise TypeError(
                f"`output_mode='affine'` requires the backend to return an AffineTransform, "
                f"got {type(result.transform).__name__}. Use `output_mode='obs'` (for "
                f"`align_obs`) or `output_mode='return'` to access non-affine fits."
            )
        return apply_affine_to_cs(pair, result.transform, inplace=inplace)

    if output_mode == "obs":
        return materialise_obs(pair, result, key_added=key_added, inplace=inplace)

    raise ValueError(f"Unknown output_mode {output_mode!r}.")
