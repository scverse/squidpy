"""Input resolvers and output writers for the alignment skeleton.

This module is the *only* place that knows about the duck-typed
``AnnData | SpatialData`` argument shape of the public functions and the
``output_mode`` writeback strategies.  Backends operate on the canonical
:class:`AlignPair` produced here.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from squidpy._validators import assert_isinstance, assert_key_in_sdata
from squidpy.experimental.im._utils import get_element_data
from squidpy.experimental.tl._align._types import (
    AffineTransform,
    AlignPair,
    AlignResult,
)
from squidpy.experimental.tl._align._validation import (
    validate_required,
    validate_unexpected,
)

# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def resolve_obs_pair(
    data_ref: AnnData | SpatialData,
    data_query: AnnData | SpatialData | None,
    adata_ref_name: str | None,
    adata_query_name: str | None,
) -> AlignPair:
    """Normalise the arguments of :func:`align_obs` into an :class:`AlignPair`.

    See the table in the design doc for the exhaustive case matrix.  In short:

    - both AnnData → use directly, ``adata_*_name`` must be ``None``;
    - both SpatialData → ``adata_*_name`` required, extract from each;
    - only ``data_ref`` is SpatialData (``data_query is None``) → both
      ``adata_*_name`` required, extract from the same sdata;
    - mixed AnnData/SpatialData → :class:`TypeError`;
    - ``data_ref`` is AnnData with no ``data_query`` → :class:`ValueError`.
    """
    if isinstance(data_ref, AnnData):
        if data_query is None:
            raise ValueError("`data_query` is required when `data_ref` is an AnnData.")
        if not isinstance(data_query, AnnData):
            raise TypeError(
                f"Mixed AnnData/SpatialData inputs are not supported. "
                f"`data_ref` is AnnData but `data_query` is {type(data_query).__name__}."
            )
        validate_unexpected(
            name="adata_ref_name",
            value=adata_ref_name,
            when="`data_ref` is a SpatialData",
            hint="Both inputs are AnnData, so there is no table to look up by name.",
        )
        validate_unexpected(
            name="adata_query_name",
            value=adata_query_name,
            when="`data_query` is a SpatialData",
            hint="Both inputs are AnnData, so there is no table to look up by name.",
        )
        return AlignPair(ref=data_ref, query=data_query)

    if not isinstance(data_ref, SpatialData):
        raise TypeError(f"`data_ref` must be AnnData or SpatialData, got {type(data_ref).__name__}.")

    if data_query is None:
        sdata_query: SpatialData = data_ref
    elif isinstance(data_query, SpatialData):
        sdata_query = data_query
    else:
        raise TypeError(
            f"Mixed AnnData/SpatialData inputs are not supported. "
            f"`data_ref` is SpatialData but `data_query` is {type(data_query).__name__}."
        )

    validate_required(name="adata_ref_name", value=adata_ref_name, when="`data_ref` is a SpatialData")
    validate_required(name="adata_query_name", value=adata_query_name, when="`data_query` is a SpatialData")
    assert_key_in_sdata(data_ref, adata_ref_name, attr="tables")
    assert_key_in_sdata(sdata_query, adata_query_name, attr="tables")
    return AlignPair(
        ref=data_ref.tables[adata_ref_name],
        query=sdata_query.tables[adata_query_name],
        ref_container=data_ref,
        query_container=sdata_query,
        ref_element_key=adata_ref_name,
        query_element_key=adata_query_name,
    )


def resolve_image_pair(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None,
    img_ref_name: str,
    img_query_name: str,
    *,
    scale_ref: str | Literal["auto"] = "auto",
    scale_query: str | Literal["auto"] = "auto",
) -> AlignPair:
    """Normalise the arguments of :func:`align_images` into an :class:`AlignPair`.

    Both single-scale ``xr.DataArray`` and multi-scale ``xr.DataTree`` image
    elements are accepted.  Multiscale nodes are flattened via
    :func:`squidpy.experimental.im._utils.get_element_data`, but the original
    element node is remembered in the :class:`AlignPair` so the writer can
    register the transformation on the parent so all scales inherit it.
    """
    assert_isinstance(sdata_ref, SpatialData, name="sdata_ref")
    if sdata_query is None:
        sdata_query = sdata_ref
    else:
        assert_isinstance(sdata_query, SpatialData, name="sdata_query")

    assert_key_in_sdata(sdata_ref, img_ref_name, attr="images")
    assert_key_in_sdata(sdata_query, img_query_name, attr="images")

    ref_node = sdata_ref.images[img_ref_name]
    query_node = sdata_query.images[img_query_name]

    ref_data = get_element_data(ref_node, scale_ref, element_type="image", element_key=img_ref_name)
    query_data = get_element_data(query_node, scale_query, element_type="image", element_key=img_query_name)

    return AlignPair(
        ref=ref_data,
        query=query_data,
        ref_container=sdata_ref,
        query_container=sdata_query,
        ref_element_key=img_ref_name,
        query_element_key=img_query_name,
    )


def resolve_element_pair(
    sdata_ref: SpatialData,
    sdata_query: SpatialData | None,
    cs_name_ref: str,
    cs_name_query: str,
) -> AlignPair:
    """Normalise the arguments of :func:`align_by_landmarks` into an :class:`AlignPair`.

    No element data is materialised — landmark fitting only needs the
    coordinate system names plus the landmark coordinates, which are
    validated separately.  The returned pair carries the containers and cs
    names so the writer can call :func:`set_transformation` on the right
    target.
    """
    assert_isinstance(sdata_ref, SpatialData, name="sdata_ref")
    if sdata_query is None:
        sdata_query = sdata_ref
    else:
        assert_isinstance(sdata_query, SpatialData, name="sdata_query")

    _check_cs_exists(sdata_ref, cs_name_ref, name="cs_name_ref")
    _check_cs_exists(sdata_query, cs_name_query, name="cs_name_query")

    return AlignPair(
        ref=None,
        query=None,
        ref_container=sdata_ref,
        query_container=sdata_query,
        ref_cs=cs_name_ref,
        query_cs=cs_name_query,
    )


def _check_cs_exists(sdata: SpatialData, cs_name: str, *, name: str) -> None:
    available = list(sdata.coordinate_systems)
    if cs_name not in available:
        raise KeyError(
            f"`{name}={cs_name!r}` is not a coordinate system of the SpatialData object. "
            f"Available coordinate systems: {available}."
        )


# ---------------------------------------------------------------------------
# Writeback
# ---------------------------------------------------------------------------


def apply_affine_to_cs(
    pair: AlignPair,
    affine: AffineTransform,
    *,
    inplace: bool,
) -> SpatialData | AnnData | None:
    """Register ``affine`` on the query side of the pair.

    Three writeback paths, in order of specificity:

    1. **Element-keyed**: ``pair.query_container`` and ``pair.query_element_key``
       are both set (e.g. ``align_obs`` / ``align_images`` resolved an explicit
       table or image). Register the transform on that single element so all
       scales / sibling tables that share its parent element node inherit it.
    2. **Cs-keyed**: only ``pair.query_cs`` is set (e.g. ``align_by_landmarks``
       resolved a coordinate system but no specific element). Walk every
       element that has the moving cs in its transformation graph and register
       the transform on each, mapping into the reference cs.
    3. **Plain AnnData**: no spatialdata container at all - warp
       ``query.obsm['spatial']`` directly.
    """
    from spatialdata.transformations import get_transformation, set_transformation

    target_cs = affine.target_cs or pair.ref_cs or "aligned"

    if pair.query_container is not None and pair.query_element_key is not None:
        sdata = pair.query_container if inplace else _shallow_copy_sdata(pair.query_container)
        element = _get_element(sdata, pair.query_element_key)
        set_transformation(element, affine.to_spatialdata(), to_coordinate_system=target_cs)
        return None if inplace else sdata

    if pair.query_container is not None and pair.query_cs is not None:
        sdata = pair.query_container if inplace else _shallow_copy_sdata(pair.query_container)
        moving_cs = pair.query_cs
        sd_affine = affine.to_spatialdata()
        touched_any = False
        for _etype, _name, element in sdata.gen_elements():
            if isinstance(element, AnnData):
                continue
            element_transforms = get_transformation(element, get_all=True)
            if moving_cs not in element_transforms:
                continue
            set_transformation(element, sd_affine, to_coordinate_system=target_cs)
            touched_any = True
        if not touched_any:
            raise KeyError(
                f"No elements in the query SpatialData are registered to coordinate "
                f"system {moving_cs!r}; nothing to attach the alignment to."
            )
        return None if inplace else sdata

    if isinstance(pair.query, AnnData):
        adata = pair.query if inplace else pair.query.copy()
        if "spatial" not in adata.obsm:
            raise KeyError("Cannot apply an affine to an AnnData query that has no `obsm['spatial']`.")
        adata.obsm["spatial"] = affine.apply(np.asarray(adata.obsm["spatial"]))
        return None if inplace else adata

    raise RuntimeError("apply_affine_to_cs: pair has neither a SpatialData container nor an AnnData query.")


def materialise_obs(
    pair: AlignPair,
    result: AlignResult,
    *,
    key_added: str | None,
    inplace: bool,
) -> SpatialData | AnnData | None:
    """Bake the transform into a *new* AnnData living in the reference cs.

    For affine results we apply the matrix; for :class:`ObsDisplacement` we
    add the deltas.  When the source query lives inside a SpatialData, the new
    AnnData is registered as ``sdata.tables[key_added]``; otherwise it is
    returned directly.

    .. note::

       The returned AnnData **shares** ``X`` and ``var`` with the source
       query by reference to avoid copying potentially-large expression
       matrices.  Mutating one will affect the other.  Call
       ``.copy()`` on the result if you need full independence.
    """
    if not isinstance(pair.query, AnnData):
        raise TypeError("materialise_obs only works for `align_obs`; `pair.query` must be an AnnData.")
    if "spatial" not in pair.query.obsm:
        raise KeyError("Source AnnData has no `obsm['spatial']` to warp.")

    src_coords = np.asarray(pair.query.obsm["spatial"])
    new_coords = result.transform.apply(src_coords)

    # Slim copy: share X/var/obs structurally and only rewrite obsm so we
    # don't pay the cost of deep-copying potentially-large layers/obsp.
    new_obsm = dict(pair.query.obsm)
    new_obsm["spatial"] = new_coords
    new_uns = dict(pair.query.uns)
    new_uns["align"] = {
        "source_query_key": pair.query_element_key,
        "ref_key": pair.ref_element_key,
        **result.metadata,
    }
    new_adata = AnnData(
        X=pair.query.X,
        obs=pair.query.obs.copy(),
        var=pair.query.var,
        obsm=new_obsm,
        uns=new_uns,
    )

    if pair.query_container is not None:
        if key_added is None:
            raise ValueError("`key_added` is required when `output_mode='obs'` and the query is a SpatialData.")
        sdata = pair.query_container if inplace else _shallow_copy_sdata(pair.query_container)
        from spatialdata.models import TableModel

        sdata.tables[key_added] = TableModel.parse(new_adata)
        return None if inplace else sdata

    return new_adata


def _get_element(sdata: SpatialData, key: str) -> object:
    """Look up a spatial element by name across all element types."""
    for attr in ("images", "labels", "points", "shapes", "tables"):
        store = getattr(sdata, attr)
        if key in store:
            return store[key]
    raise KeyError(f"Element {key!r} not found in the SpatialData object.")


def _shallow_copy_sdata(sdata: SpatialData) -> SpatialData:
    """Shallow copy of a SpatialData object for ``inplace=False`` writeback paths.

    Uses :meth:`SpatialData.subset` over every element so tables and
    ``attrs`` propagate the same way spatialdata's own subsetting handles
    them, rather than reconstructing via the ``__init__`` constructor.
    """
    element_names = [name for _, name, _ in sdata.gen_elements()]
    return sdata.subset(element_names, filter_tables=False, include_orphan_tables=True)
