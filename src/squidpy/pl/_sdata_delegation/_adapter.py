from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations import Identity, Scale, set_transformation

from squidpy._constants._pkg_constants import Key

from ._intent import Intent

_REGION_KEY = "_sq_region"
_INSTANCE_KEY = "_sq_instance"


def _shapes_name(library_id: str) -> str:
    return f"{library_id}_spots"


def _image_name(library_id: str) -> str:
    return f"{library_id}_image"


def _labels_name(library_id: str) -> str:
    return f"{library_id}_labels"


def _points_name(library_id: str) -> str:
    return f"{library_id}_points"


def _build_shapes(adata_sub: AnnData, spatial_key: str, diameter_fullres: float) -> ShapesModel:
    coords = np.asarray(adata_sub.obsm[spatial_key], dtype=float)
    return ShapesModel.parse(coords, geometry=0, radius=float(diameter_fullres) / 2.0)


def _build_points(adata_sub: AnnData, spatial_key: str) -> PointsModel:
    coords = np.asarray(adata_sub.obsm[spatial_key], dtype=float)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    return PointsModel.parse(df)


def _build_image(image_array: np.ndarray, scalef: float, coordinate_system: str) -> Image2DModel:
    arr = np.asarray(image_array)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected image shape {arr.shape}; need 2D or 3D with channel last/first.")
    image = Image2DModel.parse(arr, dims=("c", "y", "x"))
    if scalef != 1.0:
        set_transformation(
            image, Scale([1.0 / scalef, 1.0 / scalef], axes=("x", "y")), to_coordinate_system=coordinate_system
        )
    else:
        set_transformation(image, Identity(), to_coordinate_system=coordinate_system)
    return image


def _build_labels(mask: np.ndarray, scalef: float, coordinate_system: str) -> Labels2DModel:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"Labels mask must be 2D, got shape {arr.shape}.")
    labels = Labels2DModel.parse(arr, dims=("y", "x"))
    if scalef != 1.0:
        set_transformation(
            labels, Scale([1.0 / scalef, 1.0 / scalef], axes=("x", "y")), to_coordinate_system=coordinate_system
        )
    else:
        set_transformation(labels, Identity(), to_coordinate_system=coordinate_system)
    return labels


def _make_tmp_sdata(adata: AnnData, intent: Intent, spatial_key: str = "spatial") -> SpatialData:
    """Build a transient SpatialData from a Visium-style AnnData based on the captured Intent.

    One coordinate system per library. Each library contributes either a shapes element
    (Visium spots, Path 1/2) or a labels element (segmentation masks, Path 3), an optional
    image, and a shared table annotating the region via _REGION_KEY / _INSTANCE_KEY.

    For shapes-mode, _INSTANCE_KEY is arange(n_obs). For labels-mode, _INSTANCE_KEY
    must equal adata.obs[seg_cell_id] so render_labels can match each mask label to a
    table row.
    """
    images: dict[str, object] = {}
    shapes: dict[str, object] = {}
    labels: dict[str, object] = {}
    points: dict[str, object] = {}
    region_to_instance: dict[str, AnnData] = {}

    library_key = intent.data.library_key
    library_ids = intent.data.library_ids
    size_key = intent.data.size_key or Key.uns.size_key
    img_res_key = intent.data.img_res_key
    seg_cell_id = intent.data.seg_cell_id

    needs_shapes = intent.data.needs_shapes
    needs_labels = intent.data.needs_labels
    needs_points = intent.data.needs_points
    n_elements = sum([needs_shapes, needs_labels, needs_points])
    if n_elements != 1:
        raise ValueError(
            "Intent must request exactly one of needs_shapes / needs_labels / needs_points; "
            f"got needs_shapes={needs_shapes}, needs_labels={needs_labels}, needs_points={needs_points}."
        )

    for lib in library_ids:
        if library_key is not None and library_key in adata.obs.columns:
            mask = adata.obs[library_key].astype(str).values == lib
            adata_sub = adata[mask].copy()
        else:
            adata_sub = adata.copy()

        try:
            spatial_meta = adata.uns[Key.uns.spatial][lib]
        except KeyError as e:
            raise KeyError(f"Library {lib!r} not found in adata.uns[{Key.uns.spatial!r}].") from e

        if needs_shapes:
            diameter = float(spatial_meta["scalefactors"][size_key])
            shapes_element = _build_shapes(adata_sub, spatial_key, diameter)
            set_transformation(shapes_element, Identity(), to_coordinate_system=lib)
            region_name = _shapes_name(lib)
            shapes[region_name] = shapes_element
        elif needs_points:
            points_element = _build_points(adata_sub, spatial_key)
            set_transformation(points_element, Identity(), to_coordinate_system=lib)
            region_name = _points_name(lib)
            points[region_name] = points_element
        elif needs_labels:
            seg_key = Key.uns.image_seg_key
            if seg_key not in spatial_meta["images"]:
                raise KeyError(f"Library {lib!r} has no '{seg_key}' image in uns[spatial][{lib}][images].")
            scalef_lookup = f"tissue_{seg_key}_scalef"
            seg_scalef = float(spatial_meta["scalefactors"].get(scalef_lookup, 1.0))
            labels_element = _build_labels(spatial_meta["images"][seg_key], seg_scalef, lib)
            region_name = _labels_name(lib)
            labels[region_name] = labels_element
        else:
            raise ValueError("Intent requires either shapes or labels; got neither.")

        if intent.data.needs_image and img_res_key is not None:
            scalef_lookup = f"tissue_{img_res_key}_scalef"
            scalef = float(spatial_meta["scalefactors"].get(scalef_lookup, 1.0))
            image_array = spatial_meta["images"][img_res_key]
            images[_image_name(lib)] = _build_image(image_array, scalef, lib)

        adata_sub.obs[_REGION_KEY] = region_name
        adata_sub.obs[_REGION_KEY] = adata_sub.obs[_REGION_KEY].astype("category")
        if needs_labels and seg_cell_id is not None:
            adata_sub.obs[_INSTANCE_KEY] = adata_sub.obs[seg_cell_id].astype(int).to_numpy()
        else:
            adata_sub.obs[_INSTANCE_KEY] = np.arange(adata_sub.n_obs)
        region_to_instance[region_name] = adata_sub

    if len(region_to_instance) == 1:
        combined = next(iter(region_to_instance.values()))
    else:
        # pairwise=True preserves per-library obsp (connectivity matrices) as a block-diagonal.
        # Without it, ad.concat drops obsp silently and render_graph can't find the keys.
        combined = ad.concat(list(region_to_instance.values()), join="outer", merge="same", pairwise=True)

    combined.obs[_REGION_KEY] = combined.obs[_REGION_KEY].astype("category")
    table = TableModel.parse(
        combined,
        region=list(region_to_instance.keys()),
        region_key=_REGION_KEY,
        instance_key=_INSTANCE_KEY,
    )

    return SpatialData(images=images, shapes=shapes, labels=labels, points=points, tables={"table": table})
