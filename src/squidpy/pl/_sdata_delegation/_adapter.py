from __future__ import annotations

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


def _table_name(library_id: str) -> str:
    return f"{library_id}_table"


def _build_shapes(adata_sub: AnnData, spatial_key: str, diameter_fullres: float) -> ShapesModel:
    coords = np.asarray(adata_sub.obsm[spatial_key], dtype=float)
    return ShapesModel.parse(coords, geometry=0, radius=float(diameter_fullres) / 2.0)


def _build_points(adata_sub: AnnData, spatial_key: str) -> PointsModel:
    coords = np.asarray(adata_sub.obsm[spatial_key], dtype=float)
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    return PointsModel.parse(df)


def _build_image(image_array, scalef: float, coordinate_system: str) -> Image2DModel:
    """Wrap an image as Image2DModel without materializing a dask-backed array.

    Uses np.moveaxis (NumPy and Dask compatible) instead of np.asarray+transpose,
    so a 100k x 100k Visium HD H&E stays lazy until render time.
    """
    if image_array.ndim == 3 and image_array.shape[-1] in (3, 4):
        arr = np.moveaxis(image_array, -1, 0)
    elif image_array.ndim == 2:
        arr = image_array[np.newaxis, ...]
    elif image_array.ndim == 3:
        arr = image_array
    else:
        raise ValueError(f"Unexpected image shape {image_array.shape}; need 2D or 3D.")
    image = Image2DModel.parse(arr, dims=("c", "y", "x"))
    transform = Scale([1.0 / scalef, 1.0 / scalef], axes=("x", "y")) if scalef != 1.0 else Identity()
    set_transformation(image, transform, to_coordinate_system=coordinate_system)
    return image


def _build_labels(mask, scalef: float, coordinate_system: str) -> Labels2DModel:
    if mask.ndim != 2:
        raise ValueError(f"Labels mask must be 2D, got shape {mask.shape}.")
    labels = Labels2DModel.parse(mask, dims=("y", "x"))
    transform = Scale([1.0 / scalef, 1.0 / scalef], axes=("x", "y")) if scalef != 1.0 else Identity()
    set_transformation(labels, transform, to_coordinate_system=coordinate_system)
    return labels


def _instance_ids(adata_sub: AnnData, kind: str, seg_cell_id: str | None) -> np.ndarray:
    if kind == "labels" and seg_cell_id is not None:
        return adata_sub.obs[seg_cell_id].astype(int).to_numpy()
    return np.arange(adata_sub.n_obs)


def _make_tmp_sdata(adata: AnnData, intent: Intent) -> SpatialData:
    """Build a transient SpatialData from a Visium-style AnnData based on the captured Intent.

    One coordinate system per library, and **one table per library**. Per-library tables
    avoid materializing a cross-library obsp via ad.concat(pairwise=True), which at Visium HD
    multi-library scale would be O(N_total^2). Each library's table annotates only its own
    element via _REGION_KEY / _INSTANCE_KEY, and render_* calls pass table_name=f'{lib}_table'.
    """
    images: dict[str, object] = {}
    shapes: dict[str, object] = {}
    labels: dict[str, object] = {}
    points: dict[str, object] = {}
    tables: dict[str, object] = {}

    library_key = intent.data.library_key
    library_ids = intent.data.library_ids
    spatial_key = intent.data.coordinate_system or Key.obsm.spatial
    size_key = intent.data.size_key or Key.uns.size_key
    img_res_key = intent.data.img_res_key
    seg_cell_id = intent.data.seg_cell_id
    kind = intent.data.element_kind

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

        if kind == "shapes":
            diameter = Key.uns.spot_diameter(adata, Key.uns.spatial, lib, spot_diameter_key=size_key)
            element = _build_shapes(adata_sub, spatial_key, diameter)
            set_transformation(element, Identity(), to_coordinate_system=lib)
            region_name = _shapes_name(lib)
            shapes[region_name] = element
        elif kind == "points":
            element = _build_points(adata_sub, spatial_key)
            set_transformation(element, Identity(), to_coordinate_system=lib)
            region_name = _points_name(lib)
            points[region_name] = element
        else:  # labels
            seg_key = Key.uns.image_seg_key
            if seg_key not in spatial_meta["images"]:
                raise KeyError(f"Library {lib!r} has no '{seg_key}' image in uns[spatial][{lib}][images].")
            scalef_lookup = f"tissue_{seg_key}_scalef"
            seg_scalef = float(spatial_meta["scalefactors"].get(scalef_lookup, 1.0))
            element = _build_labels(spatial_meta["images"][seg_key], seg_scalef, lib)
            region_name = _labels_name(lib)
            labels[region_name] = element

        if intent.data.needs_image and img_res_key is not None:
            scalef_lookup = f"tissue_{img_res_key}_scalef"
            scalef = float(spatial_meta["scalefactors"].get(scalef_lookup, 1.0))
            images[_image_name(lib)] = _build_image(spatial_meta["images"][img_res_key], scalef, lib)

        adata_sub.obs[_REGION_KEY] = pd.Categorical([region_name] * adata_sub.n_obs)
        adata_sub.obs[_INSTANCE_KEY] = _instance_ids(adata_sub, kind, seg_cell_id)
        tables[_table_name(lib)] = TableModel.parse(
            adata_sub,
            region=region_name,
            region_key=_REGION_KEY,
            instance_key=_INSTANCE_KEY,
        )

    return SpatialData(images=images, shapes=shapes, labels=labels, points=points, tables=tables)
