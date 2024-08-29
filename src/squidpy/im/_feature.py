from __future__ import annotations

import warnings
from collections.abc import Generator, Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import spatialdata as sd
import xarray as xr
from anndata import AnnData
from scanpy import logging as logg
from skimage.measure import perimeter, regionprops
from spatialdata import SpatialData

from squidpy._constants._constants import ImageFeature
from squidpy._docs import d, inject_docs
from squidpy._utils import Signal, SigQueue, _get_n_cores, parallelize
from squidpy.gr._utils import _save_data
from squidpy.im import _measurements
from squidpy.im._container import ImageContainer

__all__ = ["calculate_image_features", "quantify_morphology"]

IntegerNDArrayType = TypeVar("IntegerNDArrayType", bound=npt.NDArray[np.integer[Any]])
FloatNDArrayType = TypeVar("FloatNDArrayType", bound=npt.NDArray[np.floating[Any]])
RegionPropsCallableType = Callable[
    [IntegerNDArrayType, FloatNDArrayType], Union[Union[int, float, list[Union[int, float]]]]
]

RegionPropsImageCallableType = Callable[[IntegerNDArrayType, FloatNDArrayType], dict[str, Union[int, float]]]


def circularity(regionmask: IntegerNDArrayType) -> float:
    """
    Calculate the circularity of the region.

    :param regionmask: Region properties object
    :return: circularity of the region
    """
    perim = perimeter(regionmask)
    if perim == 0:
        return 0
    area = np.sum(regionmask)
    return float((4 * np.pi * area) / (perim**2))


def _get_region_props(
    label_element: xr.DataArray,
    image_element: xr.DataArray,
    props: list[str] | None = None,
    extra_methods: list[RegionPropsCallableType] | None = None,
) -> pd.DataFrame:
    if not extra_methods:
        extra_methods = []
    if props is None:
        # if we didn't get any properties, we'll do the bare minimum
        props = ["label"]

    np_rgb_image = image_element.values.transpose(1, 2, 0)  # (c, y, x) -> (y, x, c)

    # Add custom extra methods here
    rp_extra_methods = [
        circularity,
        _measurements.granularity,  # <--- Add additional measurements here that handle individual regions
    ] + extra_methods

    image_extra_methods = [
        _measurements.border_occupied_factor  # <--- Add additional measurements here that calculate on the entire label image
    ]  # type: list[RegionPropsImageCallableType]
    image_extra_methods = {method.__name__: method for method in image_extra_methods}

    # can't use regionprops_table because it only returns int
    regions = regionprops(
        label_image=label_element.values,
        intensity_image=np_rgb_image,
        extra_properties=rp_extra_methods,
    )
    # dynamically extract specified properties and create a df
    extracted_props = {prop: [] for prop in props + [e.__name__ for e in extra_methods]}  # type: dict[str, list[int | float]]
    for prop in props + [e.__name__ for e in extra_methods]:
        if prop in image_extra_methods:
            im_extra_result = image_extra_methods[prop](label_element.values, np_rgb_image)
            for region in regions:
                extracted_props[prop].append(im_extra_result.get(region.label))
            continue

        for region in regions:
            try:
                extracted_props[prop].append(getattr(region, prop))
            except AttributeError as e:
                raise ValueError(f"Property '{prop}' is not available in the region properties.") from e

    return pd.DataFrame(extracted_props)


def _subset_image_using_label(
    label_element: xr.DataArray, image_element: xr.DataArray
) -> Generator[tuple[int, xr.DataArray], None, None]:
    """
    A generator that extracts subsets of the RGB image based on the bitmap.

    :param label_element: xarray.DataArray with cell identifiers
    :param image_element: xarray.DataArray with RGB image data
    :yield: Subsets of the RGB image corresponding to each cell in the bitmap
    """
    unique_cells = np.unique(label_element.values)

    for cell_id in unique_cells:
        if cell_id == 0:
            # Assuming 0 is the background or non-cell area, skip it
            continue

        cell_mask = xr.DataArray(
            label_element.values == cell_id,
            dims=label_element.dims,
            coords=label_element.coords,
        )

        subset = image_element.where(cell_mask, drop=True)

        yield cell_id, subset


@d.dedent
@inject_docs(f=ImageFeature)
def calculate_image_features(
    adata: AnnData,
    img: ImageContainer,
    layer: str | None = None,
    library_id: str | Sequence[str] | None = None,
    features: str | Sequence[str] = ImageFeature.SUMMARY.s,
    features_kwargs: Mapping[str, Mapping[str, Any]] = MappingProxyType({}),
    key_added: str = "img_features",
    copy: bool = False,
    n_jobs: int | None = None,
    backend: str = "loky",
    show_progress_bar: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | None:
    """
    Calculate image features for all observations in ``adata``.

    Parameters
    ----------
    %(adata)s
    %(img_container)s
    %(img_layer)s
    %(img_library_id)s
    features
        Features to be calculated. Valid options are:

        - `{f.TEXTURE.s!r}` - summary stats based on repeating patterns
          :meth:`squidpy.im.ImageContainer.features_texture`.
        - `{f.SUMMARY.s!r}` - summary stats of each image channel
          :meth:`squidpy.im.ImageContainer.features_summary`.
        - `{f.COLOR_HIST.s!r}` - counts in bins of image channel's histogram
          :meth:`squidpy.im.ImageContainer.features_histogram`.
        - `{f.SEGMENTATION.s!r}` - stats of a cell segmentation mask
          :meth:`squidpy.im.ImageContainer.features_segmentation`.
        - `{f.CUSTOM.s!r}` - extract features using a custom function
          :meth:`squidpy.im.ImageContainer.features_custom`.

    features_kwargs
        Keyword arguments for the different features that should be generated, such as
        ``{{ {f.TEXTURE.s!r}: {{ ... }}, ... }}``.
    key_added
        Key in :attr:`anndata.AnnData.obsm` where to store the calculated features.
    %(copy)s
    %(parallelize)s
    kwargs
        Keyword arguments for :meth:`squidpy.im.ImageContainer.generate_spot_crops`.

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` where columns correspond to the calculated features.

    Otherwise, modifies the ``adata`` object with the following key:

        - :attr:`anndata.AnnData.uns` ``['{{key_added}}']`` - the above mentioned dataframe.

    Raises
    ------
    ValueError
        If a feature is not known.
    """
    layer = img._get_layer(layer)
    if isinstance(features, (str, ImageFeature)):
        features = [features]
    features = sorted({ImageFeature(f).s for f in features})

    n_jobs = _get_n_cores(n_jobs)
    start = logg.info(f"Calculating features `{list(features)}` using `{n_jobs}` core(s)")

    res = parallelize(
        _calculate_image_features_helper,
        collection=adata.obs_names,
        extractor=pd.concat,
        n_jobs=n_jobs,
        backend=backend,
        show_progress_bar=show_progress_bar,
    )(
        adata,
        img,
        layer=layer,
        library_id=library_id,
        features=features,
        features_kwargs=features_kwargs,
        **kwargs,
    )

    if copy:
        logg.info("Finish", time=start)
        return res

    _save_data(adata, attr="obsm", key=key_added, data=res, time=start)


def quantify_morphology(
    sdata: SpatialData,
    label: str | None = None,
    image: str | None = None,
    methods: list[str | Callable] | None = None,
    split_by_channels: bool = False,
    **kwargs: Any,
) -> pd.DataFrame | None:
    if label is None and image is None:
        raise ValueError("Either `label` or `image` must be provided.")

    if image is not None and label is None:
        raise ValueError("If `image` is provided, a `label` with matching segmentation borders must be provided.")

    if methods is None:
        # default case but without mutable argument as default value
        methods = ["label", "area", "eccentricity", "perimeter", "sphericity"]
    elif isinstance(methods, (str, Callable)):
        methods = [methods]

    if not isinstance(methods, list):
        raise ValueError("Argument `methods` must be a list of strings.")

    if not all(isinstance(method, (str, Callable)) for method in methods):
        raise ValueError("All elements in `methods` must be strings or callables.")

    if "label" not in methods:
        methods = ["label"].extend(methods)

    extra_methods = []
    for method in methods:
        if callable(method):
            extra_methods.append(method)
            methods.remove(method)

    for element in [label, image]:
        if element is not None and element not in sdata:
            raise KeyError(f"Key `{element}` not found in `sdata`.")

    table_key = kwargs.get("table_key", None)
    if table_key is None:
        tables = sd.get_element_annotators(sdata, label)
        if len(tables) > 1:
            raise ValueError(
                f"Multiple tables detected in `sdata` for {label}, "
                f"please specify a specific table with the `table_key` parameter"
            )
        table_key = next(iter(tables))

    region_key = sdata[table_key].uns["spatialdata_attrs"]["region_key"]
    if not np.any(sdata[table_key].obs[region_key] == label):
        raise ValueError(f"Label {label} not found in region key ({region_key}) column of sdata table `{table_key}`")

    instance_key = sdata[table_key].uns["spatialdata_attrs"]["instance_key"]

    label_transform = sdata[label].transform
    image_transform = sdata[image].transform

    for transform in [label_transform, image_transform]:
        if len(transform) != 1:
            raise ValueError("More than one coordinate system detected")

    coord_sys_label = next(iter(label_transform))
    coord_sys_image = next(iter(image_transform))

    if coord_sys_label != coord_sys_image:
        raise ValueError(f"Coordinate system do not match! label: {coord_sys_label}, image: {coord_sys_image}")

    # from here on we should be certain that we have a label
    label_element = sdata.transform_element_to_coordinate_system(label, coord_sys_label)
    image_element = sdata.transform_element_to_coordinate_system(image, coord_sys_image) if image is not None else None

    region_props = _get_region_props(
        label_element,
        image_element,
        props=methods,
        extra_methods=extra_methods,
    )

    if split_by_channels:
        channels = image_element.c.values
        for col in region_props.columns:
            # did the method return a list of values?
            if isinstance(region_props[col].values[0], (list, tuple, np.ndarray)):
                # are all lists of the length of the channel list?
                if all(len(val) == len(channels) for val in region_props[col].values):
                    for i, channel in enumerate(channels):
                        region_props[f"{col}_ch{channel}"] = [val[i] for val in region_props[col].values]
                    region_props.drop(columns=[col], inplace=True)

    region_props.rename(columns={"label": instance_key}, inplace=True)
    region_props[region_key] = label
    region_props.set_index([region_key, instance_key], inplace=True)

    results = sdata[table_key].obs[[region_key, instance_key]]
    results = results.join(region_props, how="left", on=[region_key, instance_key])

    # region_props = region_props.set_index("label", drop=True)
    # region_props.index.name = None
    # region_props.index = region_props.index.astype(str)

    sdata[table_key].obsm["morphology"] = results

    return region_props


def _calculate_image_features_helper(
    obs_ids: Sequence[str],
    adata: AnnData,
    img: ImageContainer,
    layer: str,
    library_id: str | Sequence[str] | None,
    features: list[ImageFeature],
    features_kwargs: Mapping[str, Any],
    queue: SigQueue | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    features_list = []
    for crop in img.generate_spot_crops(
        adata,
        obs_names=obs_ids,
        library_id=library_id,
        return_obs=False,
        as_array=False,
        **kwargs,
    ):
        if TYPE_CHECKING:
            assert isinstance(crop, ImageContainer)
        # load crop in memory to enable faster processing
        crop = crop.compute(layer)

        features_dict = {}
        for feature in features:
            feature = ImageFeature(feature)
            feature_kwargs = features_kwargs.get(feature.s, {})

            if feature == ImageFeature.TEXTURE:
                res = crop.features_texture(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.COLOR_HIST:
                res = crop.features_histogram(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.SUMMARY:
                res = crop.features_summary(layer=layer, **feature_kwargs)
            elif feature == ImageFeature.SEGMENTATION:
                res = crop.features_segmentation(intensity_layer=layer, **feature_kwargs)
            elif feature == ImageFeature.CUSTOM:
                res = crop.features_custom(layer=layer, **feature_kwargs)
            else:
                # should never get here
                raise NotImplementedError(f"Feature `{feature}` is not yet implemented.")

            features_dict.update(res)
        features_list.append(features_dict)

        if queue is not None:
            queue.put(Signal.UPDATE)

    if queue is not None:
        queue.put(Signal.FINISH)

    return pd.DataFrame(features_list, index=list(obs_ids))
