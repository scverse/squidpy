from __future__ import annotations

import typing
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

import squidpy._utils
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
    # Add additional measurements here that handle individual regions
    rp_extra_methods = [
        circularity,
        _measurements.granularity,
        _measurements.radial_distribution,
    ] + extra_methods

    # Add additional measurements here that calculate on the entire label image
    image_extra_methods = [
        _measurements.border_occupied_factor,
        _measurements.zernike,
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
@squidpy._utils.deprecated
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


def _sdata_image_features_helper(
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
    return None


def quantify_morphology(
    sdata: SpatialData,
    label: str,
    image: str,
    methods: list[str | Callable] | None = None,
    split_by_channels: bool = True,
    **kwargs: Any,
) -> pd.DataFrame | None:
    extra_methods, methods = _validate_methods(methods)

    for element in [label, image]:
        if element is not None and element not in sdata:
            raise KeyError(f"Key `{element}` not found in `sdata`.")

    table_key = _get_table_key(sdata, label, kwargs)

    region_key = sdata[table_key].uns["spatialdata_attrs"]["region_key"]
    if not np.any(sdata[table_key].obs[region_key] == label):
        raise ValueError(f"Label '{label}' not found in region key ({region_key}) column of sdata table '{table_key}'")

    instance_key = sdata[table_key].uns["spatialdata_attrs"]["instance_key"]

    image_element, label_element = _apply_transformations(image, label, sdata)

    region_props = _get_region_props(
        label_element,
        image_element,
        props=methods,
        extra_methods=extra_methods,
    )

    if split_by_channels:
        channels = image_element.c.values
        for col in region_props.columns:
            if isinstance(region_props[col].values[0], (int, float, np.integer, np.floating)):
                continue  # ignore single value returns

            is_processed = False

            # did the method return a list of values?
            if isinstance(region_props[col].values[0], (list, tuple)):
                is_processed = _extract_from_list_like(channels, col, is_processed, region_props)

            if isinstance(region_props[col].values[0], np.ndarray):
                is_processed = _extract_from_ndarray(channels, col, is_processed, region_props)
            if is_processed:
                region_props.drop(columns=[col], inplace=True)
            else:
                raise NotImplementedError(
                    f"Result of morphology method '{col}' cannot be interpreted, "
                    f"as its dtype ({type(region_props[col].values[0])} and shape "
                    f"does not conform to the expected shapes and dtypes."
                )

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


def _apply_transformations(image, label, sdata):
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
    return image_element, label_element


def _validate_methods(methods):
    if methods is None:
        # default case but without mutable argument as default value
        # noinspection PyProtectedMember
        methods = _measurements._all_regionprops_names()
    elif isinstance(methods, (str, Callable)):
        methods = [methods]
    if not isinstance(methods, list):
        raise ValueError("Argument `methods` must be a list of strings.")
    if not all(isinstance(method, (str, Callable)) for method in methods):
        raise ValueError("All elements in `methods` must be strings or callables.")
    if "label" not in methods:
        methods.append("label")
    extra_methods = []
    for method in methods:
        if callable(method):
            extra_methods.append(method)
            methods.remove(method)
    return extra_methods, methods


def _extract_from_ndarray(channels, col, is_processed, region_props):
    shape = region_props[col].values[0].shape
    if not all(val.shape == shape for val in region_props[col].values):
        raise ValueError(f"The results of the morphology method {col} have different shapes, this cannot be handled")
    # Handle cases like centroids which return coordinates for each region
    if len(shape) == 1 and shape[0] != len(channels):
        for prop_idx in range(shape[0]):
            region_props[f"{col}_{prop_idx}"] = [val[prop_idx] for val in region_props[col].values]
        is_processed = True
    # Handle cases like intensity which return one value per channel for each region
    if len(shape) == 1 and shape[0] == len(channels):
        for prop_idx in range(shape[0]):
            if region_props[col]:
                pass
            region_props[f"{col}_ch{prop_idx}"] = [val[prop_idx] for val in region_props[col].values]
        is_processed = True
    # Handle cases like granularity which return many values for each channel for each region
    if len(shape) == 2:
        if not shape[1] == len(channels):
            raise ValueError(
                f"Number of channels {len(channels)} do not match "
                f"the shape of the returned numpy arrays {shape} of the morphology method {col}. "
                f"It is expected that shape[1] should be equal to number of channels."
            )

        for channel_idx, channel in enumerate(channels):
            for prop_idx in range(shape[0]):
                region_props[f"{col}_ch{channel_idx}_{prop_idx}"] = [
                    val[prop_idx, channel_idx] for val in region_props[col].values
                ]

        is_processed = True
    return is_processed


def _extract_from_list_like(channels, col, is_processed, region_props):
    # are all lists of the length of the channel list?
    length = len(region_props[col].values[0])
    if not all(len(val) == length for val in region_props[col].values):
        raise ValueError(f"The results of the morphology method {col} have different lengths, this cannot be handled")
    if length == len(channels):
        for i, channel in enumerate(channels):
            region_props[f"{col}_ch{channel}"] = [val[i] for val in region_props[col].values]
        is_processed = True
    if length != len(channels):
        for prop_idx in range(length):
            region_props[f"{col}_{prop_idx}"] = [val[prop_idx] for val in region_props[col].values]
        is_processed = True
    return is_processed


def _get_table_key(sdata: sd.SpatialData, label: str, kwargs: dict[str, typing.Any]) -> str:
    table_key = kwargs.get("table_key", None)
    if table_key is None:
        tables = sd.get_element_annotators(sdata, label)
        if len(tables) > 1:
            raise ValueError(
                f"Multiple tables detected in `sdata` for {label}, "
                f"please specify a specific table with the `table_key` parameter"
            )
        if len(tables) == 0:
            raise ValueError(
                f"No tables automatically detected in `sdata` for {label}, "
                f"please specify a specific table with the `table_key` parameter"
            )
        table_key = next(iter(tables))
    return table_key


@squidpy._utils.deprecated
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
                # TODO: Potential bug here, should be label_layer and intensity layer must be specified via feature_kwargs
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
