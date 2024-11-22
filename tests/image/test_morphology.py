from __future__ import annotations

import time
import typing
from pyexpat import features

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
import skimage.measure
import spatialdata as sd
from anndata import AnnData
from skimage.measure import regionprops
from spatialdata import SpatialData
from spatialdata.datasets import blobs, raccoon

import squidpy as sq

# noinspection PyProtectedMember
from squidpy.im import ImageContainer, _measurements, quantify_morphology

# noinspection PyProtectedMember
from squidpy.im._feature import (
    _get_region_props,
    _get_table_key,
    _sdata_image_features_helper,
    calculate_image_features,
)

# noinspection PyProtectedMember
from squidpy.im._measurements import (
    border_occupied_factor,
    calculate_histogram,
    calculate_image_feature,
    calculate_image_texture,
    calculate_quantiles,
)


@pytest.fixture
def sdata_racoon() -> SpatialData:
    return raccoon()


@pytest.fixture
def sdata_blobs() -> SpatialData:
    return blobs()


@pytest.fixture
def morphology_methods() -> list[str]:
    return ["label", "area", "eccentricity", "circularity", "granularity", "border_occupied_factor"]


@pytest.mark.xfail(
    raises=AssertionError, reason="calling skimage.measure.label on segmentation produces additional labels"
)
class TestSkimageBackend:
    def test_label_bug_present(self, sdata_blobs):
        npt.assert_array_equal(
            np.unique(skimage.measure.label(sdata_blobs["blobs_labels"])),
            np.unique(sdata_blobs["blobs_labels"]),
        )


def dummy_callable():
    pass


@pytest.fixture
def malformed_morphology_methods() -> dict[str, typing.Any]:
    methods = {
        "wrong_container": [("label", "area"), "label,area"],
        "wrong_method_type": [["test", dummy_callable, 42.42], ["test", dummy_callable, 42]],
    }
    return methods


class TestMorphology:
    # @pytest.mark.parametrize(
    #     "sdata,methods",
    #     pytest.param(
    #         sdata_blobs, ("label", "area"), id="tuple"
    #     ),
    #     pytest.param(
    #         sdata_blobs, "label,area", id="string"
    #     ),
    # )
    def test_sanity_method_list(self, sdata_blobs, malformed_morphology_methods):
        with pytest.raises(ValueError, match="Argument `methods` must be a list of strings."):
            for methods in malformed_morphology_methods["wrong_container"]:
                sq.im.quantify_morphology(sdata=sdata_blobs, label="blobs_labels", image="blobs_image", methods=methods)

    # @pytest.mark.parametrize(
    #     "sdata,methods",
    #     pytest.param(sdata_blobs, ["test", dummy_callable, 42.42], id="float"),
    #     pytest.param(sdata_blobs, ["test", dummy_callable, 42], id="int"),
    # )
    def test_sanity_method_list_types(self, sdata_blobs, malformed_morphology_methods):
        with pytest.raises(ValueError, match="All elements in `methods` must be strings or callables."):
            for methods in malformed_morphology_methods["wrong_method_type"]:
                sq.im.quantify_morphology(sdata=sdata_blobs, label="blobs_labels", image="blobs_image", methods=methods)

    def test_get_table_key_no_annotators(self, sdata_blobs):
        label = "blobs_multiscale_labels"
        with pytest.raises(ValueError, match=f"No tables automatically detected in `sdata` for {label}"):
            _get_table_key(sdata=sdata_blobs, label=label, kwargs={})

    def test_get_table_key_multiple_annotators(self, sdata_blobs):
        sdata_blobs.tables["multi_table"] = sd.deepcopy(sdata_blobs["table"])
        label = "blobs_labels"
        with pytest.raises(ValueError, match=f"Multiple tables detected in `sdata` for {label}"):
            _get_table_key(sdata=sdata_blobs, label=label, kwargs={})

    def test_quantify_morphology_granularity(self, sdata_blobs):
        granular_spectrum_length = 16
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            methods=["label", "granularity"],
            split_by_channels=True,
        )

        columns = sdata_blobs["table"].obsm["morphology"].columns
        for channel in range(3):
            for granular_spectrum in range(granular_spectrum_length):
                assert f"granularity_ch{channel}_{granular_spectrum}" in columns

    def test_quantify_morphology_border_occupied_factor(self, sdata_blobs):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            methods=["label", "border_occupied_factor"],
            split_by_channels=True,
        )
        assert "border_occupied_factor" in sdata_blobs["table"].obsm["morphology"].columns

    def test__get_region_props(self, sdata_blobs):
        region_props = _get_region_props(
            label_element=sdata_blobs["blobs_labels"],
            image_element=sdata_blobs["blobs_image"],
            props=["label", "area", "eccentricity"],
        )
        assert len(region_props) == len(sdata_blobs["table"].obs)

    def test_quantify_morphology_callables(self, sdata_blobs, morphology_methods):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            methods=morphology_methods,
            split_by_channels=True,
        )

        assert "morphology" in sdata_blobs["table"].obsm.keys()
        assert isinstance(sdata_blobs["table"].obsm["morphology"], pd.DataFrame)
        assert "circularity" in sdata_blobs["table"].obsm["morphology"].columns
        assert "border_occupied_factor" in sdata_blobs["table"].obsm["morphology"].columns

    def test_quantify_morphology_all(self, sdata_blobs):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            split_by_channels=True,
        )

        for name in _measurements._all_regionprops_names():
            assert any([column.startswith(name) for column in sdata_blobs["table"].obsm["morphology"].columns])
        # for column in sdata_blobs["table"].obsm["morphology"].columns:
        #     assert any(column.startswith(name) for name in _measurements._all_regionprops_names())

    @pytest.mark.xfail(
        raises=ValueError,
        reason="For the moment, there is no association " "between the multiscale labels and a table in blobs.",
    )
    def test_quantify_morphology_multiscale(self, sdata_blobs, morphology_methods):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_multiscale_labels",
            image="blobs_multiscale_image",
            methods=morphology_methods,
            split_by_channels=True,
        )

    def test_quantify_morphology_cp_measure(self, sdata_blobs):
        sq.im.quantify_morphology(
            sdata=sdata_blobs,
            label="blobs_labels",
            image="blobs_image",
            split_by_channels=True,
            methods=["label", "radial_distribution"],
        )

        print(sdata_blobs["table"].obsm["morphology"].columns)


# TODO Remove for release
@pytest.fixture
def sdata_merfish() -> SpatialData:
    zarr_path = "./data/merfish.zarr"
    return sd.read_zarr(zarr_path)


# TODO Remove for release
@pytest.fixture
def sdata_mibitof() -> SpatialData:
    zarr_path = "./data/mibitof.zarr"
    return sd.read_zarr(zarr_path)


class TestMorphologyPerformance:
    def test_performance(self, sdata_mibitof, morphology_methods):
        start_time = time.perf_counter()
        sq.im.quantify_morphology(
            sdata=sdata_mibitof,
            label="point8_labels",
            image="point8_image",
            methods=None,
            split_by_channels=True,
        )
        end_time = time.perf_counter()
        assert end_time - start_time < 60


class TestMeasurements:
    def test_border_occupied_factor(self):
        label_image = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0, 0, 0, 0],
                [0, 1, 1, 1, 2, 2, 2, 0],
                [0, 1, 1, 1, 2, 2, 2, 0],
                [0, 0, 3, 3, 2, 2, 2, 0],
                [0, 0, 3, 3, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        expected = {1: 3 / 8, 2: 3 / 8, 3: 2 / 4}
        actual = border_occupied_factor(label_image)
        assert actual == expected

    def test_radial_distribution(self, sdata_mibitof):
        pixels = np.random.randint(100, size=64**2).reshape((64, 64))
        mask = np.zeros_like(pixels, dtype=bool)
        mask[2:-3, 2:-3] = True

        result = _measurements.radial_distribution(
            labels=pixels,
            pixels=mask,
        )

        # result = _measurements.radial_distribution(
        #     labels=np.array(sdata_mibitof["point8_labels"]),
        #     pixels=np.array(sdata_mibitof["point8_image"]),
        # )

        print(result)

    def test_cp_measure(self, sdata_blobs):
        from cp_measure.bulk import get_fast_measurements

        measurements = get_fast_measurements()

        size = 200
        rng = np.random.default_rng(42)
        pixels = rng.integers(low=0, high=10, size=(size, size))

        masks = np.zeros_like(pixels)
        masks[5:-6, 5:-6] = 1

        results = {}
        for measurement_name, measurement in measurements.items():
            results[measurement_name] = measurement(masks, pixels)

        print(results)


@pytest.fixture()
def blobs_as_image_container(sdata_blobs: SpatialData) -> ImageContainer:
    img_layer_name = "blobs_image"
    seg_layer_name = "blobs_labels"
    img = ImageContainer(sdata_blobs[img_layer_name].to_numpy(), layer=img_layer_name)
    img.add_img(sdata_blobs[seg_layer_name].to_numpy(), layer=seg_layer_name)

    return img


@pytest.fixture()
def blobs_as_adata(sdata_blobs: SpatialData) -> AnnData:
    s_adata = sdata_blobs.tables["table"]
    # print(s_adata.uns)
    return s_adata


@pytest.fixture()
def mibitof_as_image_container(sdata_mibitof: SpatialData) -> ImageContainer:
    img_layer_name = "point8_image"
    seg_layer_name = "point8_labels"
    img = ImageContainer(sdata_mibitof[img_layer_name].to_numpy(), layer=img_layer_name)
    img.add_img(sdata_mibitof[seg_layer_name].to_numpy(), layer=seg_layer_name)
    return img


@pytest.fixture()
def mibitof_as_adata(sdata_mibitof: SpatialData) -> AnnData:
    s_adata = sdata_mibitof.tables["table"]
    s_adata.uns["spatial"] = {"point8_labels": {"scalefactors": {"spot_diameter_fullres": 7.0}}}
    return s_adata


@pytest.fixture()
def visium_adata() -> AnnData:
    return sq.datasets.visium_fluo_adata_crop()


@pytest.fixture()
def visium_img() -> ImageContainer:
    img = sq.datasets.visium_fluo_image_crop()
    sq.im.segment(
        img=img,
        layer="image",
        layer_added="segmented_watershed",
        method="watershed",
        channel=0,
    )
    return img


@pytest.fixture()
def calc_im_features_kwargs() -> dict[str, typing.Any]:
    kwargs = {
        # "adata": visium_adata,
        # "img": visium_img,
        # "features": features,
        "layer": "image",
        # "library_id": "point8_labels",
        "key_added": "segmentation_features",
        "features_kwargs": {
            "segmentation": {
                "label_layer": "segmented_watershed",
                "props": ["label", "area", "mean_intensity"],
                "channels": [1, 2],
            }
        },
        "copy": True,
        "mask_circle": True,
    }
    return kwargs


class TestMorphologyImageFeaturesCompatibility:
    def test_quantiles(
        self, calc_im_features_kwargs: dict[str, typing.Any], visium_adata: AnnData, visium_img: ImageContainer
    ):
        calc_im_features_kwargs.update(
            {
                "features": ["summary"],
                "adata": visium_adata,
                "img": visium_img,
            }
        )

        # expected = calculate_image_features(
        #     **calc_im_features_kwargs
        # )
        actual = calculate_quantiles(
            mask=visium_img["segmented_watershed"].to_numpy(), pixels=visium_img["image"].to_numpy()
        )
        print(actual)

    def test_calculate_image_features(self, visium_adata: AnnData, visium_img: ImageContainer):
        actual = calculate_image_feature(
            # feature=calculate_histogram,
            feature=calculate_image_texture,
            mask=visium_img["segmented_watershed"].to_numpy(),
            pixels=visium_img["image"].to_numpy(),
        )
        print(actual)

    def test_calculate_image_features_performance(self, visium_adata: AnnData, visium_img: ImageContainer):
        start_time = time.perf_counter()
        props = regionprops(
            label_image=visium_img["segmented_watershed"].to_numpy()[:, :, 0, 0],
            intensity_image=visium_img["image"].to_numpy()[:, :, 0, 0],
            extra_properties=calculate_histogram,
        )
        actual = {prop.label: prop.calculate_histogram for prop in props}
        end_time = time.perf_counter()
        # calc_im_features_result = calculate_image_feature(
        #     feature=calculate_histogram,
        #     mask=visium_img["segmented_watershed"].to_numpy(),
        #     pixels=visium_img["image"].to_numpy()
        # )
        # end_time = time.perf_counter()
        # assert end_time - start_time < 60

        print(end_time - start_time)

    def test_im_features_morphology_equivalence(
        self,
        # blobs_as_adata: AnnData, blobs_as_image_container: ImageContainer,
        # mibitof_as_adata: AnnData, mibitof_as_image_container: ImageContainer,
        calc_im_features_kwargs: dict[str, typing.Any],
        visium_adata: AnnData,
        visium_img: ImageContainer,
        # adata: AnnData
    ):
        # print(adata.uns.spatial)
        # expected = calculate_image_features(
        #     adata=mibitof_as_adata,
        #     img=mibitof_as_image_container,
        #     layer="point8_labels",
        #     library_id="point8_labels",
        #     features=features,
        #     key_added="foo",
        #     copy=True,
        #     features_kwargs={"segmentation": {"label_layer": "point8_labels", "intensity_layer": "point8_image"}},
        # )

        calc_im_features_kwargs.update(
            {
                "features": ["texture", "summary", "histogram", "segmentation"],
                "adata": visium_adata,
                "img": visium_img,
            }
        )

        expected = calculate_image_features(**calc_im_features_kwargs)

        actual = _sdata_image_features_helper(**calc_im_features_kwargs)
        # morphology = quantify_morphology()

        pd.testing.assert_frame_equal(actual, expected)

    # def test_helper_equivalence(self, adata: AnnData, cont: ImageContainer):
    #     expected = _calculate_image_features_helper(
    #         adata=adata,
    #         img=cont["image"],
    #     )
