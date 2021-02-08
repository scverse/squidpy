from typing import Tuple, Sequence
from pytest_mock import MockerFixture
import pytest

from anndata import AnnData

import numpy as np
import pandas as pd

from squidpy.im._feature import calculate_image_features
from squidpy.im._container import ImageContainer
from squidpy._constants._constants import ImageFeature


class TestFeatureMixin:
    def test_container_empty(self):
        cont = ImageContainer()
        with pytest.raises(ValueError, match=r"The container is empty."):
            cont.features_summary("image")

    def test_invalid_layer(self, small_cont: ImageContainer):
        with pytest.raises(KeyError, match=r"Image layer `foobar` not found in"):
            small_cont.features_summary("foobar")

    def test_invalid_channels(self, small_cont: ImageContainer):
        with pytest.raises(ValueError, match=r"Channel `-1` is not in"):
            small_cont.features_summary("image", channels=-1)

    @pytest.mark.parametrize("quantiles", [(), (0.5,), (0.1, 0.9)])
    def test_summary_quantiles(self, small_cont: ImageContainer, quantiles: Tuple[float, ...]):
        if not len(quantiles):
            with pytest.raises(ValueError, match=r"No quantiles have been selected."):
                small_cont.features_summary("image", quantiles=quantiles, feature_name="foo", channels=(0, 1))
        else:
            features = small_cont.features_summary("image", quantiles=quantiles, feature_name="foo", channels=(0, 1))
            haystack = features.keys()

            assert isinstance(features, dict)
            for c in (0, 1):
                for agg in ("mean", "std"):
                    assert f"foo_ch-{c}_{agg}" in haystack, haystack
                for q in quantiles:
                    assert f"foo_ch-{c}_quantile-{q}" in haystack, haystack

    @pytest.mark.parametrize("bins", [5, 10, 20])
    def test_histogram_bins(self, small_cont: ImageContainer, bins: int):
        features = small_cont.features_histogram("image", bins=bins, feature_name="histogram", channels=(0,))

        assert isinstance(features, dict)
        haystack = features.keys()

        for c in (0,):
            for b in range(bins):
                assert f"histogram_ch-{c}_bin-{b}" in features, haystack

    @pytest.mark.parametrize("props", [(), ("contrast", "ASM")])
    def test_textures_props(self, small_cont: ImageContainer, props: Sequence[str]):
        if not len(props):
            with pytest.raises(ValueError, match=r"No properties have been selected."):
                small_cont.features_texture("image", feature_name="foo", props=props)
        else:
            features = small_cont.features_texture("image", feature_name="foo", props=props)
            haystack = features.keys()

            for prop in props:
                assert any(f"{prop}_dist" in h for h in haystack), haystack

    @pytest.mark.parametrize("angles", [(), (0, 0.5 * np.pi)])
    def test_textures_angles(self, small_cont: ImageContainer, angles: Sequence[float]):
        if not len(angles):
            with pytest.raises(ValueError, match=r"No angles have been selected."):
                small_cont.features_texture("image", feature_name="foo", angles=angles)
        else:
            features = small_cont.features_texture("image", feature_name="foo", angles=angles)
            haystack = features.keys()

            for a in angles:
                assert any(f"angle-{a:.2f}" in h for h in haystack), haystack

    @pytest.mark.parametrize("distances", [(), (1, 2, 10)])
    def test_textures_distances(self, small_cont: ImageContainer, distances: Sequence[int]):
        if not len(distances):
            with pytest.raises(ValueError, match=r"No distances have been selected."):
                small_cont.features_texture("image", feature_name="foo", distances=distances)
        else:
            features = small_cont.features_texture("image", feature_name="foo", distances=distances)
            haystack = features.keys()

            for d in distances:
                assert any(f"dist-{d}" in h for h in haystack), haystack

    def test_segmentation_invalid_props(self, small_cont: ImageContainer):
        with pytest.raises(ValueError, match=r"Invalid property `foobar`. Valid properties are"):
            small_cont.features_segmentation("image", feature_name="foo", props=["foobar"])

    def test_segmentation_label(self, small_cont_seg: ImageContainer):
        features = small_cont_seg.features_segmentation("image", feature_name="foo", props=["label"])

        assert isinstance(features, dict)
        assert "foo_label" in features
        assert features["foo_label"] == 254

    def test_segmentation_centroid(self, small_cont_seg: ImageContainer):
        features = small_cont_seg.features_segmentation(
            label_layer="segmented", intensity_layer=None, feature_name="foo", props=["centroid"]
        )

        assert isinstance(features, dict)
        assert "foo_centroid" in features
        assert isinstance(features["foo_centroid"], np.ndarray)
        assert features["foo_centroid"].ndim == 2

    @pytest.mark.parametrize("props", [(), ("extent",), ("area", "solidity", "mean_intensity")])
    def test_segmentation_props(self, small_cont_seg: ImageContainer, props: Sequence[str]):
        if not len(props):
            with pytest.raises(ValueError, match=r"No properties have been selected."):
                small_cont_seg.features_segmentation(
                    label_layer="segmented", intensity_layer="image", feature_name="foo", props=props
                )
        else:
            features = small_cont_seg.features_segmentation(
                label_layer="segmented", intensity_layer="image", feature_name="foo", props=props, channels=[0]
            )
            haystack = features.keys()

            int_props = [p for p in props if "intensity" in props]
            no_int_props = [p for p in props if "intensity" not in props]

            for p in no_int_props:
                assert any(f"{p}_mean" in h for h in haystack), haystack
                assert any(f"{p}_std" in h for h in haystack), haystack

            for p in int_props:
                assert any(f"ch-0_{p}_mean" in h for h in haystack), haystack
                assert any(f"ch-0_{p}_std" in h for h in haystack), haystack

    def test_custom_default_name(self, small_cont: ImageContainer):
        custom_features = small_cont.features_custom(np.mean, layer="image", channels=[0])
        summary_features = small_cont.features_summary("image", feature_name="summary", channels=[0])

        assert len(custom_features) == 1
        assert f"{np.mean.__name__}_0" in custom_features
        assert custom_features[f"{np.mean.__name__}_0"] == summary_features["summary_ch-0_mean"]

    def test_custom_returns_iterable(self, small_cont: ImageContainer):
        def dummy(_: np.ndarray) -> Tuple[int, int]:
            return 0, 1

        features = small_cont.features_custom(dummy, layer="image", feature_name="foo")

        assert len(features) == 2
        assert features["foo_0"] == 0
        assert features["foo_1"] == 1


class TestHighLevel:
    def test_invalid_layer(self, adata: AnnData, cont: ImageContainer):
        with pytest.raises(KeyError, match=r"Image layer `foo` not found"):
            calculate_image_features(adata, cont, layer="foo")

    def test_invalid_feature(self, adata: AnnData, cont: ImageContainer):
        with pytest.raises(ValueError, match=r"Invalid option `foo` for `ImageFeature`"):
            calculate_image_features(adata, cont, features="foo")

    def test_passing_spot_crops_kwargs(self, adata: AnnData, cont: ImageContainer, mocker: MockerFixture):
        spy = mocker.spy(cont, "generate_spot_crops")
        calculate_image_features(adata, cont, mask_circle=True)

        spy.assert_called_once()
        call = spy.call_args_list[0]
        assert call[-1]["mask_circle"]

    def test_passing_feature_kwargs(self, adata: AnnData, cont: ImageContainer):
        def dummy(_: np.ndarray, sentinel: bool = False) -> int:
            assert sentinel
            return 42

        res = calculate_image_features(
            adata,
            cont,
            key_added="foo",
            features=ImageFeature.CUSTOM.s,
            features_kwargs={ImageFeature.CUSTOM.s: {"func": dummy, "sentinel": True, "channels": [0]}},
            copy=True,
        )

        assert isinstance(res, pd.DataFrame)
        np.testing.assert_array_equal(res.index, adata.obs_names)
        np.testing.assert_array_equal(res.columns, ["dummy_0"])
        np.testing.assert_array_equal(res["dummy_0"].values, 42)

    def test_key_added(self, adata: AnnData, cont: ImageContainer):
        assert "foo" not in adata.obsm
        res = calculate_image_features(adata, cont, key_added="foo", copy=False)

        assert res is None
        assert "foo" in adata.obsm
        assert isinstance(adata.obsm["foo"], pd.DataFrame)

    def test_copy(self, adata: AnnData, cont: ImageContainer):
        orig_keys = set(adata.obsm.keys())
        res = calculate_image_features(adata, cont, key_added="foo", copy=True)

        assert isinstance(res, pd.DataFrame)
        np.testing.assert_array_equal(res.index, adata.obs_names)
        assert set(adata.obsm.keys()) == orig_keys

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_parallelize(self, adata: AnnData, cont: ImageContainer, n_jobs: int):
        features = ["texture", "summary", "histogram"]
        res = calculate_image_features(adata, cont, features=features, copy=True, n_jobs=n_jobs)

        assert isinstance(res, pd.DataFrame)
        np.testing.assert_array_equal(res.index, adata.obs_names)
        assert [key for key in res.keys() if "texture" in key] != [], "feature name texture not in dict keys"
        assert [key for key in res.keys() if "summary" in key] != [], "feature name summary not in dict keys"
        assert [key for key in res.keys() if "histogram" in key] != [], "feature name histogram not in dict keys"
