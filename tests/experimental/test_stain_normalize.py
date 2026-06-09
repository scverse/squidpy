from __future__ import annotations

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest
import spatialdata as sd
import spatialdata_plot as sdp
import xarray as xr
from spatialdata.models import Image2DModel
from spatialdata.transformations import Scale, get_transformation, set_transformation

import squidpy as sq
from squidpy.experimental.im import (
    ReinhardParams,
    StainReference,
    apply_stain_normalization,
    fit_stain_reference,
)
from squidpy.experimental.im._utils import get_element_data
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp  # registers the `.pl` spatialdata accessor


def _make_sdata(values: np.ndarray, *, scale_factors: list[int] | None = None) -> sd.SpatialData:
    img = Image2DModel.parse(values, dims=("c", "y", "x"), scale_factors=scale_factors)
    return sd.SpatialData(images={"img": img})


@pytest.fixture
def rgb_values() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.uniform(40.0, 200.0, size=(3, 64, 64)).astype(np.float32)


class TestFitStainReference:
    def test_end_to_end(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        assert isinstance(ref, StainReference)
        assert ref.method == "reinhard"

    def test_missing_image_key_raises(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        with pytest.raises(ValueError, match="not found, valid keys"):
            fit_stain_reference(sdata, "nope")

    def test_macenko_not_implemented(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        with pytest.raises(NotImplementedError, match="decomposition is not yet implemented"):
            fit_stain_reference(sdata, "img", method="macenko")

    def test_unknown_method_raises(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        with pytest.raises(ValueError, match="Unknown method"):
            fit_stain_reference(sdata, "img", method="bogus")


class TestApplyStainNormalization:
    def test_returns_lazy_and_leaves_sdata_untouched(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        out = apply_stain_normalization(sdata, "img", ref)
        assert isinstance(out, xr.DataArray)
        assert isinstance(out.data, da.Array)
        assert list(sdata.images.keys()) == ["img"]

    def test_writes_and_preserves_transform_and_dims(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        result = apply_stain_normalization(sdata, "img", ref, image_key_added="norm")
        assert result is None
        assert "norm" in sdata.images
        out = sdata.images["norm"]
        assert out.dims == ("c", "y", "x")
        assert out.shape == rgb_values.shape
        assert (
            get_transformation(out, get_all=True).keys() == get_transformation(sdata.images["img"], get_all=True).keys()
        )

    def test_multiscale_rebuilds_pyramid(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values, scale_factors=[2])
        ref = fit_stain_reference(sdata, "img")
        apply_stain_normalization(sdata, "img", ref, image_key_added="norm")
        src, out = sdata.images["img"], sdata.images["norm"]
        assert hasattr(out, "keys")
        src_shapes = [src[k].image.shape for k in src]
        out_shapes = [out[k].image.shape for k in out]
        assert out_shapes == src_shapes

    def test_preserves_channel_coords_and_nonidentity_transform(self, rgb_values: np.ndarray) -> None:
        img = Image2DModel.parse(rgb_values, dims=("c", "y", "x"), c_coords=["r", "g", "b"])
        set_transformation(img, Scale([2.0, 2.0], axes=("y", "x")), to_coordinate_system="global")
        sdata = sd.SpatialData(images={"img": img})
        ref = fit_stain_reference(sdata, "img")
        apply_stain_normalization(sdata, "img", ref, image_key_added="norm")
        out = sdata.images["norm"]
        assert list(out.coords["c"].values) == ["r", "g", "b"]
        assert get_transformation(out, get_all=True) == get_transformation(img, get_all=True)

    def test_existing_key_raises(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        with pytest.raises(ValueError, match="already exists"):
            apply_stain_normalization(sdata, "img", ref, image_key_added="img")

    def test_decomposition_reference_not_implemented(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = StainReference(
            method="macenko",
            stain_matrix=np.eye(3),
            background_intensity=np.array([255.0, 255.0, 255.0]),
        )
        with pytest.raises(NotImplementedError, match="decomposition is not yet implemented"):
            apply_stain_normalization(sdata, "img", ref)

    def test_method_params_mapping(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img", method_params={"mask_background": False})
        out = apply_stain_normalization(sdata, "img", ref, method_params=ReinhardParams(mask_background=False))
        assert isinstance(out, xr.DataArray)


class TestStainNormalizationOnHnE:
    def test_fit_apply_smoke(self, sdata_hne) -> None:
        image_key = next(iter(sdata_hne.images))
        ref = sq.experimental.im.fit_stain_reference(sdata_hne, image_key)
        assert ref.method == "reinhard"
        out = sq.experimental.im.apply_stain_normalization(sdata_hne, image_key, ref)
        assert "c" in out.dims
        assert out.sizes["c"] == 3


class TestStainNormalizationVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_reinhard_before_after(self, sdata_hne) -> None:
        """Visual: a re-stained source (left) normalized back to the H&E reference (right)."""
        image_key = next(iter(sdata_hne.images))
        reference = fit_stain_reference(sdata_hne, image_key)

        # Deterministically warm/cool the channels to simulate a different
        # staining batch, so the before/after panels are visibly distinct.
        da_rgb = get_element_data(sdata_hne.images[image_key], "auto", "image", image_key).astype("float32")
        weights = xr.DataArray([1.4, 1.0, 0.6], dims="c", coords={"c": da_rgb.coords["c"]})
        shifted = (da_rgb * weights).clip(0, 255)
        sdata_hne.images["hne_shifted"] = Image2DModel.parse(shifted.data, dims=shifted.dims)

        apply_stain_normalization(sdata_hne, "hne_shifted", reference, image_key_added="hne_normalized")

        _, axes = plt.subplots(1, 2, figsize=(8, 4))
        sdata_hne.pl.render_images("hne_shifted").pl.show(ax=axes[0], title="before")
        sdata_hne.pl.render_images("hne_normalized").pl.show(ax=axes[1], title="after")
