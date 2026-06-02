from __future__ import annotations

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pytest
import spatialdata as sd
import spatialdata_plot as sdp
import xarray as xr
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import Scale, get_transformation, set_transformation

import squidpy as sq
from squidpy.experimental.im import (
    ReinhardParams,
    StainReference,
    fit_stain_reference,
    normalize_stains,
)
from squidpy.experimental.im._utils import get_element_data
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp  # registers the `.pl` spatialdata accessor


def _make_sdata(
    values: np.ndarray, *, scale_factors: list[int] | None = None, with_tissue: bool = True
) -> sd.SpatialData:
    img = Image2DModel.parse(values, dims=("c", "y", "x"), scale_factors=scale_factors)
    sdata = sd.SpatialData(images={"img": img})
    if with_tissue:
        h, w = values.shape[-2], values.shape[-1]
        sdata.labels["img_tissue"] = Labels2DModel.parse(np.ones((h, w), dtype=np.uint32), dims=("y", "x"))
    return sdata


@pytest.fixture
def rgb_values() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.uniform(40.0, 200.0, size=(3, 64, 64)).astype(np.uint8)


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

    def test_unknown_method_raises(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        with pytest.raises(ValueError, match="Unknown method"):
            fit_stain_reference(sdata, "img", method="bogus")


class TestApplyStainNormalization:
    def test_returns_lazy_and_leaves_sdata_untouched(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        out = normalize_stains(sdata, "img", ref, inplace=False)
        assert isinstance(out, xr.DataArray)
        assert isinstance(out.data, da.Array)
        assert list(sdata.images.keys()) == ["img"]

    def test_inplace_default_writes_derived_key(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        result = normalize_stains(sdata, "img", ref)  # inplace=True, image_key_added defaults to f"{key}_normalized"
        assert result is None
        assert "img_normalized" in sdata.images
        out = sdata.images["img_normalized"]
        assert out.dtype == rgb_values.dtype  # cast back to the source dtype at the write boundary
        assert out.shape == rgb_values.shape

    def test_output_dtype_override(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        out = normalize_stains(sdata, "img", ref, inplace=False, output_dtype=np.uint16)
        assert out.dtype == np.uint16

    def test_writes_and_preserves_transform_and_dims(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        result = normalize_stains(sdata, "img", ref, image_key_added="norm")
        assert result is None
        assert "norm" in sdata.images
        out = sdata.images["norm"]
        assert out.dims == ("c", "y", "x")
        assert out.shape == rgb_values.shape
        assert out.dtype == rgb_values.dtype
        assert (
            get_transformation(out, get_all=True).keys() == get_transformation(sdata.images["img"], get_all=True).keys()
        )

    def test_multiscale_rebuilds_pyramid(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values, scale_factors=[2])
        ref = fit_stain_reference(sdata, "img")
        normalize_stains(sdata, "img", ref, image_key_added="norm")
        src, out = sdata.images["img"], sdata.images["norm"]
        assert hasattr(out, "keys")
        src_shapes = [src[k].image.shape for k in src]
        out_shapes = [out[k].image.shape for k in out]
        assert out_shapes == src_shapes

    def test_preserves_channel_coords_and_nonidentity_transform(self, rgb_values: np.ndarray) -> None:
        img = Image2DModel.parse(rgb_values, dims=("c", "y", "x"), c_coords=["r", "g", "b"])
        set_transformation(img, Scale([2.0, 2.0], axes=("y", "x")), to_coordinate_system="global")
        sdata = sd.SpatialData(images={"img": img})
        h, w = rgb_values.shape[-2], rgb_values.shape[-1]
        sdata.labels["img_tissue"] = Labels2DModel.parse(np.ones((h, w), dtype=np.uint32), dims=("y", "x"))
        ref = fit_stain_reference(sdata, "img")
        normalize_stains(sdata, "img", ref, image_key_added="norm")
        out = sdata.images["norm"]
        assert list(out.coords["c"].values) == ["r", "g", "b"]
        assert get_transformation(out, get_all=True) == get_transformation(img, get_all=True)

    def test_existing_key_raises(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img")
        with pytest.raises(ValueError, match="already exists"):
            normalize_stains(sdata, "img", ref, image_key_added="img")

    def test_decomposition_reference_without_max_concentrations_raises(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = StainReference(
            method="macenko",
            stain_matrix=np.eye(3),
            white_point=np.array([255.0, 255.0, 255.0]),
        )
        with pytest.raises(ValueError, match="max_concentrations"):
            normalize_stains(sdata, "img", ref)

    def test_method_params_mapping(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        ref = fit_stain_reference(sdata, "img", method_params={"mask_background": False})
        out = normalize_stains(sdata, "img", ref, method_params=ReinhardParams(mask_background=False), inplace=False)
        assert isinstance(out, xr.DataArray)


class TestTissueMaskMandate:
    def test_fit_requires_tissue_mask(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values, with_tissue=False)
        with pytest.raises(KeyError, match="detect_tissue"):
            fit_stain_reference(sdata, "img")

    def test_apply_requires_tissue_mask(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)  # has a mask -> fit works
        ref = fit_stain_reference(sdata, "img")
        del sdata.labels["img_tissue"]  # ... but now the source has none
        with pytest.raises(KeyError, match="detect_tissue"):
            normalize_stains(sdata, "img", ref)

    def test_explicit_missing_key_raises(self, rgb_values: np.ndarray) -> None:
        sdata = _make_sdata(rgb_values)
        with pytest.raises(KeyError, match="not found in sdata.labels"):
            fit_stain_reference(sdata, "img", tissue_mask_key="nope")

    def test_float_0_255_source_rejected_on_apply(self, rgb_values: np.ndarray) -> None:
        # A float image holding 0-255 values would otherwise clip to [0, 1] in the
        # reconstruction (dtype_max(float)=1.0); apply must reject it, not silently destroy it.
        sdata = _make_sdata(rgb_values)  # uint8
        ref = fit_stain_reference(sdata, "img")
        floaty = rgb_values.astype(np.float32)
        sdata.images["floaty"] = Image2DModel.parse(floaty, dims=("c", "y", "x"))
        sdata.labels["floaty_tissue"] = Labels2DModel.parse(
            np.ones(floaty.shape[-2:], dtype=np.uint32), dims=("y", "x")
        )
        with pytest.raises(ValueError, match="stored as float"):
            normalize_stains(sdata, "floaty", ref)

    def test_mask_is_used_in_the_fit(self, rgb_values: np.ndarray) -> None:
        # A different tissue region yields different channel statistics, proving
        # the mask actually drives the fit (not silently ignored).
        ref_full = fit_stain_reference(_make_sdata(rgb_values), "img")

        sdata_part = _make_sdata(rgb_values, with_tissue=False)
        h, w = rgb_values.shape[-2], rgb_values.shape[-1]
        partial = np.zeros((h, w), dtype=np.uint32)
        partial[: h // 2] = 1  # only the top half is tissue
        sdata_part.labels["img_tissue"] = Labels2DModel.parse(partial, dims=("y", "x"))
        ref_part = fit_stain_reference(sdata_part, "img")

        assert not np.allclose(ref_full.mu, ref_part.mu)


class TestPreserveBackground:
    def test_background_passthrough_vs_full_frame(self, rgb_values: np.ndarray) -> None:
        # tissue = top half only; bottom half is background
        h, w = rgb_values.shape[-2], rgb_values.shape[-1]
        sdata = _make_sdata(rgb_values, with_tissue=False)
        partial = np.zeros((h, w), dtype=np.uint32)
        partial[: h // 2] = 1
        sdata.labels["img_tissue"] = Labels2DModel.parse(partial, dims=("y", "x"))

        # a differently-coloured reference so the transform is non-trivial
        shifted = np.clip(rgb_values * np.array([1.3, 0.8, 1.1])[:, None, None], 0, 255).astype(np.uint8)
        sdata.images["ref_img"] = Image2DModel.parse(shifted, dims=("c", "y", "x"))
        sdata.labels["ref_img_tissue"] = Labels2DModel.parse(np.ones((h, w), dtype=np.uint32), dims=("y", "x"))
        ref = fit_stain_reference(sdata, "ref_img")

        original = get_element_data(sdata.images["img"], "auto", "image", "img").values
        kept = normalize_stains(sdata, "img", ref, inplace=False).values  # preserve_background=True (default)
        full = normalize_stains(sdata, "img", ref, preserve_background=False, inplace=False).values

        bg = slice(h // 2, None)
        np.testing.assert_allclose(kept[:, bg], original[:, bg])  # background untouched
        assert not np.allclose(full[:, bg], original[:, bg])  # full-frame recolours it


class TestStainNormalizationOnHnE:
    def test_fit_apply_smoke(self, sdata_hne) -> None:
        image_key = next(iter(sdata_hne.images))
        sq.experimental.im.detect_tissue(sdata_hne, image_key)
        ref = sq.experimental.im.fit_stain_reference(sdata_hne, image_key)
        assert ref.method == "reinhard"
        out = sq.experimental.im.normalize_stains(sdata_hne, image_key, ref, inplace=False)
        assert "c" in out.dims
        assert out.sizes["c"] == 3


class TestStainNormalizationVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_reinhard_before_after(self, sdata_hne) -> None:
        """Visual: a re-stained source (left) normalized back to the H&E reference (right)."""
        image_key = next(iter(sdata_hne.images))
        sq.experimental.im.detect_tissue(sdata_hne, image_key)
        reference = fit_stain_reference(sdata_hne, image_key)

        # Deterministically warm/cool the channels to simulate a different
        # staining batch, so the before/after panels are visibly distinct.
        da_rgb = get_element_data(sdata_hne.images[image_key], "auto", "image", image_key).astype("float32")
        weights = xr.DataArray([1.4, 1.0, 0.6], dims="c", coords={"c": da_rgb.coords["c"]})
        shifted = (da_rgb * weights).clip(0, 255).astype("uint8")
        sdata_hne.images["hne_shifted"] = Image2DModel.parse(shifted.data, dims=shifted.dims)

        # `hne_shifted` shares geometry with `image_key`; reuse its tissue mask.
        normalize_stains(
            sdata_hne, "hne_shifted", reference, image_key_added="hne_normalized", tissue_mask_key=f"{image_key}_tissue"
        )

        _, axes = plt.subplots(1, 2, figsize=(8, 4))
        sdata_hne.pl.render_images("hne_shifted").pl.show(ax=axes[0], title="before")
        sdata_hne.pl.render_images("hne_normalized").pl.show(ax=axes[1], title="after")
