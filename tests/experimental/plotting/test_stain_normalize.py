from __future__ import annotations

import matplotlib.pyplot as plt
import spatialdata_plot as sdp
import xarray as xr
from spatialdata.models import Image2DModel

import squidpy as sq
from squidpy.experimental.im import fit_stain_reference, normalize_stains
from squidpy.experimental.im._utils import get_element_data
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp  # registers the `.pl` spatialdata accessor


class TestStainNormalizationVisual(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_reinhard_before_after(self, sdata_hne) -> None:
        """Visual: a re-stained source (left) normalized back to the H&E reference (right)."""
        image_key = next(iter(sdata_hne.images))
        sq.experimental.im.detect_tissue(sdata_hne, image_key)
        reference = fit_stain_reference(sdata_hne, image_key, method="reinhard")

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
