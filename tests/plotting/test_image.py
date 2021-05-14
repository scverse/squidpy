import pytest

from anndata import AnnData
import scanpy as sc

import numpy as np

import matplotlib.pyplot as plt

from squidpy.im import ImageContainer
from tests.conftest import DPI, PlotTester, PlotTesterMeta
import squidpy as sq


class TestContainerShow(PlotTester, metaclass=PlotTesterMeta):
    def test_channelwise_wrong_number_of_axes(self, cont: ImageContainer):
        fig, ax = plt.subplots(dpi=DPI, tight_layout=True)
        with pytest.raises(ValueError, match=r"Expected `3` axes, found `1`."):
            cont.show(ax=ax, channelwise=True)

    def test_plot_axis(self, cont: ImageContainer):
        cont.add_img(np.random.RandomState(42).normal(size=(*cont.shape, 3)), layer="foo")
        fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=DPI, tight_layout=True)

        cont.show("image", ax=ax1)
        cont.show("foo", ax=ax2)

    def test_plot_channel(self, cont: ImageContainer):
        cont.show(channel=1, dpi=DPI)

    def test_plot_segmentation(self, cont: ImageContainer):
        seg = np.random.RandomState(43).randint(0, 255, size=(*cont.shape, 1))
        seg[seg <= 200] = 0
        cont.add_img(seg, layer="foo")
        cont["foo"].attrs["segmentation"] = True

        cont.show("image", segmentation_layer="foo", dpi=DPI)

    def test_plot_imshow_kwargs(self, cont: ImageContainer):
        cont.show(channel=2, cmap="inferno", dpi=DPI)

    def test_plot_channelwise(self, cont: ImageContainer):
        cont.show(channelwise=True, dpi=DPI)

    def test_plot_channelwise_segmentation(self, cont: ImageContainer):
        seg = np.random.RandomState(43).randint(0, 255, size=(*cont.shape, 1))
        seg[seg <= 200] = 0
        cont.add_img(seg, layer="foo")
        cont["foo"].attrs["segmentation"] = True

        cont.show("image", channelwise=True, segmentation_layer="foo", dpi=DPI, segmentation_alpha=1)


def test_extract(adata: AnnData, cont: ImageContainer, caplog):
    """
    Calculate features and extract columns to obs
    """
    # get obsm
    sq.im.calculate_image_features(adata, cont, features=["summary"])

    # extract columns (default values)
    extr_adata = sq.pl.extract(adata)
    # Test that expected columns exist
    for col in [
        "summary_ch-0_quantile-0.9",
        "summary_ch-0_quantile-0.5",
        "summary_ch-0_quantile-0.1",
        "summary_ch-1_quantile-0.9",
        "summary_ch-1_quantile-0.5",
        "summary_ch-1_quantile-0.1",
        "summary_ch-2_quantile-0.9",
        "summary_ch-2_quantile-0.5",
        "summary_ch-2_quantile-0.1",
    ]:
        assert col in extr_adata.obs.columns

    # get obsm that is a numpy array
    adata.obsm["pca_features"] = sc.pp.pca(adata.obsm["img_features"], n_comps=3)
    # extract columns
    extr_adata = sq.pl.extract(adata, obsm_key="pca_features", prefix="pca_features")
    # Test that expected columns exist
    for col in ["pca_features_0", "pca_features_1", "pca_features_2"]:
        assert col in extr_adata.obs.columns

    # extract multiple obsm at once (no prefix)
    extr_adata = sq.pl.extract(adata, obsm_key=["img_features", "pca_features"])
    # Test that expected columns exist
    for col in [
        "summary_ch-0_quantile-0.9",
        "summary_ch-0_quantile-0.5",
        "summary_ch-0_quantile-0.1",
        "summary_ch-1_quantile-0.9",
        "summary_ch-1_quantile-0.5",
        "summary_ch-1_quantile-0.1",
        "summary_ch-2_quantile-0.9",
        "summary_ch-2_quantile-0.5",
        "summary_ch-2_quantile-0.1",
        "0",
        "1",
        "2",
    ]:
        assert col in extr_adata.obs.columns

    # TODO: test similarly to ligrec
    # currently logging to stderr, and not captured by caplog
    # extract obsm twice and make sure that warnings are issued
    # with caplog.at_level(logging.WARNING):
    #    extr2_adata = sq.pl.extract(extr_adata, obsm_key=['pca_features'])
    #    log = caplog.text
    #    assert "will be overwritten by extract" in log
