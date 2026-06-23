"""Tests for squidpy's dataset loaders + downloader (built on scverse_misc.datasets)."""

from __future__ import annotations

import pytest
from scanpy import settings
from scverse_misc.datasets import available_loaders

from squidpy.datasets._downloader import download


class TestLoaderRegistration:
    def test_squidpy_loaders_registered(self):
        # importing the downloader module registers squidpy's domain loaders;
        # anndata + spatialdata are shipped by scverse-misc
        assert {"anndata", "image", "spatialdata", "visium_10x"} <= set(available_loaders())


class TestDownloadFunction:
    def test_unknown_dataset_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset"):
            download("nonexistent_dataset")


class TestDownloaderIntegration:
    """Integration tests that require network access."""

    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_download_anndata(self):
        from anndata import AnnData

        adata = download("imc", settings.datasetdir)
        assert isinstance(adata, AnnData)
        assert adata.shape == (4668, 34)

    @pytest.mark.timeout(180)
    @pytest.mark.internet()
    def test_download_visium_sample(self):
        from anndata import AnnData

        adata = download("V1_Mouse_Kidney", settings.datasetdir, include_hires_tiff=False)
        assert isinstance(adata, AnnData)
        assert "spatial" in adata.uns
