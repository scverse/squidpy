"""Tests for squidpy's dataset loaders + downloader (built on scverse_misc.datasets)."""

from __future__ import annotations

from pathlib import Path

import pytest
from scanpy import settings
from scverse_misc.datasets import available_loaders

from squidpy.datasets import visium
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


class TestVisiumValidation:
    """Offline validation of the ``visium()`` sample-name guard (no network)."""

    def test_unknown_sample_raises(self):
        with pytest.raises(ValueError, match="Unknown Visium sample"):
            visium("definitely_not_a_sample")

    def test_valid_but_wrong_type_rejected(self):
        # `imc` is a valid registry entry but an AnnData dataset, not a Visium 10x
        # sample. It must be rejected before any download is attempted.
        with pytest.raises(ValueError, match="Unknown Visium sample"):
            visium("imc")


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

    @pytest.mark.timeout(300)
    @pytest.mark.internet()
    def test_include_hires_tiff_metadata_toggle(self):
        """``include_hires_tiff`` toggles ``source_image_path`` in ``adata.uns`` (squidpy read.visium wiring).

        On CI V1_Mouse_Kidney is pre-cached with the hires image via
        ``.scripts/ci/download_data.py``, so this exercises the return behavior offline-of-network.
        """
        sample_id = "V1_Mouse_Kidney"

        adata = visium(sample_id, include_hires_tiff=False)
        assert "source_image_path" not in adata.uns["spatial"][sample_id].get("metadata", {})

        adata = visium(sample_id, include_hires_tiff=True)
        metadata = adata.uns["spatial"][sample_id].get("metadata", {})
        assert "source_image_path" in metadata
        assert Path(metadata["source_image_path"]).exists()
