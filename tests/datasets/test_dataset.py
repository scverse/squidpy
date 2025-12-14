from __future__ import annotations

import warnings
from http.client import RemoteDisconnected

import pytest
from anndata import AnnData, OldFormatWarning

import squidpy as sq

# All public dataset functions that should be importable
_DATASET_FUNCTIONS = [
    # AnnData datasets
    "four_i",
    "imc",
    "seqfish",
    "visium_hne_adata",
    "visium_hne_adata_crop",
    "visium_fluo_adata",
    "visium_fluo_adata_crop",
    "sc_mouse_cortex",
    "mibitof",
    "merfish",
    "slideseqv2",
    # Image datasets
    "visium_fluo_image_crop",
    "visium_hne_image_crop",
    "visium_hne_image",
    # 10x Visium
    "visium",
    "visium_hne_sdata",
]


class TestDatasetsImports:
    @pytest.mark.parametrize("func", _DATASET_FUNCTIONS)
    def test_import(self, func):
        assert hasattr(sq.datasets, func), dir(sq.datasets)
        fn = getattr(sq.datasets, func)

        assert callable(fn)


# TODO(michalk8): parse the code and xfail iff server issue
class TestDatasetsDownload:
    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_download_imc(self):
        # Not passing path uses scanpy.settings.datasetdir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OldFormatWarning)
            try:
                adata = sq.datasets.imc()

                assert isinstance(adata, AnnData)
                assert adata.shape == (4668, 34)
            except RemoteDisconnected as e:
                pytest.xfail(str(e))

    @pytest.mark.timeout(120)
    @pytest.mark.internet()
    def test_download_visium_hne_image_crop(self):
        # Not passing path uses scanpy.settings.datasetdir
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OldFormatWarning)
            try:
                img = sq.datasets.visium_hne_image_crop()

                assert isinstance(img, sq.im.ImageContainer)
                assert img.shape == (3527, 3527)
            except RemoteDisconnected as e:
                pytest.xfail(str(e))
