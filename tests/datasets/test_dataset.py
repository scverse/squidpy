from types import FunctionType
from pathlib import Path
from http.client import RemoteDisconnected
import pytest

from anndata import AnnData

import squidpy as sq


class TestDatasetsImports:
    @pytest.mark.parametrize("func", sq.datasets._dataset.__all__ + sq.datasets._image.__all__)
    def test_import(self, func):
        assert hasattr(sq.datasets, func), dir(sq.datasets)
        fn = getattr(sq.datasets, func)

        assert isinstance(fn, FunctionType)


class TestDatasetsDownload:
    def test_download_imc(self, tmp_path: Path):
        try:
            adata = sq.datasets.imc(tmp_path / "foo")

            assert isinstance(adata, AnnData)
            assert adata.shape == (4668, 34)
        except RemoteDisconnected as e:
            pytest.skip(str(e))

    def test_download_visium_hne_image_crop(self, tmp_path: Path):
        try:
            img = sq.datasets.visium_hne_image_crop(tmp_path / "foo")

            assert isinstance(img, sq.im.ImageContainer)
            assert img.shape == (3527, 3527)
        except RemoteDisconnected as e:
            pytest.skip(str(e))
