import warnings
from http.client import RemoteDisconnected
from pathlib import Path
from types import FunctionType

import pytest
import squidpy as sq
from anndata import AnnData, OldFormatWarning


class TestDatasetsImports:
    @pytest.mark.parametrize("func", sq.datasets._dataset.__all__ + sq.datasets._image.__all__)
    def test_import(self, func):
        assert hasattr(sq.datasets, func), dir(sq.datasets)
        fn = getattr(sq.datasets, func)

        assert isinstance(fn, FunctionType)


# TODO(michalk8): parse the code and xfail iff server issue
class TestDatasetsDownload:
    @pytest.mark.timeout(120)
    def test_download_imc(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OldFormatWarning)
            try:
                adata = sq.datasets.imc(tmp_path / "foo")

                assert isinstance(adata, AnnData)
                assert adata.shape == (4668, 34)
            except RemoteDisconnected as e:
                pytest.xfail(str(e))

    @pytest.mark.timeout(120)
    def test_download_visium_hne_image_crop(self, tmp_path: Path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=OldFormatWarning)
            try:
                img = sq.datasets.visium_hne_image_crop(tmp_path / "foo")

                assert isinstance(img, sq.im.ImageContainer)
                assert img.shape == (3527, 3527)
            except RemoteDisconnected as e:
                pytest.xfail(str(e))
