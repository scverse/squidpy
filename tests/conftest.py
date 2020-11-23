import sys
import pickle
from typing import Tuple, Sequence, NamedTuple
from pathlib import Path
from itertools import product

import pytest

import scanpy as sc
from anndata import AnnData

import numpy as np
import pandas as pd

import matplotlib as mpl
from matplotlib import pyplot
from matplotlib.testing.compare import compare_images

import squidpy as se
from squidpy.image.object import ImageContainer

mpl.use("agg")

_adata = sc.read("tests/_data/test_data.h5ad")
_adata.raw = _adata.copy()


@pytest.fixture(scope="function")
def adata() -> AnnData:
    return _adata.copy()


@pytest.fixture(scope="function")
def nhood_data(adata: AnnData) -> AnnData:
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added="leiden")
    se.graph.spatial_connectivity(adata)

    return adata


@pytest.fixture(scope="function")
def dummy_adata() -> AnnData:
    r = np.random.RandomState(100)
    adata = AnnData(r.rand(200, 100), obs={"cluster": r.randint(0, 3, 200)})

    adata.obsm["X_spatial"] = np.stack([r.randint(0, 500, 200), r.randint(0, 500, 200)], axis=1)
    se.graph.spatial_connectivity(adata, obsm="X_spatial", n_rings=2)

    return adata


@pytest.fixture(scope="session")
def paul15() -> AnnData:
    # session because we don't modify this dataset
    adata = sc.datasets.paul15()
    sc.pp.normalize_per_cell(adata)
    adata.raw = adata.copy()

    return adata


@pytest.fixture(scope="session")
def paul15_means() -> pd.DataFrame:
    with open("tests/_data/paul15_means.pickle", "rb") as fin:
        return pickle.load(fin)


@pytest.fixture(scope="function")
def cont() -> ImageContainer:
    return ImageContainer("tests/_data/test_img.jpg")


@pytest.fixture
def interactions(adata: AnnData) -> Tuple[Sequence[str], Sequence[str]]:
    return tuple(product(adata.raw.var_names[:5], adata.raw.var_names[:5]))


@pytest.fixture
def complexes(adata: AnnData) -> Sequence[Tuple[str, str]]:
    g = adata.raw.var_names
    return [
        (g[0], g[1]),
        (f"{g[2]}_{g[3]}", g[4]),
        (g[5], f"{g[6]}_{g[7]}"),
        (f"{g[8]}_{g[9]}", f"{g[10]}_{g[11]}"),
        (f"foo_{g[12]}_bar_baz", g[13]),
    ]


@pytest.fixture(scope="session")
def ligrec_no_numba() -> NamedTuple:
    with open("tests/_data/ligrec_no_numba.pickle", "rb") as fin:
        return pickle.load(fin)


@pytest.fixture(scope="function")
def logging_state():
    # modified from scanpy
    verbosity_orig = sc.settings.verbosity
    yield
    sc.settings.logfile = sys.stderr
    sc.settings.verbosity = verbosity_orig


def make_comparer(path_expected: Path, path_actual: Path, *, tol: int):
    def save_and_compare(basename, tolerance=None):
        path_actual.mkdir(parents=True, exist_ok=True)
        out_path = path_actual / f"{basename}.png"
        pyplot.savefig(out_path, dpi=40)
        pyplot.close()
        if tolerance is None:
            tolerance = tol
        res = compare_images(str(path_expected / f"{basename}.png"), str(out_path), tolerance)
        assert res is None, res

    return save_and_compare


@pytest.fixture
def image_comparer():
    return make_comparer


@pytest.fixture(scope="function")
def visium_adata():
    visium_coords = np.array(
        [
            [4193, 7848],
            [4469, 7848],
            [4400, 7968],
            [4262, 7729],
            [3849, 7968],
            [4124, 7729],
            [4469, 7609],
            [3987, 8208],
            [4331, 8088],
            [4262, 7968],
            [4124, 7968],
            [4124, 7489],
            [4537, 7968],
            [4469, 8088],
            [4331, 7848],
            [4056, 7848],
            [3849, 7729],
            [4262, 7489],
            [4400, 8208],
            [4056, 7609],
            [3987, 7489],
            [4262, 8208],
            [4400, 7489],
            [4537, 7729],
            [4606, 7848],
            [3987, 7968],
            [3918, 8088],
            [3918, 7848],
            [4193, 8088],
            [4056, 8088],
            [4193, 7609],
            [3987, 7729],
            [4331, 7609],
            [4124, 8208],
            [3780, 7848],
            [3918, 7609],
            [4400, 7729],
        ]
    )
    adata = AnnData(X=np.ones((visium_coords.shape[0], 3)))
    adata.obsm["X_spatial"] = visium_coords
    return adata


@pytest.fixture(scope="function")
def non_visium_adata():
    non_visium_coords = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])
    adata = AnnData(X=non_visium_coords)
    adata.obsm["X_spatial"] = non_visium_coords
    return adata
