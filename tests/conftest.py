import sys
import pickle
from typing import Tuple, Sequence, NamedTuple
from itertools import product

import pytest

import scanpy as sc
from anndata import AnnData

import numpy as np
import pandas as pd

import spatial_tools as se
from spatial_tools.image.object import ImageContainer

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

    adata.obsm["spatial"] = np.stack([r.randint(0, 500, 200), r.randint(0, 500, 200)], axis=1)
    se.graph.spatial_connectivity(adata, obsm="spatial", n_rings=2)

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


@pytest.fixture
def logging_state():
    # modified from scanpy
    verbosity_orig = sc.settings.verbosity
    yield
    sc.settings.logfile = sys.stderr
    sc.settings.verbosity = verbosity_orig
