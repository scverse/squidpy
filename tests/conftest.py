import sys
import pickle
from typing import Tuple, Sequence, NamedTuple
from itertools import product

import pytest

import scanpy as sc
from anndata import AnnData

_adata = sc.read("tests/_data/test_data.h5ad")
_adata.raw = _adata.copy()


@pytest.fixture(scope="function")
def adata() -> AnnData:
    return _adata.copy()


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
