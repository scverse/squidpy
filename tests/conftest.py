from __future__ import annotations

import pickle
import sys
import warnings
from abc import ABC, ABCMeta
from functools import wraps
from itertools import product
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import squidpy as sq
from anndata import AnnData, OldFormatWarning
from matplotlib.testing.compare import compare_images
from scipy.sparse import csr_matrix
from squidpy._constants._pkg_constants import Key
from squidpy.gr import spatial_neighbors
from squidpy.im._container import ImageContainer

HERE: Path = Path(__file__).parent

EXPECTED = HERE / "_images"
ACTUAL = HERE / "figures"
TOL = 50
DPI = 40

C_KEY_PALETTE = "leiden"

_adata = sc.read("tests/_data/test_data.h5ad")
_adata.raw = _adata.copy()


def pytest_sessionstart(session: pytest.Session) -> None:
    matplotlib.use("Agg")
    matplotlib.rcParams["figure.max_open_warning"] = 0
    np.random.seed(42)

    warnings.simplefilter("ignore", OldFormatWarning)
    sc.pl.set_rcParams_defaults()


@pytest.fixture(scope="session")
def adata_hne() -> AnnData:
    return sq.datasets.visium_hne_adata_crop()


@pytest.fixture(scope="session")
def adata_hne_concat() -> AnnData:
    adata1 = sq.datasets.visium_hne_adata_crop()
    spatial_neighbors(adata1)
    adata2 = adata1[:100, :].copy()
    adata2.uns["spatial"] = {}
    adata2.uns["spatial"]["V2_Adult_Mouse_Brain"] = adata1.uns["spatial"]["V1_Adult_Mouse_Brain"]
    adata_concat = ad.concat(
        {"V1_Adult_Mouse_Brain": adata1, "V2_Adult_Mouse_Brain": adata2},
        label="library_id",
        uns_merge="unique",
        pairwise=True,
    )
    return adata_concat


@pytest.fixture(scope="session")
def adata_mibitof() -> AnnData:
    return sq.datasets.mibitof().copy()


@pytest.fixture(scope="session")
def adata_seqfish() -> AnnData:
    return sq.datasets.seqfish().copy()


@pytest.fixture()
def adata() -> AnnData:
    return _adata.copy()


@pytest.fixture()
def adata_palette() -> AnnData:
    from matplotlib.cm import get_cmap

    cmap = get_cmap("Set1")

    adata_palette = _adata.copy()
    adata_palette.uns[f"{C_KEY_PALETTE}_colors"] = cmap(range(adata_palette.obs[C_KEY_PALETTE].unique().shape[0]))
    return adata_palette.copy()


@pytest.fixture()
def nhood_data(adata: AnnData) -> AnnData:
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, key_added="leiden")
    sq.gr.spatial_neighbors(adata)

    return adata


@pytest.fixture()
def dummy_adata() -> AnnData:
    r = np.random.RandomState(100)
    adata = AnnData(r.rand(200, 100), obs={"cluster": r.randint(0, 3, 200)}, dtype=float)

    adata.obsm[Key.obsm.spatial] = np.stack([r.randint(0, 500, 200), r.randint(0, 500, 200)], axis=1)
    sq.gr.spatial_neighbors(adata, spatial_key=Key.obsm.spatial, n_rings=2)

    return adata


@pytest.fixture()
def adata_intmat() -> AnnData:
    graph = csr_matrix(
        np.array(
            [
                [0, 1, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 2, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 0, 1, 2, 0],
            ]
        )
    )
    return AnnData(
        np.zeros((5, 5)),
        obs={"cat": pd.Categorical.from_codes([0, 0, 0, 1, 1], ("a", "b"))},
        obsp={"spatial_connectivities": graph},
        dtype=float,
    )


@pytest.fixture()
def adata_ripley() -> AnnData:
    from matplotlib.cm import get_cmap

    adata = _adata[_adata.obs.leiden.isin(["0", "2"])].copy()
    cmap = get_cmap("Set1")

    adata.uns[f"{C_KEY_PALETTE}_colors"] = cmap(range(adata.obs[C_KEY_PALETTE].unique().shape[0]))
    return adata


@pytest.fixture()
def adata_squaregrid() -> AnnData:
    rng = np.random.default_rng(42)
    coord = rng.integers(0, 10, size=(400, 2))
    coord = np.unique(coord, axis=0)
    counts = rng.integers(0, 10, size=(coord.shape[0], 10))
    adata = AnnData(counts, dtype=counts.dtype)
    adata.obsm["spatial"] = coord
    sc.pp.scale(adata)
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


@pytest.fixture()
def cont() -> ImageContainer:
    return ImageContainer("tests/_data/test_img.jpg")


@pytest.fixture()
def small_cont() -> ImageContainer:
    np.random.seed(42)
    return ImageContainer(np.random.uniform(size=(100, 100, 3), low=0, high=1), layer="image")


@pytest.fixture()
def small_cont_4d() -> ImageContainer:
    np.random.seed(42)
    return ImageContainer(
        np.random.uniform(size=(100, 50, 2, 3), low=0, high=1), dims=["y", "x", "z", "channels"], layer="image"
    )


@pytest.fixture()
def cont_4d() -> ImageContainer:
    arrs = [np.linspace(0, 1, 10 * 10 * 3).reshape(10, 10, 3), np.zeros((10, 10, 3)) + 0.5, np.zeros((10, 10, 3))]
    arrs[1][4:6, 4:6] = 0.8
    arrs[2][2:8, 2:8, 0] = 0.5
    arrs[2][2:8, 2:8, 1] = 0.1
    arrs[2][2:8, 2:8, 2] = 0.9
    return ImageContainer.concat([ImageContainer(arr) for arr in arrs], library_ids=["3", "1", "2"])


@pytest.fixture()
def small_cont_seg() -> ImageContainer:
    np.random.seed(42)
    img = ImageContainer(np.random.randint(low=0, high=255, size=(100, 100, 3), dtype=np.uint8), layer="image")
    mask = np.zeros((100, 100), dtype="uint8")
    mask[20:30, 10:20] = 1
    mask[50:60, 30:40] = 2
    img.add_img(mask, layer="segmented", channel_dim="mask")

    return img


@pytest.fixture()
def small_cont_1c() -> ImageContainer:
    np.random.seed(42)
    return ImageContainer(np.random.normal(size=(100, 100, 1)) + 1, layer="image")


@pytest.fixture()
def cont_dot() -> ImageContainer:
    ys, xs = 100, 200
    img_orig = np.zeros((ys, xs, 10), dtype=np.uint8)
    img_orig[20, 50, :] = range(10, 20)  # put a dot at y 20, x 50
    return ImageContainer(img_orig, layer="image_0")


@pytest.fixture()
def napari_cont() -> ImageContainer:
    return ImageContainer("tests/_data/test_img.jpg", layer="V1_Adult_Mouse_Brain", library_id="V1_Adult_Mouse_Brain")


@pytest.fixture()
def interactions(adata: AnnData) -> tuple[Sequence[str], Sequence[str]]:
    return tuple(product(adata.raw.var_names[:5], adata.raw.var_names[:5]))  # type: ignore


@pytest.fixture()
def complexes(adata: AnnData) -> Sequence[tuple[str, str]]:
    g = adata.raw.var_names
    return [
        (g[0], g[1]),
        (f"{g[2]}_{g[3]}", g[4]),
        (g[5], f"{g[6]}_{g[7]}"),
        (f"{g[8]}_{g[9]}", f"{g[10]}_{g[11]}"),
        (f"foo_{g[12]}_bar_baz", g[13]),
    ]


@pytest.fixture(scope="session")
def ligrec_no_numba() -> Mapping[str, pd.DataFrame]:
    with open("tests/_data/ligrec_no_numba.pickle", "rb") as fin:
        data = pickle.load(fin)
        return {"means": data[0], "pvalues": data[1], "metadata": data[2]}


@pytest.fixture(scope="session")
def ligrec_result() -> Mapping[str, pd.DataFrame]:
    adata = _adata.copy()
    interactions = tuple(product(adata.raw.var_names[:5], adata.raw.var_names[:5]))
    return sq.gr.ligrec(
        adata, "leiden", interactions=interactions, n_perms=25, n_jobs=1, show_progress_bar=False, copy=True, seed=0
    )


@pytest.fixture(autouse=True)
def _logging_state():
    # modified from scanpy
    verbosity_orig = sc.settings.verbosity
    yield
    sc.settings.logfile = sys.stderr
    sc.settings.verbosity = verbosity_orig


@pytest.fixture()
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
    adata = AnnData(X=np.ones((visium_coords.shape[0], 3)), dtype=float)
    adata.obsm[Key.obsm.spatial] = visium_coords
    adata.uns[Key.uns.spatial] = {}
    return adata


@pytest.fixture()
def non_visium_adata():
    non_visium_coords = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])
    adata = AnnData(X=non_visium_coords, dtype=int)
    adata.obsm[Key.obsm.spatial] = non_visium_coords
    return adata


def _decorate(fn: Callable, clsname: str, name: str | None = None) -> Callable:
    @wraps(fn)
    def save_and_compare(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        self.compare(fig_name)

    if not callable(fn):
        raise TypeError(f"Expected a `callable` for class `{clsname}`, found `{type(fn).__name__}`.")

    name = fn.__name__ if name is None else name

    if not name.startswith("test_plot_") or not clsname.startswith("Test"):
        return fn

    fig_name = f"{clsname[4:]}_{name[10:]}"

    return save_and_compare


class PlotTesterMeta(ABCMeta):
    def __new__(cls, clsname, superclasses, attributedict):
        for key, value in attributedict.items():
            if callable(value):
                attributedict[key] = _decorate(value, clsname, name=key)
        return super().__new__(cls, clsname, superclasses, attributedict)


# ideally, we would you metaclass=PlotTesterMeta and all plotting tests just subclass this
# but for some reason, pytest erases the metaclass info
class PlotTester(ABC):
    @classmethod
    def compare(cls, basename: str, tolerance: float | None = None):
        ACTUAL.mkdir(parents=True, exist_ok=True)
        out_path = ACTUAL / f"{basename}.png"

        plt.savefig(out_path, dpi=DPI)
        plt.close()

        if tolerance is None:
            # see https://github.com/scverse/squidpy/pull/302
            tolerance = 2 * TOL if "Napari" in str(basename) else TOL

        res = compare_images(str(EXPECTED / f"{basename}.png"), str(out_path), tolerance)

        assert res is None, res


def pytest_addoption(parser):
    parser.addoption("--test-napari", action="store_true", help="Test interactive image view")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--test-napari"):
        return
    skip_slow = pytest.mark.skip(reason="Need --test-napari option to test interactive image view")
    for item in items:
        if "qt" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def _test_napari(pytestconfig):
    _ = pytestconfig.getoption("--test-napari", skip=True)
