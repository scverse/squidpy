import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from squidpy._constants._pkg_constants import Key
from squidpy.gr._utils import _shuffle_group


class TestUtils:
    @pytest.mark.parametrize("cluster_annotations_type", [int, str])
    @pytest.mark.parametrize("library_annotations_type", [int, str])
    @pytest.mark.parametrize("seed", [422, 422222])
    def test_shuffle_group(self, cluster_annotations_type: type, library_annotations_type: type, seed: int):
        size = 6
        rng = np.random.default_rng(seed)
        if cluster_annotations_type == int:
            libraries = pd.Series(rng.choice([1, 2, 3, 4], size=(size,)), dtype="category")
        else:
            libraries = pd.Series(rng.choice(["a", "b", "c"], size=(size,)), dtype="category")

        if library_annotations_type == int:
            cluster_annotations = rng.choice([1, 2, 3, 4], size=(size,))
        else:
            cluster_annotations = rng.choice(["X", "Y", "Z"], size=(size,))
        out = _shuffle_group(cluster_annotations, libraries, rng)
        for c in libraries.cat.categories:
            assert set(out[libraries == c]) == set(cluster_annotations[libraries == c])
