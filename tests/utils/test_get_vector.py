"""``get_vector`` reads ``.raw`` without scanpy's ``use_raw`` FutureWarning."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix

from squidpy._compat import get_vector


def test_get_vector_raw_no_warning():
    raw_x = np.random.default_rng(0).random((5, 3))
    adata = AnnData(np.zeros((5, 2)), var=pd.DataFrame(index=["a", "b"]))
    adata.raw = AnnData(csr_matrix(raw_x), var=pd.DataFrame(index=["a", "b", "c"]))

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always", FutureWarning)
        vec = get_vector(adata, "c", use_raw=True)
    assert [w for w in record if issubclass(w.category, FutureWarning)] == []
    np.testing.assert_array_equal(vec, raw_x[:, 2])
