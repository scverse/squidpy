"""SpaGE port vs the original, on outputs upstream SpaGE computed (squidpy-ports' SpaGE bundle)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import squidpy as sq

from ._reference import Bundle, read_bundle

pytestmark = pytest.mark.internet

_BUNDLE_URL = "https://github.com/theislab/squidpy-ports/releases/download/spage-reference-v1/spage_reference.zarr.zip"
_BUNDLE_SHA256 = "8da69ca95afe63587be26e521b63e9240ed52f8cbc6a67bce20c929816a33662"


@pytest.fixture(scope="module")
def bundle() -> Bundle:
    return read_bundle(_BUNDLE_URL, _BUNDLE_SHA256)


@pytest.mark.parametrize("cosine_threshold", [0.3, 0.99])
@pytest.mark.parametrize("n_neighbors", [20, 50, 300])
def test_matches_upstream(bundle, cosine_threshold, n_neighbors):
    section = bundle.sections[f"cosine{round(cosine_threshold * 100)}_k{n_neighbors}"]
    genes, measured = list(bundle.fixtures["genes"]), list(bundle.fixtures["measured"])
    ref = AnnData(bundle.inputs["ref"], var=pd.DataFrame(index=genes))
    query = AnnData(bundle.inputs["query"], var=pd.DataFrame(index=measured))

    res = sq.experimental.tl.impute_spage(
        ref,
        query,
        genes=list(bundle.fixtures["unmeasured"]),
        n_pv=int(bundle.fixtures["N_PV"]),
        n_neighbors=n_neighbors,
        cosine_threshold=cosine_threshold,
        inplace=False,
    )

    # identical reference cells tied at the last neighbor may be kept by either implementation
    untied = np.setdiff1d(np.arange(query.n_obs), section["tied_rows"])
    np.testing.assert_allclose(res.to_numpy()[untied], section["imputed"][untied], rtol=1e-10, atol=1e-12)
