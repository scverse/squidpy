from __future__ import annotations

import pytest

from squidpy._constants._constants import SpatialAutocorr
from squidpy.gr import spatial_autocorr


def test_spagft_incompatible_shapes():
    import numpy as np
    from scipy.sparse import lil_matrix

    from squidpy.gr._spagft import _spagft

    n = 10
    g = lil_matrix((n, n))
    for i in range(n):
        g[i, (i + 1) % n] = 1
        g[i, (i - 1) % n] = 1
    g = g.tocsr()
    vals = np.random.rand(5, 7)
    with pytest.raises(ValueError):
        _spagft(g, vals)


def test_spagft_svg_identification():
    import numpy as np
    from anndata import AnnData

    from squidpy.gr import spatial_autocorr

    n_cells = 50
    np.random.seed(42)
    spatial_pattern = np.sin(np.linspace(0, 2 * np.pi, n_cells))
    random_gene = np.random.normal(size=n_cells)
    X = np.vstack([spatial_pattern, random_gene])
    adata = AnnData(X=X.T)
    from scipy.sparse import lil_matrix

    g = lil_matrix((n_cells, n_cells))
    for i in range(n_cells):
        g[i, (i + 1) % n_cells] = 1
        g[i, (i - 1) % n_cells] = 1
    adata.obsp["spatial_connectivities"] = g.tocsr()
    df = spatial_autocorr(adata, mode="spagft", copy=True)
    assert "GFT" in df.columns
    assert df["GFT"].iloc[0] > df["GFT"].iloc[1]


def test_spagft_enum_recognition():
    # Check that the enum contains "spagft"
    assert hasattr(SpatialAutocorr, "SPAGFT")
    # Check that spatial_autocorr accepts the enum member
    import numpy as np
    from anndata import AnnData

    n_cells = 10
    np.random.seed(0)
    X = np.random.normal(size=(n_cells, 2))
    adata = AnnData(X=X)
    from scipy.sparse import lil_matrix

    g = lil_matrix((n_cells, n_cells))
    for i in range(n_cells):
        g[i, (i + 1) % n_cells] = 1
        g[i, (i - 1) % n_cells] = 1
    adata.obsp["spatial_connectivities"] = g.tocsr()
    # Should not raise
    df = spatial_autocorr(adata, mode=SpatialAutocorr.SPAGFT, copy=True)
    assert "GFT" in df.columns
