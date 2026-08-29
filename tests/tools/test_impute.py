from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csr_matrix

from squidpy.tl import impute


def _make_adata(n_obs: int, genes: list[str], rng: np.random.Generator) -> AnnData:
    x = rng.normal(size=(n_obs, len(genes)))
    return AnnData(x, var=pd.DataFrame(index=genes))


class TestSpaGE:
    def test_spage_impute_dense(self):
        rng = np.random.default_rng(0)
        sc_genes = [f"g{i}" for i in range(10)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(40, sc_genes, rng)
        st_adata = _make_adata(20, st_genes, rng)

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=5,
            key_added="spage",
        )

        assert res is None
        assert "spage" in st_adata.obsm

        df = st_adata.obsm["spage"]
        assert df.shape == (st_adata.n_obs, 10)
        assert list(df.columns) == [f"g{i}" for i in range(10)]
        assert df.index.equals(st_adata.obs_names)

    def test_spage_impute_sparse(self):
        rng = np.random.default_rng(1)
        sc_genes = [f"g{i}" for i in range(8)]
        st_genes = [f"g{i}" for i in range(4)]

        sc_adata = _make_adata(30, sc_genes, rng)
        st_adata = _make_adata(15, st_genes, rng)
        sc_adata.X = csr_matrix(sc_adata.X)
        st_adata.X = csr_matrix(st_adata.X)

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=4,
            key_added="spage",
        )

        assert res is None
        df = st_adata.obsm["spage"]
        assert df.shape == (st_adata.n_obs, 8)
        assert list(df.columns) == [f"g{i}" for i in range(8)]

    def test_spage_impute_in_place_write(self):
        rng = np.random.default_rng(5)
        sc_genes = [f"g{i}" for i in range(9)]
        st_genes = [f"g{i}" for i in range(6)]

        sc_adata = _make_adata(25, sc_genes, rng)
        st_adata = _make_adata(12, st_genes, rng)

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=4,
            key_added="spage",
        )

        assert res is None
        assert "spage" in st_adata.obsm

    def test_spage_impute_copy_returns_dataframe(self):
        rng = np.random.default_rng(15)
        sc_genes = [f"g{i}" for i in range(9)]
        st_genes = [f"g{i}" for i in range(6)]

        sc_adata = _make_adata(25, sc_genes, rng)
        st_adata = _make_adata(12, st_genes, rng)

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=4,
            key_added="spage",
            copy=True,
        )

        assert isinstance(res, pd.DataFrame)
        assert res.shape == (st_adata.n_obs, 9)
        assert "spage" not in st_adata.obsm

    def test_spage_impute_genes_subset_order(self):
        rng = np.random.default_rng(6)
        sc_genes = [f"g{i}" for i in range(10)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(30, sc_genes, rng)
        st_adata = _make_adata(14, st_genes, rng)

        genes = ["g7", "g5"]
        res = impute(
            st_adata,
            sc_adata,
            genes=genes,
            method="spage",
            n_pv=3,
            n_neighbors=5,
            key_added="spage",
        )

        assert res is None
        df = st_adata.obsm["spage"]
        assert list(df.columns) == genes

    def test_spage_impute_cosine_threshold_too_strict(self):
        rng = np.random.default_rng(7)
        sc_genes = [f"g{i}" for i in range(8)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(22, sc_genes, rng)
        st_adata = _make_adata(11, st_genes, rng)

        with pytest.raises(ValueError, match="No effective principal vectors"):
            impute(
                st_adata,
                sc_adata,
                method="spage",
                n_pv=3,
                n_neighbors=4,
                cosine_threshold=1.1,
            )

    def test_spage_impute_n_neighbors_clamped(self):
        rng = np.random.default_rng(8)
        sc_genes = [f"g{i}" for i in range(7)]
        st_genes = [f"g{i}" for i in range(4)]

        sc_adata = _make_adata(6, sc_genes, rng)
        st_adata = _make_adata(5, st_genes, rng)

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=50,
            key_added="spage",
        )

        assert res is None
        df = st_adata.obsm["spage"]
        assert df.shape == (st_adata.n_obs, 7)

    def test_spage_impute_use_raw(self):
        rng = np.random.default_rng(9)
        sc_genes = [f"g{i}" for i in range(8)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(18, sc_genes, rng)
        st_adata = _make_adata(12, st_genes, rng)

        sc_adata.raw = sc_adata.copy()
        st_adata.raw = st_adata.copy()

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=4,
            key_added="spage",
            use_raw=True,
        )

        assert res is None
        assert "spage" in st_adata.obsm

    def test_spage_impute_layer(self):
        rng = np.random.default_rng(10)
        sc_genes = [f"g{i}" for i in range(8)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(18, sc_genes, rng)
        st_adata = _make_adata(12, st_genes, rng)

        sc_adata.layers["counts"] = sc_adata.X.copy()
        st_adata.layers["counts"] = st_adata.X.copy()

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=4,
            key_added="spage",
            layer="counts",
        )

        assert res is None
        assert "spage" in st_adata.obsm

    def test_spage_impute_shared_genes_are_kept(self):
        rng = np.random.default_rng(16)
        sc_genes = [f"g{i}" for i in range(10)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(40, sc_genes, rng)
        st_adata = _make_adata(20, st_genes, rng)

        impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=5,
            key_added="spage",
        )

        df = st_adata.obsm["spage"]
        assert df.shape == (st_adata.n_obs, 10)
        assert all(gene in df.columns for gene in st_genes)

    def test_spage_impute_invalid_genes(self):
        rng = np.random.default_rng(2)
        sc_genes = [f"g{i}" for i in range(6)]
        st_genes = [f"g{i}" for i in range(3)]

        sc_adata = _make_adata(20, sc_genes, rng)
        st_adata = _make_adata(10, st_genes, rng)

        with pytest.raises(ValueError, match="Genes not found in `sc_adata`"):
            impute(st_adata, sc_adata, method="spage", genes=["g4", "gX"], n_pv=2, n_neighbors=3)

    def test_spage_impute_no_shared_genes(self):
        rng = np.random.default_rng(3)
        sc_genes = [f"h{i}" for i in range(6)]
        st_genes = [f"g{i}" for i in range(3)]

        sc_adata = _make_adata(20, sc_genes, rng)
        st_adata = _make_adata(10, st_genes, rng)

        with pytest.raises(ValueError, match="No shared genes"):
            impute(st_adata, sc_adata, method="spage", n_pv=2, n_neighbors=3)

    def test_spage_impute_no_genes_to_impute(self):
        rng = np.random.default_rng(11)
        sc_genes = [f"g{i}" for i in range(5)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(20, sc_genes, rng)
        st_adata = _make_adata(10, st_genes, rng)

        with pytest.raises(ValueError, match="No genes to impute"):
            impute(st_adata, sc_adata, method="spage", genes=[], n_pv=2, n_neighbors=3)

    def test_spage_impute_n_pv_too_large(self):
        rng = np.random.default_rng(4)
        sc_genes = [f"g{i}" for i in range(7)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(20, sc_genes, rng)
        st_adata = _make_adata(10, st_genes, rng)

        with pytest.raises(ValueError, match="`n_pv` must be <= number of shared genes"):
            impute(st_adata, sc_adata, method="spage", n_pv=10, n_neighbors=3)


class TestImputeDispatch:
    def test_invalid_method_raises(self):
        rng = np.random.default_rng(13)
        sc_genes = [f"g{i}" for i in range(6)]
        st_genes = [f"g{i}" for i in range(4)]

        sc_adata = _make_adata(16, sc_genes, rng)
        st_adata = _make_adata(8, st_genes, rng)

        with pytest.raises(ValueError, match="one of"):
            impute(st_adata, sc_adata, method="tangram")

    def test_spage_args_are_supported(self):
        rng = np.random.default_rng(14)
        sc_genes = [f"g{i}" for i in range(8)]
        st_genes = [f"g{i}" for i in range(5)]

        sc_adata = _make_adata(20, sc_genes, rng)
        st_adata = _make_adata(10, st_genes, rng)

        res = impute(
            st_adata,
            sc_adata,
            method="spage",
            n_pv=3,
            n_neighbors=4,
        )

        assert res is None
        assert "spage" in st_adata.obsm
