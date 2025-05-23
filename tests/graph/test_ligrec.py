from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from itertools import product
from time import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from pandas.testing import assert_frame_equal
from scanpy import settings as s
from scanpy.datasets import blobs
from scipy.sparse import csc_matrix

from squidpy._constants._pkg_constants import Key
from squidpy.gr import ligrec
from squidpy.gr._ligrec import PermutationTest

_CK = "leiden"
Interactions_t = tuple[Sequence[str], Sequence[str]]
Complexes_t = Sequence[tuple[str, str]]


class TestInvalidBehavior:
    def test_not_adata(self):
        with pytest.raises(TypeError, match=r"Expected `adata` to be of type `anndata.AnnData`"):
            ligrec(None, _CK)

    def test_adata_no_raw(self, adata: AnnData):
        del adata.raw
        with pytest.raises(AttributeError, match=r"No `.raw` attribute"):
            ligrec(adata, _CK, use_raw=True)

    def test_raw_has_different_n_obs(self, adata: AnnData):
        adata.raw = blobs(n_observations=adata.n_obs + 1)
        # raise below happend with anndata < 0.9
        # with pytest.raises(ValueError, match=rf"Expected `{adata.n_obs}` cells in `.raw`"):
        with pytest.raises(ValueError, match=rf"Index length mismatch: {adata.n_obs} vs. {adata.n_obs + 1}"):
            ligrec(adata, _CK)

    def test_invalid_cluster_key(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(KeyError, match=r"Cluster key `foobar` not found"):
            ligrec(adata, cluster_key="foobar", interactions=interactions)

    def test_cluster_key_is_not_categorical(self, adata: AnnData, interactions: Interactions_t):
        adata.obs[_CK] = adata.obs[_CK].astype("string")
        with pytest.raises(TypeError, match=rf"Expected `adata.obs\[{_CK!r}\]` to be `categorical`"):
            ligrec(adata, _CK, interactions=interactions)

    def test_only_1_cluster(self, adata: AnnData, interactions: Interactions_t):
        adata.obs["foo"] = 1
        adata.obs["foo"] = adata.obs["foo"].astype("category")
        with pytest.raises(ValueError, match=r"Expected at least `2` clusters, found `1`."):
            ligrec(adata, "foo", interactions=interactions)

    def test_invalid_complex_policy(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Invalid option `foobar` for `ComplexPolicy`."):
            ligrec(adata, _CK, interactions=interactions, complex_policy="foobar")

    def test_invalid_fdr_axis(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Invalid option `foobar` for `CorrAxis`."):
            ligrec(adata, _CK, interactions=interactions, corr_axis="foobar", corr_method="fdr_bh")

    def test_too_few_permutations(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Expected `n_perms` to be positive"):
            ligrec(adata, _CK, interactions=interactions, n_perms=0)

    def test_invalid_interactions_type(self, adata: AnnData):
        with pytest.raises(TypeError, match=r"Expected either a `pandas.DataFrame`"):
            ligrec(adata, _CK, interactions=42)

    def test_invalid_interactions_dict(self, adata: AnnData):
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions={"foo": ["foo"], "target": ["bar"]})
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions={"source": ["foo"], "bar": ["bar"]})

    def test_invalid_interactions_dataframe(self, adata: AnnData, interactions: Interactions_t):
        df = pd.DataFrame(interactions, columns=["foo", "target"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions=df)

        df = pd.DataFrame(interactions, columns=["source", "bar"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            ligrec(adata, _CK, interactions=df)

    def test_interactions_invalid_sequence(self, adata: AnnData, interactions: Interactions_t):
        interactions += ("foo", "bar", "bar")  # type: ignore
        with pytest.raises(ValueError, match=r"Not all interactions are of length `2`."):
            ligrec(adata, _CK, interactions=interactions)

    def test_interactions_only_invalid_names(self, adata: AnnData):
        with pytest.raises(ValueError, match=r"After filtering by genes"):
            ligrec(adata, _CK, interactions=["foo", "bar", "baz"])

    def test_invalid_clusters(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Invalid cluster `'foo'`."):
            ligrec(adata, _CK, interactions=interactions, clusters=["foo"])

    def test_invalid_clusters_mix(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Expected a `tuple` of length `2`, found `3`."):
            ligrec(adata, _CK, interactions=interactions, clusters=["foo", ("bar", "baz")])


class TestValidBehavior:
    def test_do_not_use_raw(self, adata: AnnData, interactions: Interactions_t):
        del adata.raw

        _ = PermutationTest(adata, use_raw=False)

    def test_all_genes_capitalized(self, adata: AnnData, interactions: Interactions_t):
        pt = PermutationTest(adata).prepare(interactions=interactions)
        genes = pd.Series([g for gs in pt.interactions[["source", "target"]].values for g in gs], dtype="string")

        np.testing.assert_array_equal(genes.values, genes.str.upper().values)
        np.testing.assert_array_equal(pt._data.columns, pt._data.columns.str.upper())

    def test_complex_policy_min(self, adata: AnnData, complexes: Complexes_t):
        g = adata.raw.var_names
        pt = PermutationTest(adata).prepare(interactions=complexes, complex_policy="min")

        assert pt.interactions.shape == (5, 2)

        assert np.mean(adata.raw[:, g[2]].X) > np.mean(adata.raw[:, g[3]].X)  # S
        assert np.mean(adata.raw[:, g[6]].X) < np.mean(adata.raw[:, g[7]].X)  # T
        assert np.mean(adata.raw[:, g[8]].X) < np.mean(adata.raw[:, g[9]].X)  # S
        assert np.mean(adata.raw[:, g[10]].X) > np.mean(adata.raw[:, g[11]].X)  # T

        np.testing.assert_array_equal(pt.interactions["source"], list(map(str.upper, [g[0], g[3], g[5], g[8], g[12]])))
        np.testing.assert_array_equal(pt.interactions["target"], list(map(str.upper, [g[1], g[4], g[6], g[11], g[13]])))

    def test_complex_policy_all(self, adata: AnnData, complexes: Complexes_t):
        g = adata.raw.var_names
        pt = PermutationTest(adata).prepare(interactions=complexes, complex_policy="all")

        assert pt.interactions.shape == (10, 2)

        np.testing.assert_array_equal(
            pt.interactions.values,
            pd.DataFrame(
                [
                    [g[0], g[1]],
                    [g[2], g[4]],
                    [g[3], g[4]],
                    [g[5], g[6]],
                    [g[5], g[7]],
                    [g[8], g[10]],
                    [g[8], g[11]],
                    [g[9], g[10]],
                    [g[9], g[11]],
                    [g[12], g[13]],
                ]
            )
            .map(str.upper)
            .values,
        )

    def test_fdr_axis_works(self, adata: AnnData, interactions: Interactions_t):
        rc = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            corr_axis="clusters",
            seed=42,
            n_jobs=1,
            show_progress_bar=False,
            copy=True,
        )
        ri = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            corr_axis="interactions",
            n_jobs=1,
            show_progress_bar=False,
            seed=42,
            copy=True,
        )

        np.testing.assert_array_equal(np.where(np.isnan(rc["pvalues"])), np.where(np.isnan(ri["pvalues"])))
        mask = np.isnan(rc["pvalues"])

        assert not np.allclose(rc["pvalues"].values[mask], ri["pvalues"].values[mask])

    def test_inplace_default_key(self, adata: AnnData, interactions: Interactions_t):
        key = Key.uns.ligrec(_CK)
        assert key not in adata.uns
        res = ligrec(adata, _CK, interactions=interactions, n_perms=5, copy=False, show_progress_bar=False)

        assert res is None
        assert isinstance(adata.uns[key], dict)
        r = adata.uns[key]
        assert len(r) == 3
        assert isinstance(r["means"], pd.DataFrame)
        assert isinstance(r["pvalues"], pd.DataFrame)
        assert isinstance(r["metadata"], pd.DataFrame)

    def test_inplace_key_added(self, adata: AnnData, interactions: Interactions_t):
        assert "foobar" not in adata.uns
        res = ligrec(
            adata, _CK, interactions=interactions, n_perms=5, copy=False, key_added="foobar", show_progress_bar=False
        )

        assert res is None
        assert isinstance(adata.uns["foobar"], dict)
        r = adata.uns["foobar"]
        assert len(r) == 3
        assert isinstance(r["means"], pd.DataFrame)
        assert isinstance(r["pvalues"], pd.DataFrame)
        assert isinstance(r["metadata"], pd.DataFrame)

    def test_return_no_write(self, adata: AnnData, interactions: Interactions_t):
        assert "foobar" not in adata.uns
        r = ligrec(
            adata, _CK, interactions=interactions, n_perms=5, copy=True, key_added="foobar", show_progress_bar=False
        )

        assert "foobar" not in adata.uns
        assert len(r) == 3
        assert isinstance(r["means"], pd.DataFrame)
        assert isinstance(r["pvalues"], pd.DataFrame)
        assert isinstance(r["metadata"], pd.DataFrame)

    @pytest.mark.parametrize("fdr_method", [None, "fdr_bh"])
    def test_pvals_in_correct_range(self, adata: AnnData, interactions: Interactions_t, fdr_method: str | None):
        r = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            copy=True,
            show_progress_bar=False,
            corr_method=fdr_method,
            threshold=0,
        )

        if np.sum(np.isnan(r["pvalues"].values)) == np.prod(r["pvalues"].shape):
            assert fdr_method == "fdr_bh"
        else:
            assert np.nanmax(r["pvalues"].values) <= 1.0, np.nanmax(r["pvalues"].values)
            assert np.nanmin(r["pvalues"].values) >= 0, np.nanmin(r["pvalues"].values)

    def test_result_correct_index(self, adata: AnnData, interactions: Interactions_t):
        r = ligrec(adata, _CK, interactions=interactions, n_perms=5, copy=True, show_progress_bar=False)

        np.testing.assert_array_equal(r["means"].index, r["pvalues"].index)
        np.testing.assert_array_equal(r["pvalues"].index, r["metadata"].index)

        np.testing.assert_array_equal(r["means"].columns, r["pvalues"].columns)
        assert not np.array_equal(r["means"].columns, r["metadata"].columns)
        assert not np.array_equal(r["pvalues"].columns, r["metadata"].columns)

    def test_result_is_sparse(self, adata: AnnData, interactions: Interactions_t):
        interactions = pd.DataFrame(interactions, columns=["source", "target"])
        if TYPE_CHECKING:
            assert isinstance(interactions, pd.DataFrame)
        interactions["metadata"] = "foo"
        r = ligrec(adata, _CK, interactions=interactions, n_perms=5, seed=2, copy=True, show_progress_bar=False)

        assert r["means"].sparse.density <= 0.15
        assert r["pvalues"].sparse.density <= 0.95

        with pytest.raises(AttributeError, match=r"Can only use the '.sparse' accessor with Sparse data."):
            _ = r["metadata"].sparse

        np.testing.assert_array_equal(r["metadata"].columns, ["metadata"])
        np.testing.assert_array_equal(r["metadata"]["metadata"], interactions["metadata"])

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_reproducibility_cores(self, adata: AnnData, interactions: Interactions_t, n_jobs: int):
        r1 = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=25,
            copy=True,
            show_progress_bar=False,
            seed=42,
            n_jobs=n_jobs,
        )
        r2 = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=25,
            copy=True,
            show_progress_bar=False,
            seed=42,
            n_jobs=n_jobs,
        )
        r3 = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=25,
            copy=True,
            show_progress_bar=False,
            seed=43,
            n_jobs=n_jobs,
        )

        assert r1 is not r2
        np.testing.assert_allclose(r1["means"], r2["means"])
        np.testing.assert_allclose(r2["means"], r3["means"])
        np.testing.assert_allclose(r1["pvalues"], r2["pvalues"])

        assert not np.allclose(r3["pvalues"], r1["pvalues"])
        assert not np.allclose(r3["pvalues"], r2["pvalues"])

    def test_reproducibility_numba_parallel_off(self, adata: AnnData, interactions: Interactions_t):
        t1 = time()
        r1 = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=25,
            copy=True,
            show_progress_bar=False,
            seed=42,
            numba_parallel=False,
        )
        t1 = time() - t1

        t2 = time()
        r2 = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=25,
            copy=True,
            show_progress_bar=False,
            seed=42,
            numba_parallel=True,
        )
        t2 = time() - t2

        assert r1 is not r2
        # for such a small data, overhead from parallelization is too high
        assert t1 <= t2, (t1, t2)
        np.testing.assert_allclose(r1["means"], r2["means"])
        np.testing.assert_allclose(r1["pvalues"], r2["pvalues"])

    def test_paul15_correct_means(self, paul15: AnnData, paul15_means: pd.DataFrame):
        res = ligrec(
            paul15,
            "paul15_clusters",
            interactions=list(paul15_means.index.to_list()),
            corr_method=None,
            copy=True,
            show_progress_bar=False,
            threshold=0.01,
            seed=0,
            n_perms=1,
            n_jobs=1,
        )

        np.testing.assert_array_equal(res["means"].index, paul15_means.index)
        np.testing.assert_array_equal(res["means"].columns, paul15_means.columns)
        np.testing.assert_allclose(res["means"].values, paul15_means.values)

    def test_reproducibility_numba_off(
        self, adata: AnnData, interactions: Interactions_t, ligrec_no_numba: Mapping[str, pd.DataFrame]
    ):
        r = ligrec(
            adata, _CK, interactions=interactions, n_perms=5, copy=True, show_progress_bar=False, seed=42, n_jobs=1
        )
        np.testing.assert_array_equal(r["means"].index, ligrec_no_numba["means"].index)
        np.testing.assert_array_equal(r["means"].columns, ligrec_no_numba["means"].columns)
        np.testing.assert_array_equal(r["pvalues"].index, ligrec_no_numba["pvalues"].index)
        np.testing.assert_array_equal(r["pvalues"].columns, ligrec_no_numba["pvalues"].columns)

        np.testing.assert_allclose(r["means"], ligrec_no_numba["means"])
        np.testing.assert_allclose(r["pvalues"], ligrec_no_numba["pvalues"])
        np.testing.assert_array_equal(np.where(np.isnan(r["pvalues"])), np.where(np.isnan(ligrec_no_numba["pvalues"])))

    def test_logging(self, adata: AnnData, interactions: Interactions_t, capsys):
        s.logfile = sys.stderr
        s.verbosity = 4

        ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            copy=False,
            show_progress_bar=False,
            complex_policy="all",
            key_added="ligrec_test",
            n_jobs=2,
        )

        err = capsys.readouterr().err

        assert "DEBUG: Removing duplicate interactions" in err
        assert "DEBUG: Removing duplicate genes in the data" in err
        assert "DEBUG: Creating all gene combinations within complexes" in err
        assert "DEBUG: Removing interactions with no genes in the data" in err
        assert "DEBUG: Removing genes not in any interaction" in err
        assert "Running `5` permutations on `25` interactions and `25` cluster combinations using `2` core(s)" in err
        assert "Adding `adata.uns['ligrec_test']`" in err

    def test_non_uniqueness(self, adata: AnnData, interactions: Interactions_t):
        # add complexes
        expected = {(r.upper(), l.upper()) for r, l in interactions}
        interactions += (  # type: ignore
            (f"{interactions[-1][0]}_{interactions[-1][1]}", f"{interactions[-2][0]}_{interactions[-2][1]}"),
        ) * 2
        interactions += interactions[:3]  # type: ignore
        res = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=1,
            copy=True,
            show_progress_bar=False,
            seed=42,
            numba_parallel=False,
        )

        assert len(res["pvalues"]) == len(expected)
        assert set(res["pvalues"].index.to_list()) == expected

    @pytest.mark.xfail(reason="AnnData cannot handle writing MultiIndex")
    def test_writeable(self, adata: AnnData, interactions: Interactions_t, tmpdir):
        ligrec(adata, _CK, interactions=interactions, n_perms=5, copy=False, show_progress_bar=False, key_added="foo")
        res = adata.uns["foo"]

        sc.write(tmpdir / "ligrec.h5ad", adata)
        bdata = sc.read(tmpdir / "ligrec.h5ad")

        for key in ["means", "pvalues", "metadata"]:
            assert_frame_equal(res[key], bdata.uns["foo"][key])

    @pytest.mark.parametrize("use_raw", [False, True])
    def test_gene_symbols(self, adata: AnnData, use_raw: bool):
        gene_ids = adata.var["gene_ids"]
        interactions = tuple(product(gene_ids[:5], gene_ids[:5]))
        res = ligrec(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            use_raw=use_raw,
            copy=True,
            show_progress_bar=False,
            gene_symbols="gene_ids",
        )

        np.testing.assert_array_equal(res["means"].index, pd.MultiIndex.from_tuples(interactions))
        np.testing.assert_array_equal(res["pvalues"].index, pd.MultiIndex.from_tuples(interactions))
        np.testing.assert_array_equal(res["metadata"].index, pd.MultiIndex.from_tuples(interactions))

    def test_none_source_target(self, adata: AnnData):
        pt = PermutationTest(adata).prepare(
            {"source": [None, adata.var_names[0]], "target": [None, adata.var_names[1]]}
        )
        assert isinstance(pt.interactions, pd.DataFrame)
        assert len(pt.interactions) == 1

    def test_ligrec_nan_counts(self):
        """
        For the test case with 2 clusters (A, B) and 3 gene pairs (Gene1→Gene2, Gene2→Gene3, Gene3→Gene1):

        The mask is computed for each gene in each cluster as:
        mask[gene, cluster] = (number of cells with value > 0) / (total cells in cluster) >= threshold

        Number of cells with value > 0 in each cluster:
        Cluster A: [1, 3, 0]
        Cluster B: [1, 0, 3]

        Number of cells with value > 0 in each cluster divided by total number of cells in the cluster:
        Cluster A: [1/3, 3/3, 0/3] = [0.33, 1.0, 0.0]
        Cluster B: [1/3, 0/3, 3/3] = [0.33, 0.0, 1.0]

        Using threshold=0.8 on this data, the mask is:
        Cluster A: [False, True, False]
        Cluster B: [False, False, True]

        A value in the result becomes NaN if either:
        - The ligand's mask is False in the source cluster, OR
        - The receptor's mask is False in the target cluster

        Only in one combination, the mask is both True in the source and target cluster.
        This is the case for Gene2→Gene3 in A→B.

        This means from all the possible cluster pairs (A→A, A→B, B→A, B→B) and gene pairs (Gene1→Gene2, Gene2→Gene3, Gene3→Gene1),
        (4 cluster pairs * 3 gene pairs = 12 combinations) only one combination is non-NaN.

        Therefore, the total number of NaNs is 11.

        The expected p-values are:
        cluster_1        A         B
        cluster_2        A    B    A    B
        source target
        GENE1  GENE2   NaN  NaN  NaN  NaN
        GENE2  GENE3   NaN  0.0  NaN  NaN
        GENE3  GENE1   NaN  NaN  NaN  NaN

        """
        # only Gene2→Gene3 is non-NaN
        #

        expected_pvalues = np.array(
            [
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    0.0,
                    np.nan,
                    np.nan,
                ],
                [
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                ],
            ]
        )

        expected_nans = 11
        # Setup test data
        threshold = 0.8
        interactions = pd.DataFrame({"source": ["Gene1", "Gene2", "Gene3"], "target": ["Gene2", "Gene3", "Gene1"]})

        # Create sparse matrix with test data
        X = csc_matrix(
            [
                [1.0, 0.1, 0.0],  # A1
                [0.0, 1.0, 0.0],  # A2
                [0.0, 1.0, 0.0],  # A3
                [0.1, 0.0, 1.0],  # B1
                [0.0, 0.0, 1.0],  # B2
                [0.0, 0.0, 1.0],  # B3
            ]
        )

        # Create AnnData object
        adata = AnnData(
            X=X,
            obs=pd.DataFrame({"cluster": ["A"] * 3 + ["B"] * 3}, index=[f"cell{i}" for i in range(1, 7)]),
            var=pd.DataFrame(index=["Gene1", "Gene2", "Gene3"]),
        )
        adata.obs["cluster"] = adata.obs["cluster"].astype("category")

        # Run ligrec and compare NaN counts
        res = ligrec(
            adata, cluster_key="cluster", interactions=interactions, threshold=threshold, use_raw=False, copy=True
        )

        actual_nans = np.sum(np.isnan(res["pvalues"].values))

        assert actual_nans == expected_nans, f"NaN count mismatch: expected {expected_nans}, got {actual_nans}"
        np.testing.assert_array_equal(res["pvalues"].values, expected_pvalues)
