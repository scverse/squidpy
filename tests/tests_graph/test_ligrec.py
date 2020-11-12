import sys
from typing import Tuple, Optional, Sequence, NamedTuple

import pytest

from scanpy import settings as s
from anndata import AnnData
from scanpy.datasets import blobs

import numpy as np
import pandas as pd

from spatial_tools.graph import perm_test
from spatial_tools.graph.perm_test import PermutationTest

_CK = "leiden"
Interactions_t = Tuple[Sequence[str], Sequence[str]]
Complexes_t = Sequence[Tuple[str, str]]


class TestInvalidBehavior:
    def test_not_adata(self):
        with pytest.raises(TypeError, match=r"Expected `adata` to be of type `anndata.AnnData`"):
            perm_test(None, _CK)

    def test_adata_no_raw(self, adata: AnnData):
        del adata.raw
        with pytest.raises(AttributeError, match=r"No `.raw` attribute"):
            perm_test(adata, _CK)

    def test_raw_has_different_n_obs(self, adata: AnnData):
        adata.raw = blobs(n_observations=adata.n_obs + 1)
        with pytest.raises(ValueError, match=rf"Expected `{adata.n_obs}` cells in `.raw`"):
            perm_test(adata, _CK)

    def test_invalid_cluster_key(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(KeyError, match=r"Cluster key `'foobar'` not found"):
            perm_test(adata, cluster_key="foobar", interactions=interactions)

    def test_cluster_key_is_not_categorical(self, adata: AnnData, interactions: Interactions_t):
        adata.obs[_CK] = adata.obs[_CK].astype("string")
        with pytest.raises(TypeError, match=rf"Expected `adata.obs\[{_CK!r}\]` to be `categorical`"):
            perm_test(adata, _CK, interactions=interactions)

    def test_only_1_cluster(self, adata: AnnData, interactions: Interactions_t):
        adata.obs["foo"] = 1
        adata.obs["foo"] = adata.obs["foo"].astype("category")
        with pytest.raises(ValueError, match=r"Expected at least `2` clusters, found `1`."):
            perm_test(adata, "foo", interactions=interactions)

    def test_invalid_complex_policy(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError):
            perm_test(adata, _CK, interactions=interactions, complex_policy="foobar")

    def test_invalid_fdr_axis(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError):
            perm_test(adata, _CK, interactions=interactions, fdr_axis="foobar", fdr_method="fdr_bh")

    def test_too_few_permutations(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Expected `n_perms` to be positive"):
            perm_test(adata, _CK, interactions=interactions, n_perms=0)

    def test_invalid_interactions_type(self, adata: AnnData):
        with pytest.raises(TypeError, match=r"Expected either a `pandas.DataFrame`"):
            perm_test(adata, _CK, interactions=42)

    def test_invalid_interactions_dict(self, adata: AnnData):
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            perm_test(adata, _CK, interactions={"foo": ["foo"], "target": ["bar"]})
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            perm_test(adata, _CK, interactions={"source": ["foo"], "bar": ["bar"]})

    def test_invalid_interactions_dataframe(self, adata: AnnData, interactions: Interactions_t):
        df = pd.DataFrame(interactions, columns=["foo", "target"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            perm_test(adata, _CK, interactions=df)

        df = pd.DataFrame(interactions, columns=["source", "bar"])
        with pytest.raises(KeyError, match=r"Column .* is not in `interactions`."):
            perm_test(adata, _CK, interactions=df)

    def test_interactions_invalid_sequence(self, adata: AnnData, interactions: Interactions_t):
        interactions += ("foo", "bar", "bar")
        with pytest.raises(ValueError, match=r"Not all interactions are of length `2`."):
            perm_test(adata, _CK, interactions=interactions)

    def test_interactions_only_invalid_names(self, adata: AnnData):
        with pytest.raises(ValueError, match=r"After filtering by genes"):
            perm_test(adata, _CK, interactions=["foo", "bar", "baz"])

    def test_invalid_clusters(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(ValueError, match=r"Invalid cluster `'foo'`."):
            perm_test(adata, _CK, interactions=interactions, clusters=["foo"])

    def test_invalid_clusters_mix(self, adata: AnnData, interactions: Interactions_t):
        with pytest.raises(TypeError, match=r"Expected a `Sequence`."):
            perm_test(adata, _CK, interactions=interactions, clusters=["foo", ("bar", "baz")])

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
            .applymap(str.upper)
            .values,
        )

    def test_fdr_axis_works(self, adata: AnnData, interactions: Interactions_t):
        rc = perm_test(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            fdr_axis="clusters",
            seed=42,
            n_jobs=1,
            show_progress_bar=False,
            copy=True,
        )
        ri = perm_test(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            fdr_axis="interactions",
            n_jobs=1,
            show_progress_bar=False,
            seed=42,
            copy=True,
        )

        np.testing.assert_array_equal(np.where(np.isnan(rc.pvalues)), np.where(np.isnan(ri.pvalues)))
        mask = np.isnan(rc.pvalues)

        assert not np.allclose(rc.pvalues.values[mask], ri.pvalues.values[mask])

    def test_inplace_key_added(self, adata: AnnData, interactions: Interactions_t):
        assert "foobar" not in adata.uns
        res = perm_test(
            adata, _CK, interactions=interactions, n_perms=5, copy=False, key_added="foobar", show_progress_bar=False
        )

        assert res is None
        assert isinstance(adata.uns["foobar"], tuple)
        r = adata.uns["foobar"]
        assert len(r) == 3
        assert isinstance(r.means, pd.DataFrame)
        assert isinstance(r.pvalues, pd.DataFrame)
        assert isinstance(r.metadata, pd.DataFrame)

    def test_return_no_write(self, adata: AnnData, interactions: Interactions_t):
        assert "foobar" not in adata.uns
        r = perm_test(
            adata, _CK, interactions=interactions, n_perms=5, copy=True, key_added="foobar", show_progress_bar=False
        )

        assert "foobar" not in adata.uns
        assert len(r) == 3
        assert isinstance(r.means, pd.DataFrame)
        assert isinstance(r.pvalues, pd.DataFrame)
        assert isinstance(r.metadata, pd.DataFrame)

    @pytest.mark.parametrize("fdr_method", [None, "fdr_bh"])
    def test_pvals_in_correct_range(self, adata: AnnData, interactions: Interactions_t, fdr_method: Optional[str]):
        r = perm_test(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            copy=True,
            show_progress_bar=False,
            fdr_method=fdr_method,
            threshold=0,
        )

        if np.sum(np.isnan(r.pvalues.values)) == np.prod(r.pvalues.shape):
            assert fdr_method == "fdr_bh"
        else:
            assert np.nanmax(r.pvalues.values) <= 1.0, np.nanmax(r.pvalues.values)
            assert np.nanmin(r.pvalues.values) >= 0, np.nanmin(r.pvalues.values)

    def test_result_correct_index(self, adata: AnnData, interactions: Interactions_t):
        r = perm_test(adata, _CK, interactions=interactions, n_perms=5, copy=True, show_progress_bar=False)

        np.testing.assert_array_equal(r.means.index, r.pvalues.index)
        np.testing.assert_array_equal(r.pvalues.index, r.metadata.index)

        np.testing.assert_array_equal(r.means.columns, r.pvalues.columns)
        assert not np.array_equal(r.means.columns, r.metadata.columns)
        assert not np.array_equal(r.pvalues.columns, r.metadata.columns)

    def test_result_is_sparse(self, adata: AnnData, interactions: Interactions_t):
        interactions = pd.DataFrame(interactions, columns=["source", "target"])
        interactions["metadata"] = "foo"
        r = perm_test(adata, _CK, interactions=interactions, n_perms=5, seed=2, copy=True, show_progress_bar=False)

        assert r.means.sparse.density <= 0.15
        assert r.pvalues.sparse.density <= 0.95
        with pytest.raises(AttributeError, match=r"Can only use the '.sparse' accessor with Sparse data."):
            _ = r.metadata.sparse

        np.testing.assert_array_equal(r.metadata.columns, ["metadata"])
        np.testing.assert_array_equal(r.metadata["metadata"], interactions["metadata"])

    @pytest.mark.parametrize("n_jobs", [1, 2])
    def test_reproducibility_cores(self, adata: AnnData, interactions: Interactions_t, n_jobs: int):
        r1 = perm_test(
            adata,
            _CK,
            interactions=interactions,
            n_perms=25,
            copy=True,
            show_progress_bar=False,
            seed=42,
            n_jobs=n_jobs,
        )
        r2 = perm_test(
            adata,
            _CK,
            interactions=interactions,
            n_perms=25,
            copy=True,
            show_progress_bar=False,
            seed=42,
            n_jobs=n_jobs,
        )
        r3 = perm_test(
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
        np.testing.assert_allclose(r1.means, r2.means)
        np.testing.assert_allclose(r2.means, r3.means)
        np.testing.assert_allclose(r1.pvalues, r2.pvalues)

        assert not np.allclose(r3.pvalues, r1.pvalues)
        assert not np.allclose(r3.pvalues, r2.pvalues)

    def test_reproducibility_numba_off(self, adata: AnnData, interactions: Interactions_t, ligrec_no_numba: NamedTuple):
        r = perm_test(
            adata, _CK, interactions=interactions, n_perms=5, copy=True, show_progress_bar=False, seed=42, n_jobs=1
        )
        import pickle

        with open("ligrec_no_numba.pickle", "wb") as fout:
            pickle.dump(r, fout)

        np.testing.assert_array_equal(r.means.index, ligrec_no_numba.means.index)
        np.testing.assert_array_equal(r.means.columns, ligrec_no_numba.means.columns)
        np.testing.assert_array_equal(r.pvalues.index, ligrec_no_numba.pvalues.index)
        np.testing.assert_array_equal(r.pvalues.columns, ligrec_no_numba.pvalues.columns)

        np.testing.assert_allclose(r.means, ligrec_no_numba.means)
        np.testing.assert_allclose(r.pvalues, ligrec_no_numba.pvalues)
        np.testing.assert_array_equal(np.where(np.isnan(r.pvalues)), np.where(np.isnan(ligrec_no_numba.pvalues)))

    def test_logging(self, adata: AnnData, interactions: Interactions_t, capsys):
        s.logfile = sys.stderr
        s.verbosity = 4

        perm_test(
            adata,
            _CK,
            interactions=interactions,
            n_perms=5,
            copy=False,
            show_progress_bar=False,
            complex_policy="all",
        )

        err = capsys.readouterr().err

        assert "DEBUG: Removing duplicate interactions" in err
        assert "DEBUG: Removing duplicate genes in the data" in err
        assert "DEBUG: Creating all gene combinations within complexes" in err
        assert "DEBUG: Removing interactions with no genes in the data" in err
        assert "DEBUG: Removing genes not in any interaction" in err
        assert "Running `5` permutations on `25` interactions and `25` cluster combinations using `1` core(s)" in err
        assert "Adding `adata.uns['ligrec_test']`" in err
