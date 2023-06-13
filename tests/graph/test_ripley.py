import numpy as np
import pytest
from anndata import AnnData
from squidpy._constants._constants import RipleyStat
from squidpy.gr import ripley

CLUSTER_KEY = "leiden"


@pytest.mark.parametrize("mode", list(RipleyStat))
def test_ripley_modes(adata_ripley: AnnData, mode: RipleyStat):
    adata = adata_ripley

    ripley(adata, cluster_key=CLUSTER_KEY, mode=mode.s)

    UNS_KEY = f"{CLUSTER_KEY}_ripley_{mode}"

    # assert uns
    assert UNS_KEY in adata.uns.keys()
    assert f"{mode}_stat" in adata.uns[UNS_KEY].keys()
    assert "sims_stat" in adata.uns[UNS_KEY].keys()
    assert "bins" in adata.uns[UNS_KEY].keys()
    assert "pvalues" in adata.uns[UNS_KEY].keys()

    obs_df = adata.uns[UNS_KEY][f"{mode}_stat"]
    sims_df = adata.uns[UNS_KEY]["sims_stat"]
    bins = adata.uns[UNS_KEY]["bins"]
    pvalues = adata.uns[UNS_KEY]["pvalues"]

    # assert shapes
    np.testing.assert_array_equal(adata.obs[CLUSTER_KEY].cat.categories, obs_df[CLUSTER_KEY].cat.categories)
    assert obs_df.shape[1] == sims_df.shape[1]
    assert pvalues.shape[0] == adata.obs[CLUSTER_KEY].cat.categories.shape[0]
    assert bins.shape[0] == 50


@pytest.mark.parametrize("mode", list(RipleyStat))
@pytest.mark.parametrize(
    "n_simulations",
    [20, 50],
)
@pytest.mark.parametrize(
    "n_observations",
    [10, 100],
)
@pytest.mark.parametrize(
    "max_dist",
    [None, 1000],
)
@pytest.mark.parametrize(
    "n_steps",
    [2, 50, 100],
)
def test_ripley_results(
    adata_ripley: AnnData, mode: RipleyStat, n_simulations: int, n_observations: int, max_dist: np.float_, n_steps: int
):
    adata = adata_ripley
    n_clusters = adata.obs[CLUSTER_KEY].cat.categories.shape[0]

    res = ripley(
        adata,
        cluster_key=CLUSTER_KEY,
        mode=mode.s,
        n_simulations=n_simulations,
        n_observations=n_observations,
        max_dist=max_dist,
        n_steps=n_steps,
        copy=True,
    )

    obs_df = res[f"{mode}_stat"]
    sims_df = res["sims_stat"]
    bins = res["bins"]
    pvalues = res["pvalues"]

    # assert shapes
    assert obs_df.shape == (n_steps * n_clusters, 3)
    assert bins.shape == (n_steps,)
    assert sims_df.shape == (n_steps * n_simulations, 3)
    assert pvalues.shape == (n_clusters, n_steps)

    # assert first values is 0
    assert sims_df.bins.values[0] == 0.0
    assert sims_df.bins.values[0] == obs_df.bins.values[0]
    assert sims_df.stats.values[0] == 0.0
    assert sims_df.stats.values[0] == obs_df.stats.values[0]

    # assert n_zeros == n_clusters
    idx = np.nonzero(obs_df.bins.values)[0]
    assert idx.shape[0] == n_steps * n_clusters - n_clusters
