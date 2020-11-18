"""Plotting for graph functions."""

from typing import Union

import scanpy as sc
from anndata import AnnData

import numpy as np
import pandas as pd
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt


def centrality_scores(
    adata: AnnData, cluster_key: str, selected_score: Union[str, None] = None, *args, **kwargs
) -> None:
    """
    Plot centrality scores as seaborn stripplot.

    Parameters
    ----------
    adata
        The AnnData object.
    cluster_key
        Key to cluster_interactions_key in uns.
    selected_score
        Whether to plot all scores or only just a selected one.

    Returns
    -------
    None
    """
    scores_key = f"{cluster_key}_centrality_scores"
    if scores_key in adata.uns_keys():
        df = adata.uns[scores_key]
    else:
        raise ValueError(
            f"centrality_scores_key {scores_key} not found. \n"
            "Choose a different key or run first nhood.centrality_scores(adata)"
        )
    var = DataFrame(df.columns, columns=[scores_key])
    var["index"] = var[scores_key]
    var = var.set_index("index")

    cat = adata.obs[cluster_key].cat.categories.values.astype(str)
    idx = {cluster_key: pd.Categorical(cat, categories=cat)}

    ad = AnnData(X=np.array(df), obs=idx, var=var)

    colors_key = f"{cluster_key}_colors"
    if colors_key in adata.uns.keys():
        ad.uns[colors_key] = adata.uns[colors_key]

    sc.set_figure_params(figsize=(4, 6))
    if selected_score is not None:
        sc.pl.scatter(ad, x=selected_score, y=cluster_key, color=cluster_key, size=1000, title="", frameon=True, *args, **kwargs)
    else:
        plt.ioff()
        nrows = len(ad.var.index) - 1
        fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=(4, 6 * nrows))
        for i in range(nrows):
            x = list(ad.var.index)[i + 1]
            sc.pl.scatter(ad, x=str(x), y=cluster_key, color=cluster_key, size=1000, ax=ax[i], show=False, frameon=True, *args, **kwargs)
            if i > 0:
                ax[i].set_ylabel("")

        plt.show()
        plt.close(fig)
        plt.ion()
    # resetting scanpy plotting params to defaults
    sc.set_figure_params()

def interaction_matrix(adata: AnnData, cluster_key: str, *args, **kwargs) -> None:
    # noqa: D205
    """
    Plot cluster interaction matrix, as computed with graph.interaction_matrix.
    Wraps scanpy.plotting.heatmap. Args and kwargs can be passed.

    Parameters
    ----------
    adata
        The AnnData object.
    cluster_key
        Key to cluster_interactions_key in uns.

    Returns
    -------
    None
    """
    int_key = f"{cluster_key}_interactions"
    if int_key in adata.uns_keys():
        array = adata.uns[int_key]
    else:
        raise ValueError(
            f"cluster_interactions_key {int_key} not found. \n"
            "Choose a different key or run first nhood.interaction_matrix(adata)"
        )
    cat = adata.obs[cluster_key].cat.categories.values.astype(str)
    idx = {cluster_key: pd.Categorical(cat, categories=cat)}

    ad = AnnData(X=array, obs=idx, var=idx)

    colors_key = f"{cluster_key}_colors"
    if colors_key in adata.uns.keys():
        ad.uns[colors_key] = adata.uns[colors_key]
    sc.pl.heatmap(ad, var_names=ad.var_names, groupby=cluster_key, *args, **kwargs)


def plot_ripley_k(
    adata: AnnData,
    cluster_key: str,
    df: Union[DataFrame, None] = None,
):
    """
    Plot Ripley K estimate for each cluster.

    Parameters
    ----------
    adata
        The AnnData object.
    cluster_key
        cluster key used to compute ripley's K
        and stored in adata.uns[f"ripley_k_{cluster_key}"].

    Returns
    -------
    None
    """
    if df is None:
        try:
            df = adata.uns[f"ripley_k_{cluster_key}"]

            try:
                hue_order = list(np.sort(adata.obs[cluster_key].unique()))
                palette = list(adata.uns[f"{cluster_key}_colors"])
            except ValueError:
                raise Warning(f"\there is no color palette in adata_uns for {cluster_key}\n")

        except ValueError:
            raise ValueError(
                f"\\looks like `riply_k_{cluster_key}` was not used\n\n"
                "\\is not present in adata.uns,\n"
                "\tplease rerun ripley_k or pass\n"
                "\ta dataframe"
            )
    else:
        hue_order = None
        palette = None

    sns.lineplot(
        "distance",
        "ripley_k",
        hue="leiden",
        hue_order=hue_order,
        data=df,
        palette=palette,
    )
