"""Plotting for graph functions."""

from typing import Union
from anndata import AnnData
import numpy as np
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc


def centrality_scores(adata: AnnData, centrality_scores_key: str = "centrality_scores", selected_score: Union[str, None] = None):
    """
    Plot centrality scores as seaborn stripplot.

    Parameters
    ----------
    adata
        The AnnData object.
    centrality_scores_key
        Key to centrality_scores_key in uns.
    selected_score
        Whether to plot all scores or only just a selected one.

    Returns
    -------
    None
    """
    if centrality_scores_key in adata.uns_keys():
        df = adata.uns[centrality_scores_key]
    else:
        raise ValueError(
            "centrality_scores_key %s not recognized. Choose a different key or run first "
            "nhood.centrality_scores(adata) on your AnnData object." % centrality_scores_key
        )
    var = DataFrame(df.columns, columns=[centrality_scores_key])
    var['index'] = var[centrality_scores_key]
    var = var.set_index('index')

    obs = DataFrame(df['cluster']).rename(columns={'cluster': 'louvain'})

    intermediate_adata = AnnData(
        X=np.array(df),
        obs=obs,
        var=var
    )

    if selected_score is not None:
        sc.pl.scatter(
            intermediate_adata,
            x=selected_score,
            y='louvain',
            color='louvain',
            size=1000,
            title=''
        )
    else:
        plt.ioff()
        ncols = len(intermediate_adata.var.index) - 1
        fig, ax = plt.subplots(
            nrows=1, ncols=ncols,
            figsize=(4 * ncols, 6)
        )
        for i in range(ncols):
            x = list(intermediate_adata.var.index)[i + 1]
            sc.set_figure_params(figsize=[4, 6])
            scatter = sc.pl.scatter(
                intermediate_adata,
                x=str(x),
                y='louvain',
                size=1000,
                frameon=True,
                ax=ax[i],
                show=False
            )
            if i > 0:
                ax[i].set_ylabel('')

        plt.show()
        plt.close(fig)
        plt.ion()


def interaction_matrix(adata: AnnData, cluster_interactions_key: str = "interaction_matrix"):
    """
    Plot cluster interactions as matshow plot.

    Parameters
    ----------
    adata
        The AnnData object.
    cluster_interactions_key
        Key to cluster_interactions_key in uns.

    Returns
    -------
    None
    """
    if cluster_interactions_key in adata.uns_keys():
        int_matrix = adata.uns[cluster_interactions_key]
    else:
        raise ValueError(
            "cluster_interactions_key %s not recognized. Choose a different key or run first "
            "nhood.cluster_interactions(adata) on your AnnData object." % cluster_interactions_key
        )

    array = int_matrix[0]
    clusters = DataFrame(int_matrix[1], columns=['cluster'])
    clusters["louvain"] = clusters["cluster"].astype('category')
    clusters = clusters.set_index('cluster')

    intermediate_adata = AnnData(
        X=array,
        obs=clusters,
        var=clusters
    )

    sc.pl.heatmap(intermediate_adata, intermediate_adata.var_names, 'louvain')


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
