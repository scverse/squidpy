"""Plotting for graph functions."""

from typing import Tuple, Union, Optional
from pathlib import Path

import scanpy as sc
from anndata import AnnData

import numpy as np
import pandas as pd
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt

from squidpy._docs import d
from squidpy.plotting._utils import save_fig
from squidpy.constants._pkg_constants import Key


@d.dedent
def spatial_graph(
    adata: AnnData,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs,
) -> None:
    """
    Plot wrapper for :mod:`scanpy` plotting function for spatial graphs.

    Parameters
    ----------
    %(adata)s
    %(plotting)s
    kwargs
        Keyword arguments for :func:`scanpy.pl.embedding`.

    Returns
    -------
    %(plotting_returns)s
    """
    # TODO: expose keys?
    conns_key = "spatial_connectivities"
    neighbors_dict = adata.uns[Key.uns.spatial] = {}
    neighbors_dict["connectivities_key"] = conns_key
    neighbors_dict["distances_key"] = "dummy"  # TODO?

    fig, ax = plt.subplots(dpi=dpi, figsize=figsize)
    sc.pl.embedding(
        adata,
        basis=Key.obsm.spatial,
        edges=True,
        neighbors_key="spatial",
        edges_width=4,
        ax=ax,
        **kwargs,
    )

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    selected_score: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs,
) -> None:
    """
    Plot centrality scores.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(plotting)s
    selected_score
        Whether to plot all scores or only just a selected one.

    Returns
    -------
    %(plotting_returns)s
    """
    # TODO: better logic - check first in obs and if it's categorical
    scores_key = f"{cluster_key}_centrality_scores"
    if scores_key not in adata.uns_keys():
        raise KeyError(
            f"`centrality_scores_key` {scores_key} not found. \n"
            "Choose a different key or run first as `squidpy.nhood.centrality_scores()`."
        )
    df = adata.uns[scores_key]

    var = DataFrame(df.columns, columns=[scores_key])
    var["index"] = var[scores_key]
    var = var.set_index("index")

    cat = adata.obs[cluster_key].cat.categories.values.astype(str)
    idx = {cluster_key: pd.Categorical(cat, categories=cat)}

    ad = AnnData(X=np.array(df), obs=idx, var=var)

    colors_key = f"{cluster_key}_colors"
    if colors_key in adata.uns.keys():
        ad.uns[colors_key] = adata.uns[colors_key]

    # without contrained_layout, the legend is clipped
    if selected_score is not None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)
        sc.pl.scatter(
            ad, x=selected_score, y=cluster_key, color=cluster_key, size=1000, title="", frameon=True, ax=ax, **kwargs
        )
    else:
        nrows = len(ad.var.index) - 1
        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=1,
            figsize=(4, 6 * nrows) if figsize is None else figsize,
            dpi=dpi,
            constrained_layout=True,
        )
        for i in range(nrows):
            x = list(ad.var.index)[i + 1]
            sc.pl.scatter(
                ad,
                x=str(x),
                y=cluster_key,
                color=cluster_key,
                size=1000,
                ax=ax[i],
                show=False,
                frameon=True,
                **kwargs,
            )

    fig.show()

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def interaction_matrix(
    adata: AnnData,
    cluster_key: str,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs,
) -> None:
    """
    Plot cluster interaction matrix.

    The interaction matrix is computed by :func:`squidpy.graph.interaction_matrix`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(plotting)s
    kwargs
        Keyword arguments for :func:`scanpy.pl.heatmap`.

    Returns
    -------
    %(plotting_returns)s
    """
    # TODO: better logic - check first in obs and if it's categorical
    int_key = f"{cluster_key}_interactions"
    if int_key not in adata.uns_keys():
        raise KeyError(
            f"cluster_interactions_key {int_key} not found. \n"
            "Choose a different key or run first `squidpy.graph.interaction_matrix()`."
        )
    array = adata.uns[int_key]

    cat = adata.obs[cluster_key].cat.categories.values.astype(str)
    idx = {cluster_key: pd.Categorical(cat, categories=cat)}

    ad = AnnData(X=array, obs=idx, var=idx)

    colors_key = f"{cluster_key}_colors"
    if colors_key in adata.uns.keys():
        ad.uns[colors_key] = adata.uns[colors_key]

    # TODO: handle dpi
    sc.pl.heatmap(ad, var_names=ad.var_names, groupby=cluster_key, figsize=figsize, save=save, **kwargs)


@d.dedent
def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    mode: str = "zscore",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs,
) -> None:
    """
    Plot neighborhood enrichement.

    The enrichment is computed by :func:`squidpy.graph.nhood_enrichment`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    mode
        TODO.
    %(plotting)s
    kwargs
        Keyword arguments for :func:`scanpy.pl.heatmap`.

    Returns
    -------
    %(plotting_returns)s
    """
    # TODO: better logic - check first in obs and if it's categorical
    int_key = f"{cluster_key}_nhood_enrichment"
    if int_key not in adata.uns_keys():
        raise ValueError(
            f"key {int_key} not found. \n" "Choose a different key or run first `squidpy.graph.nhood_enrichment()`."
        )
    array = adata.uns[int_key][mode]

    cat = adata.obs[cluster_key].cat.categories.values.astype(str)
    idx = {cluster_key: pd.Categorical(cat, categories=cat)}

    ad = AnnData(X=array, obs=idx, var=idx)

    colors_key = f"{cluster_key}_colors"
    if colors_key in adata.uns.keys():
        ad.uns[colors_key] = adata.uns[colors_key]

    sc.pl.heatmap(ad, var_names=ad.var_names, groupby=cluster_key, figsize=figsize, save=save, **kwargs)


@d.dedent
def plot_ripley_k(
    adata: AnnData,
    cluster_key: str,
    df: Optional[DataFrame] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs,
):
    """
    Plot Ripley K estimate for each cluster.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    df
        Data to plot. If `None`, try getting from ``adata.uns['ripley_k_{cluster_key}']``.
    %(plotting)s
    kwargs
        Keyword arguments to :func:`seaborn.lineplot`.

    Returns
    -------
    %(plotting_returns)s
    """
    # TODO: I really, really discourage this, should be refactored as which key to use
    if df is None:
        try:
            df = adata.uns[f"ripley_k_{cluster_key}"]
            hue_order = list(np.sort(adata.obs[cluster_key].unique()))

            try:
                palette = list(adata.uns[f"{cluster_key}_colors"])
            except KeyError:
                palette = None

        except KeyError:
            raise KeyError(
                f"\\looks like `riply_k_{cluster_key}` was not used\n\n"
                "\\is not present in adata.uns,\n"
                "\tplease rerun ripley_k or pass\n"
                "\ta dataframe"
            ) from None
    else:
        hue_order = palette = None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.lineplot(
        "distance",
        "ripley_k",
        hue="leiden",
        hue_order=hue_order,
        data=df,
        palette=palette,
        ax=ax,
        **kwargs,
    )

    if save is not None:
        save_fig(fig, path=save)
