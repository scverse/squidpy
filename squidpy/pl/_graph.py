"""Plotting for gr functions."""

from types import MappingProxyType
from typing import Any, Tuple, Union, Mapping, Optional, Sequence, TYPE_CHECKING
from pathlib import Path

from anndata import AnnData
import scanpy as sc

from pandas import DataFrame
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from squidpy._docs import d
from squidpy.pl._utils import save_fig


@d.dedent
def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    selected_score: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs: Any,
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
    dpi: Optional[int] = None,  # FIXME
    save: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> None:
    """
    Plot cluster interaction matrix.

    The interaction matrix is computed by :func:`squidpy.gr.interaction_matrix`.

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
            "Choose a different key or run first `squidpy.gr.interaction_matrix()`."
        )
    array = adata.uns[int_key]

    cat = adata.obs[cluster_key].cat.categories.values.astype(str)
    idx = {cluster_key: pd.Categorical(cat, categories=cat)}

    ad = AnnData(
        X=array,
        obs=idx,
    )
    ad.var_names = idx[cluster_key]

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
    dpi: Optional[int] = None,  # FIXME
    save: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> None:
    """
    Plot neighborhood enrichement.

    The enrichment is computed by :func:`squidpy.gr.nhood_enrichment`.

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
            f"key {int_key} not found. \n" "Choose a different key or run first `squidpy.gr.nhood_enrichment()`."
        )
    array = adata.uns[int_key][mode]

    cat = adata.obs[cluster_key].cat.categories.values.astype(str)
    idx = {cluster_key: pd.Categorical(cat, categories=cat)}

    ad = AnnData(
        X=array,
        obs=idx,
    )
    ad.var_names = idx[cluster_key]

    colors_key = f"{cluster_key}_colors"
    if colors_key in adata.uns.keys():
        ad.uns[colors_key] = adata.uns[colors_key]

    # TODO: handle dpi
    sc.pl.heatmap(ad, var_names=ad.var_names, groupby=cluster_key, figsize=figsize, save=save, **kwargs)


@d.dedent
def ripley_k(
    adata: AnnData,
    cluster_key: str,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> None:
    """
    Plot Ripley K estimate for each cluster.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(plotting)s
    legend_kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    kwargs
        Keyword arguments to :func:`seaborn.lineplot`.

    Returns
    -------
    %(plotting_returns)s
    """
    try:
        df = adata.uns[f"ripley_k_{cluster_key}"]
    except KeyError:
        raise KeyError(f"Please run `squidpy.gr.ripley_k(..., cluster_key={cluster_key!r})`.") from None

    legend_kwargs = dict(legend_kwargs)
    if "loc" not in legend_kwargs:
        legend_kwargs["loc"] = "center left"
        legend_kwargs.setdefault("bbox_to_anchor", (1, 0.5))

    categories = adata.obs[cluster_key].cat.categories
    palette = adata.uns.get(f"{cluster_key}_colors", None)
    if palette is not None:
        palette = {k: v for k, v in zip(categories, palette)}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sns.lineplot(
        x="distance",
        y="ripley_k",
        hue=cluster_key,
        hue_order=categories,
        data=df,
        palette=palette,
        ax=ax,
        **kwargs,
    )
    ax.legend(**legend_kwargs)

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def co_occurrence(
    adata: AnnData,
    cluster_key: str,
    group: Optional[Union[str, Sequence[str]]] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> None:
    """
    Plot co-occurrence probability ratio for each cluster.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    group
        Cluster instance to plot conditional probability.
    %(plotting)s
    legend_kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    kwargs
        Keyword arguments to :func:`seaborn.lineplot`.

    Returns
    -------
    %(plotting_returns)s
    """
    try:
        occurrence_data = adata.uns[f"{cluster_key}_co_occurrence"]
    except KeyError:
        raise KeyError(f"Please run `squidpy.gr.co_occurence(..., cluster_key={cluster_key!r})`.") from None

    legend_kwargs = dict(legend_kwargs)
    if "loc" not in legend_kwargs:
        legend_kwargs["loc"] = "center left"
        legend_kwargs.setdefault("bbox_to_anchor", (1, 0.5))

    if isinstance(group, str):
        group = (group,)

    out = occurrence_data["occ"]
    interval = occurrence_data["interval"][1:]
    categories = adata.obs[cluster_key].cat.categories
    if group is None:
        group = categories
    group = np.array(group)
    if TYPE_CHECKING:
        assert isinstance(group, Sequence)

    group = sorted(group[np.isin(group, categories)])
    if not len(group):
        raise ValueError("No valid groups have been found.")

    palette = adata.uns.get(f"{cluster_key}_colors", None)
    if palette is not None:
        palette = {k: v for k, v in zip(categories, palette)}

    fig, axs = plt.subplots(1, len(group), figsize=figsize, dpi=dpi, constrained_layout=True)
    axs = np.ravel(axs)  # make into iterable

    for g, ax in zip(group, axs):
        idx = np.where(categories == g)[0][0]
        df = pd.DataFrame(out[idx, :, :].T, columns=categories).melt(var_name=cluster_key, value_name="probability")
        df["distance"] = np.tile(interval, len(categories))

        sns.lineplot(
            x="distance",
            y="probability",
            data=df,
            dashes=False,
            hue=cluster_key,
            hue_order=categories,
            palette=palette,
            ax=ax,
            **kwargs,
        )
        ax.legend(**legend_kwargs)
        ax.set_ylabel(rf"$\frac{{p(exp|{g})}}{{p(exp)}}$")

    if save is not None:
        save_fig(fig, path=save)
