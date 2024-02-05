"""Plotting for graph functions."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Mapping,
    Sequence,
    Union,  # noqa: F401
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes

from squidpy._constants._constants import RipleyStat
from squidpy._constants._pkg_constants import Key
from squidpy._docs import d
from squidpy.gr._utils import (
    _assert_categorical_obs,
    _assert_non_empty_sequence,
    _get_valid_values,
)
from squidpy.pl._color_utils import Palette_t, _get_palette, _maybe_set_colors
from squidpy.pl._utils import _heatmap, save_fig

__all__ = ["centrality_scores", "interaction_matrix", "nhood_enrichment", "ripley", "co_occurrence"]


def _get_data(adata: AnnData, cluster_key: str, func_name: str, attr: str = "uns", **kwargs: Any) -> Any:
    key = getattr(Key.uns, func_name)(cluster_key, **kwargs)

    try:
        if attr == "uns":
            return adata.uns[key]
        elif attr == "obsm":
            return adata.obsm[key]
        else:
            raise ValueError(f"attr must be either 'uns' or 'obsm', got {attr}")
    except KeyError:
        raise KeyError(
            f"Unable to get the data from `adata.uns[{key!r}]`. "
            f"Please run `squidpy.gr.{func_name}(..., cluster_key={cluster_key!r})` first."
        ) from None


@d.dedent
def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    score: str | Sequence[str] | None = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    palette: Palette_t = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot centrality scores.

    The centrality scores are computed by :func:`squidpy.gr.centrality_scores`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    score
        Whether to plot all scores or only selected ones.
    legend_kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    %(cat_plotting)s

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    df = _get_data(adata, cluster_key=cluster_key, func_name="centrality_scores").copy()

    legend_kwargs = dict(legend_kwargs)
    if "loc" not in legend_kwargs:
        legend_kwargs["loc"] = "center left"
        legend_kwargs.setdefault("bbox_to_anchor", (1, 0.5))

    scores = df.columns.values
    df[cluster_key] = df.index.values

    clusters = adata.obs[cluster_key].cat.categories
    palette = _get_palette(adata, cluster_key=cluster_key, categories=clusters, palette=palette)

    score = scores if score is None else score
    score = _assert_non_empty_sequence(score, name="centrality scores")
    score = sorted(_get_valid_values(score, scores))

    fig, axs = plt.subplots(1, len(score), figsize=figsize, dpi=dpi, constrained_layout=True)
    axs = np.ravel(axs)  # make into iterable
    for g, ax in zip(score, axs):
        sns.scatterplot(
            x=g,
            y=cluster_key,
            data=df,
            hue=cluster_key,
            hue_order=clusters,
            palette=palette,
            ax=ax,
            **kwargs,
        )
        ax.set_title(str(g).replace("_", " ").capitalize())
        ax.set_xlabel("value")

        ax.set_yticks([])
        ax.legend(**legend_kwargs)

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def interaction_matrix(
    adata: AnnData,
    cluster_key: str,
    annotate: bool = False,
    method: str | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    palette: Palette_t = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot cluster interaction matrix.

    The interaction matrix is computed by :func:`squidpy.gr.interaction_matrix`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    %(heatmap_plotting)s
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.text`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    array = _get_data(adata, cluster_key=cluster_key, func_name="interaction_matrix")

    ad = AnnData(X=array, obs={cluster_key: pd.Categorical(adata.obs[cluster_key].cat.categories)}, dtype=array.dtype)
    _maybe_set_colors(source=adata, target=ad, key=cluster_key, palette=palette)
    if title is None:
        title = "Interaction matrix"
    fig = _heatmap(
        ad,
        key=cluster_key,
        title=title,
        method=method,
        cont_cmap=cmap,
        annotate=annotate,
        figsize=(2 * ad.n_obs // 3, 2 * ad.n_obs // 3) if figsize is None else figsize,
        dpi=dpi,
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        **kwargs,
    )

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    mode: Literal["zscore", "count"] = "zscore",
    annotate: bool = False,
    method: str | None = None,
    title: str | None = None,
    cmap: str = "viridis",
    palette: Palette_t = None,
    cbar_kwargs: Mapping[str, Any] = MappingProxyType({}),
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> None:
    """
    Plot neighborhood enrichment.

    The enrichment is computed by :func:`squidpy.gr.nhood_enrichment`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    mode
        Which :func:`squidpy.gr.nhood_enrichment` result to plot. Valid options are:

            - `'zscore'` - z-score values of enrichment statistic.
            - `'count'` - enrichment count.

    %(heatmap_plotting)s
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.text`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    array = _get_data(adata, cluster_key=cluster_key, func_name="nhood_enrichment")[mode]

    ad = AnnData(X=array, obs={cluster_key: pd.Categorical(adata.obs[cluster_key].cat.categories)}, dtype=array.dtype)
    _maybe_set_colors(source=adata, target=ad, key=cluster_key, palette=palette)
    if title is None:
        title = "Neighborhood enrichment"
    fig = _heatmap(
        ad,
        key=cluster_key,
        title=title,
        method=method,
        cont_cmap=cmap,
        annotate=annotate,
        figsize=(2 * ad.n_obs // 3, 2 * ad.n_obs // 3) if figsize is None else figsize,
        dpi=dpi,
        cbar_kwargs=cbar_kwargs,
        ax=ax,
        **kwargs,
    )

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def ripley(
    adata: AnnData,
    cluster_key: str,
    mode: Literal["F", "G", "L"] = "F",
    plot_sims: bool = True,
    palette: Palette_t = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    ax: Axes | None = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> None:
    """
    Plot Ripley's statistics for each cluster.

    The estimate is computed by :func:`squidpy.gr.ripley`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    mode
        Ripley's statistics to be plotted.
    plot_sims
        Whether to overlay simulations in the plot.
    %(cat_plotting)s
    ax
        Axes, :class:`matplotlib.axes.Axes`.
    legend_kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    kwargs
        Keyword arguments for :func:`seaborn.lineplot`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)

    res = _get_data(adata, cluster_key=cluster_key, func_name="ripley", mode=mode)

    mode = RipleyStat(mode)  # type: ignore[assignment]
    if TYPE_CHECKING:
        assert isinstance(mode, RipleyStat)

    legend_kwargs = dict(legend_kwargs)
    if "loc" not in legend_kwargs:
        legend_kwargs["loc"] = "center left"
        legend_kwargs.setdefault("bbox_to_anchor", (1, 0.5))

    categories = adata.obs[cluster_key].cat.categories
    palette = _get_palette(adata, cluster_key=cluster_key, categories=categories, palette=palette)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    sns.lineplot(
        y="stats",
        x="bins",
        hue=cluster_key,
        data=res[f"{mode.s}_stat"],
        hue_order=categories,
        palette=palette,
        ax=ax,
        **kwargs,
    )
    if plot_sims:
        sns.lineplot(y="stats", x="bins", ci="sd", alpha=0.01, color="gray", data=res["sims_stat"], ax=ax)
    ax.legend(**legend_kwargs)
    ax.set_ylabel("value")
    ax.set_title(f"Ripley's {mode.s}")

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def co_occurrence(
    adata: AnnData,
    cluster_key: str,
    palette: Palette_t = None,
    clusters: str | Sequence[str] | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> None:
    """
    Plot co-occurrence probability ratio for each cluster.

    The co-occurrence is computed by :func:`squidpy.gr.co_occurrence`.

    Parameters
    ----------
    %(adata)s
    %(cluster_key)s
    clusters
        Cluster instances for which to plot conditional probability.
    %(cat_plotting)s
    legend_kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    kwargs
        Keyword arguments for :func:`seaborn.lineplot`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    occurrence_data = _get_data(adata, cluster_key=cluster_key, func_name="co_occurrence")

    legend_kwargs = dict(legend_kwargs)
    if "loc" not in legend_kwargs:
        legend_kwargs["loc"] = "center left"
        legend_kwargs.setdefault("bbox_to_anchor", (1, 0.5))

    out = occurrence_data["occ"]
    interval = occurrence_data["interval"][1:]
    categories = adata.obs[cluster_key].cat.categories

    clusters = categories if clusters is None else clusters
    clusters = _assert_non_empty_sequence(clusters, name="clusters")
    clusters = sorted(_get_valid_values(clusters, categories))

    palette = _get_palette(adata, cluster_key=cluster_key, categories=categories, palette=palette)

    fig, axs = plt.subplots(
        1,
        len(clusters),
        figsize=(5 * len(clusters), 5) if figsize is None else figsize,
        dpi=dpi,
        constrained_layout=True,
    )
    axs = np.ravel(axs)  # make into iterable

    for g, ax in zip(clusters, axs):
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
        ax.set_title(rf"$\frac{{p(exp|{g})}}{{p(exp)}}$")
        ax.set_ylabel("value")

    if save is not None:
        save_fig(fig, path=save)
