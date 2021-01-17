"""Plotting for gr functions."""

from types import MappingProxyType
from typing import Any, Tuple, Union, Mapping, Optional, Sequence

from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation

try:
    from typing import Literal  # type: ignore[attr-defined]
except ImportError:
    from typing_extensions import Literal

from pathlib import Path

from anndata import AnnData

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from squidpy._docs import d
from squidpy.gr._utils import (
    _get_valid_values,
    _assert_categorical_obs,
    _assert_non_empty_sequence,
)
from squidpy.pl._utils import _heatmap, save_fig
from squidpy.constants._pkg_constants import Key

Palette_t = Optional[Union[str, mcolors.ListedColormap]]


def _maybe_set_colors(source: AnnData, target: AnnData, key: str, palette: Optional[str] = None) -> None:
    color_key = Key.uns.colors(key)
    try:
        if palette is not None:
            raise KeyError
        target.uns[color_key] = source.uns[color_key]
    except KeyError:
        add_colors_for_categorical_sample_annotation(target, key=key, force_update_colors=True, palette=palette)


def _get_data(adata: AnnData, cluster_key: str, func_name: str) -> Any:
    key = getattr(Key.uns, func_name)(cluster_key)
    try:
        return adata.uns[key]
    except KeyError:
        raise KeyError(
            f"Unable to get the data from `adata.uns[{key!r}]`. "
            f"Please run `squidpy.gr.{func_name}(..., cluster_key={cluster_key!r})` first."
        ) from None


def _get_palette(adata: AnnData, cluster_key: str, categories: Sequence[Any]) -> Optional[Mapping[str, Any]]:
    try:
        palette = adata.uns[Key.uns.colors(cluster_key)]
        if len(palette) < len(categories):
            raise ValueError(
                f"Expected to find at least `{len(categories)}` colors, "
                f"found `{len(palette)}` for key `{cluster_key}`."
            )
        return dict(zip(categories, palette))
    except KeyError:
        return None


@d.dedent
def centrality_scores(
    adata: AnnData,
    cluster_key: str,
    score: Optional[Union[str, Sequence[str]]] = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    palette: Palette_t = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
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
    palette = _get_palette(adata, cluster_key=cluster_key, categories=clusters) if palette is None else palette

    score = scores if score is None else score
    score = _assert_non_empty_sequence(score)  # type: ignore[assignment]
    score = sorted(_get_valid_values(score, scores))

    palette = adata.uns.get(f"{cluster_key}_colors", None)
    if palette is not None:
        palette = {k: v for k, v in zip(clusters, palette)}
    print(score)
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
    palette: Palette_t = None,
    annotate: bool = False,
    cmap: str = "viridis",
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
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
    %(heatmap_plotting)s
    kwargs
        Keyword arguments for :func:`matplotlib.pyplot.text`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    array = _get_data(adata, cluster_key=cluster_key, func_name="interaction_matrix")

    ad = AnnData(X=array, obs={cluster_key: pd.Categorical(adata.obs[cluster_key].cat.categories)})
    _maybe_set_colors(source=adata, target=ad, key=cluster_key, palette=palette)

    fig = _heatmap(
        ad,
        title="Interaction matrix",
        cont_cmap=cmap,
        annotate=annotate,
        figsize=(2 * ad.n_obs // 3, 2 * ad.n_obs // 3) if figsize is None else figsize,
        dpi=dpi,
        **kwargs,
    )

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def nhood_enrichment(
    adata: AnnData,
    cluster_key: str,
    mode: Literal["zscore", "count"] = "zscore",  # type: ignore[name-defined]
    annotate: bool = False,
    cmap: str = "viridis",
    palette: Palette_t = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
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
        Which :func:`squidpy.gr.nhood_enrichment` result to plot. \
            Valid options are:

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

    ad = AnnData(X=array, obs={cluster_key: pd.Categorical(adata.obs[cluster_key].cat.categories)})
    _maybe_set_colors(source=adata, target=ad, key=cluster_key, palette=palette)

    fig = _heatmap(
        ad,
        title="Neighborhood enrichment",
        cont_cmap=cmap,
        annotate=annotate,
        figsize=(2 * ad.n_obs // 3, 2 * ad.n_obs // 3) if figsize is None else figsize,
        dpi=dpi,
        **kwargs,
    )

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def ripley_k(
    adata: AnnData,
    cluster_key: str,
    palette: Palette_t = None,
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
    %(cat_plotting)s
    legend_kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    kwargs
        Keyword arguments to :func:`seaborn.lineplot`.

    Returns
    -------
    %(plotting_returns)s
    """
    _assert_categorical_obs(adata, key=cluster_key)
    df = _get_data(adata, cluster_key=cluster_key, func_name="ripley_k")

    legend_kwargs = dict(legend_kwargs)
    if "loc" not in legend_kwargs:
        legend_kwargs["loc"] = "center left"
        legend_kwargs.setdefault("bbox_to_anchor", (1, 0.5))

    categories = adata.obs[cluster_key].cat.categories
    palette = _get_palette(adata, cluster_key=cluster_key, categories=categories) if palette is None else palette

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
    ax.set_ylabel("value")
    ax.set_title("Ripley's K")

    if save is not None:
        save_fig(fig, path=save)


@d.dedent
def co_occurrence(
    adata: AnnData,
    cluster_key: str,
    palette: Palette_t = None,
    clusters: Optional[Union[str, Sequence[str]]] = None,
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
    clusters
        Cluster instance to plot conditional probability.
    %(cat_plotting)s
    legend_kwargs
        Keyword arguments for :func:`matplotlib.pyplot.legend`.
    kwargs
        Keyword arguments to :func:`seaborn.lineplot`.

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
    clusters = _assert_non_empty_sequence(clusters)  # type: ignore[assignment]
    clusters = sorted(_get_valid_values(clusters, categories))

    palette = _get_palette(adata, cluster_key=cluster_key, categories=categories) if palette is None else palette

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
