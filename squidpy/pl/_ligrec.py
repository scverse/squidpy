from typing import Tuple, Union, Optional, Sequence
from pathlib import Path
from functools import partial

import scanpy as sc
from scanpy import logging as logg
from anndata import AnnData

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase

from squidpy._docs import d
from squidpy.pl._utils import save_fig, _get_black_or_white, _unique_order_preserving
from squidpy.gr._ligrec import LigrecResult

_SEP = " | "


class CustomDotplot(sc.pl.DotPlot):  # noqa: D101

    BASE = 10

    DEFAULT_LARGEST_DOT = 50.0
    DEFAULT_NUM_COLORBAR_TICKS = 5
    DEFAULT_NUM_LEGEND_DOTS = 5

    def __init__(self, minn: float, delta: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._delta = delta
        self._minn = minn

    def _plot_size_legend(self, size_legend_ax):
        y = self.BASE ** -((self.dot_max * self._delta) + self._minn)
        x = self.BASE ** -((self.dot_min * self._delta) + self._minn)
        size_range = -(np.logspace(x, y, self.DEFAULT_NUM_LEGEND_DOTS + 1, base=10).astype(np.float64))
        size_range = (size_range - np.min(size_range)) / (np.max(size_range) - np.min(size_range))
        # no point in showing dot of size 0
        size_range = size_range[1:]

        size = size_range ** self.size_exponent
        size = size * (self.largest_dot - self.smallest_dot) + self.smallest_dot

        # plot size bar
        size_legend_ax.scatter(
            np.arange(len(size)) + 0.5,
            np.repeat(1, len(size)),
            s=size,
            color="black",
            edgecolor="black",
            linewidth=self.dot_edge_lw,
            zorder=100,
        )
        size_legend_ax.set_xticks(np.arange(len(size)) + 0.5)
        labels = [f"{(x * self._delta) + self._minn:.1f}" for x in size_range]
        size_legend_ax.set_xticklabels(labels, fontsize="small")

        # remove y ticks and labels
        size_legend_ax.tick_params(axis="y", left=False, labelleft=False, labelright=False)
        # remove surrounding lines
        for direction in ["right", "top", "left", "bottom"]:
            size_legend_ax.spines[direction].set_visible(False)

        ymax = size_legend_ax.get_ylim()[1]
        size_legend_ax.set_ylim(-1.05 - self.largest_dot * 0.003, 4)
        size_legend_ax.set_title(self.size_title, y=ymax + 0.25, size="small")

        xmin, xmax = size_legend_ax.get_xlim()
        size_legend_ax.set_xlim(xmin - 0.15, xmax + 0.5)

    def _plot_colorbar(self, color_legend_ax, normalize):
        cmap = plt.get_cmap(self.cmap)

        ColorbarBase(
            color_legend_ax,
            orientation="horizontal",
            cmap=cmap,
            norm=normalize,
            ticks=np.linspace(
                np.nanmin(self.dot_color_df.values),
                np.nanmax(self.dot_color_df.values),
                self.DEFAULT_NUM_COLORBAR_TICKS,
            ),
            format="%.2f",
        )

        color_legend_ax.set_title(self.color_legend_title, fontsize="small")
        color_legend_ax.xaxis.set_tick_params(labelsize="small")


@d.dedent
def ligrec(
    adata: Union[AnnData, Tuple[pd.DataFrame, pd.DataFrame]],
    key: Optional[str] = None,
    src_clusters: Optional[Union[str, Sequence[str]]] = None,
    tgt_clusters: Optional[Union[str, Sequence[str]]] = None,
    remove_empty_interactions: bool = True,
    dendrogram: bool = False,
    alpha: Optional[float] = 0.001,
    swap_axes: bool = False,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs,
) -> None:
    """
    Plot results of receptor-ligand permutation test.

    Parameters
    ----------
    %(adata)s
        It can also be a :class:`namedtuple` as returned by :func:`squidpy.gr.ligrec`.
    key
        Key in :attr:`anndata.AnnData.uns`. Only used when ``adata`` is of type :class:`AnnData`.
    src_clusters
        Source interaction clusters. If `None`, select all clusters.
    tgt_clusters
        Target interaction clusters. If `None`, select all clusters.
    remove_empty_interactions
        Whether to remove interactions which have `NaN` values in all cluster combinations.
    dendrogram
        Whether to show dendrogram.
    swap_axes
        Whether to show the cluster combinations as rows and the interacting pairs as columns.
    alpha
        Significance threshold. All elements with p-values less or equal to ``alpha`` will be marked by tori
        instead of dots.
    %(plotting)s
    kwargs
        Keyword arguments for :meth:`scanpy.pl.DotPlot.style`.

    Returns
    -------
    %(plotting_returns)s
    """
    if isinstance(adata, AnnData):
        if key not in adata.uns_keys():
            raise KeyError(f"Key `{key}` not found in `adata.uns`.")
        adata = adata.uns[key]

    if not isinstance(adata, LigrecResult):
        raise TypeError(
            f"Expected `adata` to be either of type `anndata.AnnData` or `LigrecResult`, "
            f"found `{type(adata).__name__}`."
        )

    if alpha is not None and not (0 <= alpha <= 1):
        raise ValueError(f"Expected `alpha` to be in range `[0, 1]`, found `{alpha}`.")

    if src_clusters is None:
        src_clusters = adata.pvalues.columns.get_level_values(0)
    elif isinstance(src_clusters, str):
        src_clusters = [src_clusters]

    if tgt_clusters is None:
        tgt_clusters = adata.pvalues.columns.get_level_values(1)
    if isinstance(tgt_clusters, str):
        tgt_clusters = [tgt_clusters]

    src_clusters = _unique_order_preserving(src_clusters)
    tgt_clusters = _unique_order_preserving(tgt_clusters)

    pvals = adata.pvalues.loc[:, (src_clusters, tgt_clusters)]
    means = adata.means.loc[:, (src_clusters, tgt_clusters)]

    if pvals.empty:
        raise ValueError("No clusters have been selected.")

    if remove_empty_interactions:
        mask = ~adata.pvalues.isnull().all(axis=1)
        pvals = pvals.loc[mask]
        means = means.loc[mask]

        if pvals.empty:
            raise ValueError("After removing empty interactions, none remain.")

    start, label_ranges = 0, {}
    for cls, size in (pvals.groupby(level=0, axis=1)).size().to_dict().items():
        label_ranges[cls] = (start, start + size - 1)
        start += size
    label_ranges = {k: label_ranges[k] for k in sorted(label_ranges.keys())}

    pvals = -np.log10(pvals).fillna(0)
    pvals = pvals[label_ranges.keys()]
    pvals.columns = map(_SEP.join, pvals.columns.to_flat_index())
    pvals.index = map(_SEP.join, pvals.index.to_flat_index())

    means = means[label_ranges.keys()]
    means.columns = map(_SEP.join, means.columns.to_flat_index())
    means.index = map(_SEP.join, means.index.to_flat_index())
    means = np.log2(means + 1)

    var = pd.DataFrame(pvals.columns)
    var.set_index((var.columns[0]), inplace=True)

    adata = AnnData(pvals.values, obs={"groups": pd.Categorical(pvals.index, pvals.index)}, var=var)
    minn = np.nanmin(adata.X)
    delta = np.nanmax(adata.X) - minn
    adata.X = (adata.X - minn) / delta

    if dendrogram:
        sc.pp.pca(adata)
        sc.tl.dendrogram(adata, groupby="groups", key_added="dendrogram")

    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("grid", True)
    cmap = plt.get_cmap(kwargs["cmap"])

    dp = (
        CustomDotplot(
            delta=delta,
            minn=minn,
            adata=adata,
            var_names=adata.var_names,
            groupby="groups",
            dot_color_df=means,
            dot_size_df=pvals,
            title="Receptor-ligand test",
            var_group_labels=tuple(label_ranges.keys()),
            var_group_positions=tuple(label_ranges.values()),
            standard_scale=None,
            figsize=figsize,
        )
        .style(
            **kwargs,
        )
        .legend(size_title=r"$-\log_{10} ~ P$", colorbar_title=r"$log_{2}(\frac{molecule1 + molecule2}{2} + 1)$")
    )
    if dendrogram:
        dp.add_dendrogram(size=1.6)
    if swap_axes:
        dp.swap_axes()
    dp.make_figure()

    labs = dp.ax_dict["mainplot_ax"].get_yticklabels() if swap_axes else dp.ax_dict["mainplot_ax"].get_xticklabels()
    for text in labs:
        text.set_text(text.get_text().split(_SEP)[1])
    if swap_axes:
        dp.ax_dict["mainplot_ax"].set_yticklabels(labs)
    else:
        dp.ax_dict["mainplot_ax"].set_xticklabels(labs)

    if alpha is not None:
        mapper = np.argsort(adata.uns["dendrogram"]["categories_idx_ordered"]) if dendrogram else np.arange(len(pvals))
        mean_min, mean_max = np.nanmin(means), np.nanmax(means)
        mean_delta = mean_max - mean_min

        yy, xx = np.where(pvals.values >= -np.log10(alpha))
        if len(xx) and len(yy):
            logg.info(f"Found `{len(yy)}` significant interactions at level `{alpha}`")
            ss = 0.33 * (adata.X[yy, xx] * (dp.largest_dot - dp.smallest_dot) + dp.smallest_dot)
            # just a precaution to
            cc = np.vectorize(partial(_get_black_or_white, cmap=cmap))((means.values[yy, xx] - mean_min) / mean_delta)

            # must be after ss = ..., cc = ...
            yy = np.array([mapper[y] for y in yy])
            if swap_axes:
                xx, yy = yy, xx
            dp.ax_dict["mainplot_ax"].scatter(xx + 0.5, yy + 0.5, color=cc, s=ss, lw=0)

    if dpi is not None:
        dp.fig.set_dpi(dpi)

    if save is not None:
        save_fig(dp.fig, save)
