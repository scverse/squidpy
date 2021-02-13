from typing import Any, Tuple, Union, Mapping, Optional, Sequence, TYPE_CHECKING
from pathlib import Path

from scanpy import logging as logg
from anndata import AnnData
import scanpy as sc

from scipy.cluster import hierarchy as sch
import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
import matplotlib.pyplot as plt

from squidpy._docs import d
from squidpy._utils import verbosity, _unique_order_preserving
from squidpy.pl._utils import save_fig, _dendrogram, _filter_kwargs
from squidpy._constants._constants import DendrogramAxis
from squidpy._constants._pkg_constants import Key

__all__ = ["ligrec"]

_SEP = " | "  # cluster separator


class CustomDotplot(sc.pl.DotPlot):

    BASE = 10

    DEFAULT_LARGEST_DOT = 50.0
    DEFAULT_NUM_COLORBAR_TICKS = 5
    DEFAULT_NUM_LEGEND_DOTS = 5

    def __init__(self, minn: float, delta: float, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._delta = delta
        self._minn = minn

    def _plot_size_legend(self, size_legend_ax: Axes) -> None:
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

    def _plot_colorbar(self, color_legend_ax: Axes, normalize: bool) -> None:
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
    adata: Union[AnnData, Mapping[str, pd.DataFrame]],
    cluster_key: Optional[str] = None,
    source_groups: Optional[Union[str, Sequence[str]]] = None,
    target_groups: Optional[Union[str, Sequence[str]]] = None,
    means_range: Tuple[float, float] = (-np.inf, np.inf),
    pvalue_threshold: float = 1.0,
    remove_empty_interactions: bool = True,
    dendrogram: Optional[str] = None,
    alpha: Optional[float] = 0.001,
    swap_axes: bool = False,
    title: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: Optional[int] = None,
    save: Optional[Union[str, Path]] = None,
    **kwargs: Any,
) -> None:
    """
    Plot the result of a receptor-ligand permutation test.

    The result was computed by :func:`squidpy.gr.ligrec`.

    :math:`molecule_1` belongs to the source clusters displayed on the top (or on the right, if ``swap_axes = True``,
    whereas :math:`molecule_2` belongs to the target clusters.

    Parameters
    ----------
    %(adata)s
        It can also be a :class:`dict`, as returned by :func:`squidpy.gr.ligrec`.
    %(cluster_key)s
        Only used when ``adata`` is of type :class:`AnnData`.
    source_groups
        Source interaction clusters. If `None`, select all clusters.
    target_groups
        Target interaction clusters. If `None`, select all clusters.
    means_range
        Only show interactions whose means are within this **closed** interval.
    pvalue_threshold
        Only show interactions with p-value <= ``pvalue_threshold``.
    dendrogram
        How to cluster based on the p-values. Valid options are:

            -  `None` - do not perform clustering.
            - `'interacting_molecules'` - cluster the interacting molecules.
            - `'interacting_clusters'` - cluster the interacting clusters.
            - `'both'` - cluster both rows and columns. Note that in this case, the dendrogram is not shown.

    swap_axes
        Whether to show the cluster combinations as rows and the interacting pairs as columns.
    title
        Title of the plot.
    alpha
        Significance threshold. All elements with p-values <= ``alpha`` will be marked by tori instead of dots.
    %(plotting)s
    kwargs
        Keyword arguments for :meth:`scanpy.pl.DotPlot.style` or :meth:`scanpy.pl.DotPlot.legend`.

    Returns
    -------
    %(plotting_returns)s
    """

    def get_dendrogram(adata: AnnData, linkage: str = "complete") -> Mapping[str, Any]:
        z_var = sch.linkage(
            adata.X,
            metric="correlation",
            method=linkage,
            optimal_ordering=adata.n_obs <= 1500,  # matplotlib will most likely give up first
        )
        dendro_info = sch.dendrogram(z_var, labels=adata.obs_names.values, no_plot=True)
        # this is what the DotPlot requires
        return {
            "linkage": z_var,
            "groupby": ["groups"],
            "cor_method": "pearson",
            "use_rep": None,
            "linkage_method": linkage,
            "categories_ordered": dendro_info["ivl"],
            "categories_idx_ordered": dendro_info["leaves"],
            "dendrogram_info": dendro_info,
        }

    if dendrogram is not None:
        dendrogram = DendrogramAxis(dendrogram)  # type: ignore[assignment]
        if TYPE_CHECKING:
            assert isinstance(dendrogram, DendrogramAxis)

    if isinstance(adata, AnnData):
        if cluster_key is None:
            raise ValueError("Please provide `cluster_key` when supplying an `AnnData` object.")

        cluster_key = Key.uns.ligrec(cluster_key)
        if cluster_key not in adata.uns_keys():
            raise KeyError(f"Key `{cluster_key}` not found in `adata.uns`.")
        adata = adata.uns[cluster_key]

    if not isinstance(adata, dict):
        raise TypeError(
            f"Expected `adata` to be either of type `anndata.AnnData` or `dict`, " f"found `{type(adata).__name__}`."
        )
    if len(means_range) != 2:
        raise ValueError(f"Expected `means_range` to be a sequence of size `2`, found `{len(means_range)}`.")
    means_range = tuple(sorted(means_range))  # type: ignore[assignment]

    if alpha is not None and not (0 <= alpha <= 1):
        raise ValueError(f"Expected `alpha` to be in range `[0, 1]`, found `{alpha}`.")

    if source_groups is None:
        source_groups = adata["pvalues"].columns.get_level_values(0)
    elif isinstance(source_groups, str):
        source_groups = (source_groups,)

    if target_groups is None:
        target_groups = adata["pvalues"].columns.get_level_values(1)
    if isinstance(target_groups, str):
        target_groups = (target_groups,)
    if title is None:
        title = "Receptor-ligand test"

    source_groups, _ = _unique_order_preserving(source_groups)  # type: ignore[no-redef,assignment]
    target_groups, _ = _unique_order_preserving(target_groups)  # type: ignore[no-redef,assignment]

    pvals: pd.DataFrame = adata["pvalues"].loc[:, (source_groups, target_groups)]
    means: pd.DataFrame = adata["means"].loc[:, (source_groups, target_groups)]

    if pvals.empty:
        raise ValueError("No valid clusters have been selected.")

    means = means[(means >= means_range[0]) & (means <= means_range[1])]
    pvals = pvals[pvals <= pvalue_threshold]

    if remove_empty_interactions:
        mask = ~(pd.isnull(means) | pd.isnull(pvals))
        mask_rows = mask.any(axis=1)
        pvals = pvals.loc[mask_rows]
        means = means.loc[mask_rows]

        if pvals.empty:
            raise ValueError("After removing rows with only NaN interactions, none remain.")

        mask_cols = mask.any(axis=0)
        pvals = pvals.loc[:, mask_cols]
        means = means.loc[:, mask_cols]

        if pvals.empty:
            raise ValueError("After removing columns with only NaN interactions, none remain.")

    start, label_ranges = 0, {}

    if dendrogram == DendrogramAxis.INTERACTING_CLUSTERS:
        # rows are now cluster combinations, not interacting pairs
        pvals = pvals.T
        means = means.T

    for cls, size in (pvals.groupby(level=0, axis=1)).size().to_dict().items():
        label_ranges[cls] = (start, start + size - 1)
        start += size
    label_ranges = {k: label_ranges[k] for k in sorted(label_ranges.keys())}

    pvals = -np.log10(pvals).fillna(0)
    pvals = pvals[label_ranges.keys()]
    pvals.columns = map(_SEP.join, pvals.columns.to_flat_index())
    pvals.index = map(_SEP.join, pvals.index.to_flat_index())

    means = means[label_ranges.keys()].fillna(0)
    means.columns = map(_SEP.join, means.columns.to_flat_index())
    means.index = map(_SEP.join, means.index.to_flat_index())
    means = np.log2(means + 1)

    var = pd.DataFrame(pvals.columns)
    var = var.set_index(var.columns[0])

    adata = AnnData(pvals.values, obs={"groups": pd.Categorical(pvals.index)}, var=var)
    adata.obs_names = pvals.index
    minn = np.nanmin(adata.X)
    delta = np.nanmax(adata.X) - minn
    adata.X = (adata.X - minn) / delta

    try:
        if dendrogram == DendrogramAxis.BOTH:
            row_order, col_order, _, _ = _dendrogram(
                adata.X, method="complete", metric="correlation", optimal_ordering=adata.n_obs <= 1500
            )
            adata = adata[row_order, :][:, col_order]
            pvals = pvals.iloc[row_order, :].iloc[:, col_order]
            means = means.iloc[row_order, :].iloc[:, col_order]
        elif dendrogram is not None:
            adata.uns["dendrogram"] = get_dendrogram(adata)
    except IndexError:
        # just in case pandas indexing fails
        raise
    except Exception as e:
        logg.warning(f"Unable to create a dendrogram. Reason: `{e}`")
        dendrogram = None

    kwargs["dot_edge_lw"] = 0
    kwargs.setdefault("cmap", "viridis")
    kwargs.setdefault("grid", True)
    kwargs.pop("color_on", None)  # interferes with tori

    dp = (
        CustomDotplot(
            delta=delta,
            minn=minn,
            adata=adata,
            var_names=adata.var_names,
            groupby="groups",
            dot_color_df=means,
            dot_size_df=pvals,
            title=title,
            var_group_labels=None if dendrogram == DendrogramAxis.BOTH else list(label_ranges.keys()),
            var_group_positions=None if dendrogram == DendrogramAxis.BOTH else list(label_ranges.values()),
            standard_scale=None,
            figsize=figsize,
        )
        .style(
            **_filter_kwargs(sc.pl.DotPlot.style, kwargs),
        )
        .legend(
            size_title=r"$-\log_{10} ~ P$",
            colorbar_title=r"$log_2(\frac{molecule_1 + molecule_2}{2} + 1)$",
            **_filter_kwargs(sc.pl.DotPlot.legend, kwargs),
        )
    )
    if dendrogram in (DendrogramAxis.INTERACTING_MOLS, DendrogramAxis.INTERACTING_CLUSTERS):
        # ignore the warning about mismatching groups
        with verbosity(0):
            dp.add_dendrogram(size=1.6, dendrogram_key="dendrogram")
    if swap_axes:
        dp.swap_axes()

    dp.make_figure()

    if dendrogram != DendrogramAxis.BOTH:
        # remove the target part in: source | target
        labs = dp.ax_dict["mainplot_ax"].get_yticklabels() if swap_axes else dp.ax_dict["mainplot_ax"].get_xticklabels()
        for text in labs:
            text.set_text(text.get_text().split(_SEP)[1])
        if swap_axes:
            dp.ax_dict["mainplot_ax"].set_yticklabels(labs)
        else:
            dp.ax_dict["mainplot_ax"].set_xticklabels(labs)

    if alpha is not None:
        yy, xx = np.where(pvals.values >= -np.log10(alpha))
        if len(xx) and len(yy):
            # for dendrogram='both', they are already re-ordered
            mapper = (
                np.argsort(adata.uns["dendrogram"]["categories_idx_ordered"])
                if "dendrogram" in adata.uns
                else np.arange(len(pvals))
            )
            logg.info(f"Found `{len(yy)}` significant interactions at level `{alpha}`")
            ss = 0.33 * (adata.X[yy, xx] * (dp.largest_dot - dp.smallest_dot) + dp.smallest_dot)

            # must be after ss = ..., cc = ...
            yy = np.array([mapper[y] for y in yy])
            if swap_axes:
                xx, yy = yy, xx
            dp.ax_dict["mainplot_ax"].scatter(xx + 0.5, yy + 0.5, color="white", s=ss, lw=0)

    if dpi is not None:
        dp.fig.set_dpi(dpi)

    if save is not None:
        save_fig(dp.fig, save)
