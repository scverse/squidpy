from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from squidpy.pl._utils import save_fig


def exp_dist(
    adata: AnnData,
    var: str,
    design_matrix_key: str = "design_matrix",
    n_bins: int = 20,
    show_model_fit: bool = False,
    raw_dist: Optional[bool] = False,
    layer: Optional[str] = None,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Union[Figure, Axes, None]:
    """
    Plot gene expression by distance to anchor point.

    Parameters
    ----------
    adata
        Annotated data matrix.
    var
        Variables to plot expression of.
    design_matrix_key
        Name of the design matrix previously computed with tl._exp_dist to use.
    n_bins
        Number of bins to use for plotting.
    show_model_fit
        If `True` plot fitted values from `tl.spatial_de` model fit for each var instead of counts from `X.`
    raw_dist
        If `True` use raw distance from anchor point instead of normalized distance to plot on x-axis.
    use_raw
        Use `raw` attribute of `adata` if present.
    layer
        sKey from `adata.layers` whose value will plotted on the y-axis.
    save
        Whether to save the plot.

    Returns
    -------
    If `show==False` a `Axes` or a list of it.
    """
    if isinstance(var, str):
        var = [var]  # type: ignore[assignment]

    dfs = {}

    df = adata.obsm[design_matrix_key].copy()

    if raw_dist:
        anchor_type = "anchor_raw"
        df = df[
            [
                value
                for key, value in adata.uns[design_matrix_key].items()
                if "anchor_raw" in key or "annotation" in key or "batch" in key
            ]
        ]
    else:
        anchor_type = "anchor_scaled"
        df = df[
            [
                value
                for key, value in adata.uns[design_matrix_key].items()
                if "anchor_scaled" in key or "annotation" in key or "batch" in key
            ]
        ]

    for v in var:
        if show_model_fit:
            # add var column with fitted values from model to design matrix
            df[v] = adata.uns[design_matrix_key + "_fitted_values"][[v]]
        else:
            # adapted from
            # https://github.com/scverse/scanpy/blob/2e98705347ea484c36caa9ba10de1987b09081bf/scanpy/tools/_rank_genes_groups.py#L114-L121
            # add var column to design matrix
            df[v] = sc.get.obs_df(adata, v, layer=layer).to_numpy()

        # set variables
        anchor = adata.uns[design_matrix_key]["batch_key"]
        metric = adata.uns[design_matrix_key]["metric"]
        annotation = adata.uns[design_matrix_key]["annotation"]
        anchor_type_ = adata.uns[design_matrix_key][anchor_type]
        # set some plot settings depending on input
        if "batch_key" in adata.uns[design_matrix_key]:
            x_axis_desc = f"{metric} distance to {annotation} cluster {anchor_type_} ({n_bins} bins)"
            df_melt = df.rename({str(anchor_type_): metric}, axis=1)
        else:
            anchor = "anchor"
            x_axis_desc = f"{metric} distance to anchor point ({n_bins} bins)"
            df_melt = df.melt(
                id_vars=[v, annotation],
                var_name=anchor,
                value_name=metric,
            )

        # sort by distance
        df_melt.sort_values(adata.uns[design_matrix_key]["metric"], inplace=True)

        # create bins and get median from each binning interval
        df_melt["bin"] = pd.cut(df_melt[adata.uns[design_matrix_key]["metric"]], n_bins, include_lowest=True)

        df_melt[x_axis_desc] = df_melt.apply(lambda row: row["bin"].mid, axis=1)

        dfs[v] = df_melt

    # generate the plots
    "exp_by_dist_" + "_".join(var)

    fig, axs = plt.subplots(
        1,
        len(var),
        figsize=(5 * len(var), 5) if figsize is None else figsize,
        dpi=dpi,
        constrained_layout=True,
    )
    axs = np.ravel(axs)  # make into iterable

    for v, ax in zip(var, axs):
        sns.lineplot(
            data=dfs[v],
            x=x_axis_desc,
            y=v,
            hue=anchor,
            ax=ax,
        )
        ax.set(xlim=(0, dfs[v][adata.uns[design_matrix_key]["metric"]].max()))
        ax.set_title(v)
        # ax.set_xlabel("value")

        # ax.set_yticks([])
        ax.legend(**legend_kwargs)

    if save is not None:
        save_fig(fig, path=save)
