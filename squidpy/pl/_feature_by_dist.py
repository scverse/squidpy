from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from tl.design_matrix import _subset_df


def plot_gexp_dist(
    adata: AnnData, design_matrix_key: Optional[str] = None,
    var: str = None,
    n_bins: int = 20,
    use_raw: Optional[bool] = False) -> Union[Figure, Axes, None]:
    """Plot gene expression by distance to anchor point."""
    if isinstance(var, str):
        var = [var]

    dfs = {}

    # get design matrix from anndata object
    if use_raw:
        df = adata.obsm[design_matrix_key + "_raw_dist"].copy()

    else:
        df = adata.obsm[design_matrix_key].copy()

    for v in var:
        # add var column to design matrix
        df[v] = sc.get.obs_df(adata, v).to_numpy()

        # set some plot settings depending on input
        if "batch_key" in adata.uns[design_matrix_key]:
            anchor = adata.uns[design_matrix_key]["batch_key"]
            x_axis_desc = f'{adata.uns[design_matrix_key]["metric"]} distance to {adata.uns[design_matrix_key]["annotation"]} cluster {adata.uns[design_matrix_key]["anchor"]} ({n_bins} bins)'
            #df = df.drop(adata.uns[design_matrix_key]["covariates"], axis=1)
            df_melt = df.rename(
                {str(adata.uns[design_matrix_key]["anchor"]): adata.uns[design_matrix_key]["metric"]}, axis=1
            )
        else:
            anchor = "anchor"
            x_axis_desc = f'{adata.uns[design_matrix_key]["metric"]} distance to anchor point ({n_bins} bins)'
            df_melt = df.melt(
                id_vars=[v, adata.uns[design_matrix_key]["annotation"], "x", "y"],
                var_name=anchor,
                value_name=adata.uns[design_matrix_key]["metric"],
            )

        # sort by euclidean distance
        df_melt.sort_values(adata.uns[design_matrix_key]["metric"], inplace=True)

        # create bins and get median from each binning interval
        df_melt["bin"] = pd.cut(df_melt[adata.uns[design_matrix_key]["metric"]], n_bins, include_lowest=True)

        df_melt[x_axis_desc] = df_melt.apply(lambda row: row["bin"].mid, axis=1)

        dfs[v] = df_melt

    # generate the plots
    for idx, v in enumerate(var):
        plt.subplot(1, len(var), idx + 1)
        plot = sns.lineplot(data=dfs[v], x=x_axis_desc, y=v, hue=anchor)
        plot.set(xlim=(0, dfs[v][adata.uns[design_matrix_key]["metric"]].max()))
    plt.show()

