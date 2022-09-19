from typing import Optional, Union

import pandas as pd
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def exp_dist(
    adata: AnnData, design_matrix_key: str = None, var: str = None, n_bins: int = 20, use_raw: Optional[bool] = False
) -> Union[Figure, Axes, None]:
    """Plot gene expression by distance to anchor point. Several slides or anchor points possible"""
    if isinstance(var, str):
        var = [var]  # type: ignore[assignment]

    dfs = {}

    df = _get_data(adata=adata, key=design_matrix_key, func_name="_exp_dist", attr="obsm")

    if use_raw:
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
        # add var column to design matrix
        df[v] = sc.get.obs_df(adata, v).to_numpy()

        # set some plot settings depending on input
        if "batch_key" in adata.uns[design_matrix_key]:
            anchor = adata.uns[design_matrix_key]["batch_key"]
            x_axis_desc = f'{adata.uns[design_matrix_key]["metric"]} distance to {adata.uns[design_matrix_key]["annotation"]} cluster {adata.uns[design_matrix_key][anchor_type]} ({n_bins} bins)'
            # df = df.drop(adata.uns[design_matrix_key]["covariates"], axis=1)
            df_melt = df.rename(
                {str(adata.uns[design_matrix_key][anchor_type]): adata.uns[design_matrix_key]["metric"]}, axis=1
            )
        else:
            anchor = "anchor"
            x_axis_desc = f'{adata.uns[design_matrix_key]["metric"]} distance to anchor point ({n_bins} bins)'
            df_melt = df.melt(
                id_vars=[v, adata.uns[design_matrix_key]["annotation"]],
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
    # plt.savefig("exp_by_dist_pruned_anchor_tree.PDF")
    plt.show()


# adapted from https://github.com/scverse/squidpy/blob/2cf664ffd9a1654b6d921307a76f5732305a371c/squidpy/pl/_graph.py#L32-L40
def _get_data(adata: AnnData, key: str, func_name: str, attr: str = "obsm") -> Any:
    try:
        if attr == "obsm":
            return adata.obsm[key]
        elif attr == "uns":
            return adata.uns[key]
        else:
            raise ValueError(f"attr must be either 'uns' or 'obsm', got {attr}")
    except KeyError:
        raise KeyError(
            f"Unable to get the dat from 'adata.{attr}[{key}]'. " f"Please run `squidpy.tl.{func_name}' first."
        )
