from typing import Union, Optional

from anndata import AnnData
from scanpy.plotting import _utils
import scanpy as sc

import pandas as pd

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns


def exp_dist(
    adata: AnnData,
    design_matrix_key: str = None,
    var: str = None,
    n_bins: int = 20,
    show_model_fit: bool = False,
    raw_dist: Optional[bool] = False,
    use_raw: Optional[bool] = None,
    layer: Optional[str] = None,
    show: Optional[bool] = None,
    save: Union[bool, str, None] = None,
) -> Union[Figure, Axes, None]:
    """
    Plot gene expression by distance to anchor point.
    Parameters
    ----------
    adata
        Annotated data matrix.
    design_matrix_key
        Name of the design matrix previously computed with tl._exp_dist to use.
    var
        Variables to plot expression of.
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
    show
        Show the plot, do not return axis.
    save
        If `True` or a `str`, save the figure. A string is appended to the default filename. Infer the filetype if ending on {`'.pdf'`, `'.png'`, `'.svg'`}.
    Returns
    -------
    If `show==False` a `Axes` or a list of it.
    """
    if isinstance(var, str):
        var = [var]  # type: ignore[assignment]

    dfs = {}

    df = _get_data(adata=adata, key=design_matrix_key, func_name="_exp_dist", attr="obsm")

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
            # adapted from https://github.com/scverse/scanpy/blob/2e98705347ea484c36caa9ba10de1987b09081bf/scanpy/tools/_rank_genes_groups.py#L114-L121
            if layer is not None:
                if use_raw:
                    raise ValueError("Cannot specify `layer` and have `use_raw=True`.")
                layer = layer
                use_raw = False
            else:
                if use_raw and adata.raw is not None:
                    use_raw = use_raw
                else:
                    use_raw = False
            # add var column to design matrix
            df[v] = sc.get.obs_df(adata, v, layer=layer, use_raw=use_raw).to_numpy()

        # set some plot settings depending on input
        if "batch_key" in adata.uns[design_matrix_key]:
            anchor = adata.uns[design_matrix_key]["batch_key"]
            x_axis_desc = f'{adata.uns[design_matrix_key]["metric"]} distance to {adata.uns[design_matrix_key]["annotation"]} cluster {adata.uns[design_matrix_key][anchor_type]} ({n_bins} bins)'
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

        # sort by distance
        df_melt.sort_values(adata.uns[design_matrix_key]["metric"], inplace=True)

        # create bins and get median from each binning interval
        df_melt["bin"] = pd.cut(df_melt[adata.uns[design_matrix_key]["metric"]], n_bins, include_lowest=True)

        df_melt[x_axis_desc] = df_melt.apply(lambda row: row["bin"].mid, axis=1)

        dfs[v] = df_melt

    # generate the plots
    name = "exp_by_dist_" + "_".join(var)
    for idx, v in enumerate(var):
        plt.subplot(1, len(var), idx + 1)
        plot = sns.lineplot(data=dfs[v], x=x_axis_desc, y=v, hue=anchor)
        plot.set(xlim=(0, dfs[v][adata.uns[design_matrix_key]["metric"]].max()))
    _utils.savefig_or_show(name, show=show, save=save)


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
            f"Unable to get the data from 'adata.{attr}[{key}]'. " f"Please run `squidpy.tl.{func_name}' first."
        )
