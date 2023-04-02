from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from cycler import Cycler
from matplotlib import rcParams
from matplotlib.axes import Axes
from scanpy.plotting._tools.scatterplots import _panel_grid
from scanpy.plotting._utils import _set_default_colors_for_categorical_obs
from scipy.sparse import issparse

from squidpy.pl._utils import save_fig

__all__ = ["var_by_distance"]


def var_by_distance(
    adata: AnnData,
    design_matrix_key: str,
    var: str | List[str],
    anchor_key: str | List[str],
    color: str | None = None,
    covariate: str | None = None,
    order: int = 5,
    show_scatter: bool = True,
    dpi: int | None = None,
    figsize: Tuple[int, int] | None = None,
    line_palette: Union[str, Sequence[str], Cycler, None] = None,
    scatter_palette: Union[str, Sequence[str], Cycler, None] = "viridis",
    save: str | Path | None = None,
    return_ax: Optional[bool] = None,
    regplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    scatterplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Union[Axes, None]:
    """
    Plot var expression by distance to anchor point.

    Parameters
    ----------
    adata
        Annotated data matrix.
    design_matrix_key
        Name of the design matrix, previously computed with :func:`squidpy.tl.var_by_distance`, to use.
    var
        Variables to plot on y-axis.
    anchor_key
        anchor point column from which distances are taken.
    color
        Variables to plot on color palette.
    covariate
        A covariate for which separate regression lines are plotted for each category.
    order
        Order of the polynomial fit for :func:`seaborn.regplot`.
    dpi
        Dpi value of the plot.
    figsize
        Size of the plot in inches.
    line_palette
        Categorical color palette used in case a covariate is specified.
    scatter_palette
        Color palette for the scatter plot underlying the `sns.regplot`
    save
        Path to save the figure to.
    return_ax
        Whether to return `matplotlib.axes.Axes` object(s).
    regplot_kwargs
        kwargs for `sns.regplot`
    scatterplot_kwargs
        kwargs for `sns.scatter`
    %(plotting_save)s
    %(plotting_dpi)s
    %(plotting_figsize)s

    Returns
    -------
    %(plotting_returns)s
    """
    dpi = rcParams["figure.dpi"] if dpi is None else dpi
    regplot_kwargs = dict(regplot_kwargs)
    scatterplot_kwargs = dict(regplot_kwargs)

    df = adata.obsm[design_matrix_key]  # get design matrix
    df[var] = np.array(adata[:, var].X.A) if issparse(adata[:, var].X) else np.array(adata[:, var].X)  # add var column

    # if several variables are plotted, make a panel grid
    if isinstance(var, List):
        fig, grid = _panel_grid(
            hspace=0.25, wspace=0.75 / rcParams["figure.figsize"][0] + 0.02, ncols=4, num_panels=len(var)
        )
        axs = []
    else:
        var = [var]

    # iterate over the variables to plot
    for i, v in enumerate(var):
        if len(var) > 1:
            ax = plt.subplot(grid[i])
            axs.append(ax)
        else:
            # if a single variable and no grid, then one ax object suffices
            fig, ax = plt.subplots(1, 1, figsize=figsize)

        # if no covariate is specified, 'sns.regplot' will take the values of all observations
        if covariate is None:
            sns.regplot(
                data=df,
                x=anchor_key,
                y=v,
                order=order,
                color=line_palette,
                scatter=show_scatter,
                ax=ax,
                line_kws=regplot_kwargs,
            )
        else:
            # make a categorical color palette if none was specified and there are several regplots to be plotted
            if isinstance(line_palette, str) or line_palette is None:
                _set_default_colors_for_categorical_obs(adata, covariate)
                line_palette = adata.uns[covariate + "_colors"]
            covariate_instances = df[covariate].unique()

            # iterate over all covariate values and make 'sns.regplot' for each
            for i, co in enumerate(covariate_instances):
                sns.regplot(
                    data=df.loc[df[covariate] == co],
                    x=anchor_key,
                    y=v,
                    order=order,
                    color=line_palette[i],
                    scatter=show_scatter,
                    ax=ax,
                    label=co,
                    line_kws=regplot_kwargs,
                )
            label_colors, _ = ax.get_legend_handles_labels()
            ax.legend(label_colors, covariate_instances)
        # add scatter plot if specified
        if show_scatter:
            if color is None:
                plt.scatter(data=df, x=anchor_key, y=v, color="grey", **scatterplot_kwargs)
            # if variable to plot on color palette is categorical, make categorical color palette
            elif df[color].dtype.name == "category":
                unique_colors = df[color].unique()
                cNorm = colors.Normalize(vmin=0, vmax=len(unique_colors))
                scalarMap = cm.ScalarMappable(norm=cNorm, cmap=scatter_palette)
                for i in range(len(unique_colors)):
                    plt.scatter(
                        data=df.loc[df[color] == unique_colors[i]],
                        x=anchor_key,
                        y=v,
                        color=scalarMap.to_rgba(i),
                        **scatterplot_kwargs,
                    )
            # if variable to plot on color palette is not categorical
            else:
                plt.scatter(data=df, x=anchor_key, y=v, c=color, cmap=scatter_palette, **scatterplot_kwargs)
        ax.set(xlabel=f"distance to {anchor_key}")

    # remove line palette if it was made earlier in function
    if f"{covariate}_colors" in adata.uns:
        del line_palette

    axs = axs if len(var) > 1 else ax

    if save is not None:
        save_fig(fig, path=save, transparent=False, dpi=dpi)
    if return_ax:
        return axs