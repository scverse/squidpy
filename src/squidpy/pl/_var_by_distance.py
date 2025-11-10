from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

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

from squidpy._docs import d
from squidpy.pl._utils import save_fig

__all__ = ["var_by_distance"]


@d.dedent
def var_by_distance(
    adata: AnnData,
    var: str | list[str],
    anchor_key: str | list[str],
    design_matrix_key: str = "design_matrix",
    stack_vars: bool = False,
    covariate: str | None = None,
    order: int = 5,
    show_scatter: bool = True,
    color: str | None = None,
    line_palette: str | Sequence[str] | Cycler | None = None,
    scatter_palette: str | Sequence[str] | Cycler | None = "viridis",
    dpi: int | None = None,
    figsize: tuple[int, int] | None = None,
    save: str | Path | None = None,
    title: str | None = None,
    axis_label: str | None = None,
    return_ax: bool | None = None,
    regplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    scatterplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
) -> Axes | None:
    """
    Plot a variable using a smooth regression line with increasing distance to an anchor point.

    Parameters
    ----------
    %(adata)s
    var
        Variables to plot on y-axis.
    anchor_key
        Anchor point column from which distances are taken.
    design_matrix_key
        Name of the design matrix, previously computed with :func:`squidpy.tl.var_by_distance`, to use.
    stack_vars
        Whether to show multiple variables on the same plot. Only works if 'covariate' is not specified.
    covariate
        A covariate for which separate regression lines are plotted for each category.
    order
        Order of the polynomial fit for :func:`seaborn.regplot`.
    show_scatter
        Whether to show a scatter plot underlying the regression line.
    color
        Variables in `adata.obs` to plot if 'show_scatter==True'.
    line_palette
        Categorical color palette used in case a covariate is specified.
    scatter_palette
        Color palette for the scatter plot underlying the :func:`seaborn.regplot`.
    dpi
        Dots per inch.
    figsize
        Size of the figure in inches.
    save
        Whether to save the plot.
    title
        Panel titles.
    axis_label
        Panel axis labels.
    return_ax
        Whether to return :class:`matplotlib.axes.Axes` object(s).
    regplot_kwargs
        Kwargs for :func:`seaborn.regplot`.
    scatterplot_kwargs
        Kwargs for :func:`matplotlib.pyplot.scatter`.

    Returns
    -------
    %(plotting_returns)s
    """
    dpi = rcParams["figure.dpi"] if dpi is None else dpi
    regplot_kwargs = dict(regplot_kwargs)
    scatterplot_kwargs = dict(scatterplot_kwargs)

    # if several variables are plotted, make a panel grid
    if isinstance(var, list) and not stack_vars:
        fig, grid = _panel_grid(
            hspace=0.25,
            wspace=0.75 / rcParams["figure.figsize"][0] + 0.02,
            ncols=4,
            num_panels=len(var),
        )
    elif isinstance(var, list) and stack_vars:
        var = var
    elif isinstance(var, str):
        var = [var]
    axs = []

    df = adata.obsm[design_matrix_key]  # get design matrix

    # add var column to design matrix
    for name in var:
        if name in adata.var_names:
            df[name] = (
                np.array(adata[:, name].X.toarray()) if issparse(adata[:, name].X) else np.array(adata[:, name].X)
            )
        elif name in adata.obs:
            df[name] = adata.obs[name].values
        else:
            raise ValueError(f"Variable {name} not found in `adata.var` or `adata.obs`.")

    if covariate is None and stack_vars:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if isinstance(line_palette, str) or line_palette is None:
            line_palette = sns.color_palette("bright", len(var))
        for i, v in enumerate(var):
            sns.regplot(
                data=df,
                x=anchor_key,
                y=v,
                label=v,
                order=order,
                color=line_palette[i],
                scatter=False,
                ax=ax,
                line_kws=regplot_kwargs,
            )
        ax.legend(title=None)
        ax.set(ylabel="var")
        if title is not None:
            ax.set(title=title)
        if axis_label is None:
            ax.set(xlabel=f"distance to {anchor_key}")
        else:
            ax.set(xlabel=axis_label)

    else:
        # iterate over the variables to plot
        for i, v in enumerate(var):
            if len(var) > 1 and not stack_vars:
                ax = plt.subplot(grid[i])
                axs.append(ax)
            else:
                # if a single variable and no covariate, then one ax object suffices
                fig, ax = plt.subplots(1, 1, figsize=figsize)

            # if no covariate is specified, 'sns.regplot' will take the values of all observations
            if covariate is None and not stack_vars:
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
            elif covariate is not None and not stack_vars:
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
            else:
                raise ValueError("Can't stack variables and plot covariate at the same time.")

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
                    plt.scatter(
                        data=df,
                        x=anchor_key,
                        y=v,
                        c=color,
                        cmap=scatter_palette,
                        **scatterplot_kwargs,
                    )
            if title is not None:
                ax.set(title=title)
            if axis_label is None:
                ax.set(xlabel=f"distance to {anchor_key}")
            else:
                ax.set(xlabel=axis_label)

    # remove line palette if it was made earlier in function
    if f"{covariate}_colors" in adata.uns:
        del line_palette

    axs = axs if len(var) and not stack_vars > 1 else ax

    if save is not None:
        save_fig(fig, path=save, transparent=False, dpi=dpi)
    if return_ax:
        return axs
