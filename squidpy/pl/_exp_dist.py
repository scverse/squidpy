from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, List, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.sparse import issparse

from squidpy.pl._utils import save_fig

__all__ = ["exp_dist"]


def exp_dist(
    adata: AnnData,
    design_matrix_key: str,
    var: str,
    anchor_key: str | List[str],
    color: str | None = None,
    order: int = 6,
    show_scatter: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    regplot_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> None:
    """
    Plot var expression by distance to anchor point.

    Parameters
    ----------
    adata
        Annotated data matrix.
    design_matrix_key
        Name of the design matrix, previously computed with tl._exp_dist, to use.
    var
        Variables to plot on y-axis.
    anchor_key
        Anchor point used to plot distances on x-axis
    color
        Variables to plot on color palette.
    order
        Order of the polynomial fit for `sns.regplot`.
    show_scatter
        Whether to show the scatter plot.
    %(plotting_save)s

    Returns
    -------
    %(plotting_returns)s
    """
    regplot_kwargs = dict(regplot_kwargs)
    df = adata.obsm[design_matrix_key]
    df[var] = np.array(adata[:, var].X.A) if issparse(adata[:, var].X) else np.array(adata[:, var].X)

    fig, ax = plt.subplots(1, 1)
    sns.regplot(data=df, x=anchor_key, y=var, order=order, color="black", scatter=show_scatter, ax=ax, **regplot_kwargs)
    ax.scatter(data=df, x=anchor_key, y=var, c=color, cmap="viridis", **kwargs)
    ax.set(xlabel=f"distance to {anchor_key}")
    if save is not None:
        save_fig(fig, path=save, transparent=False)
