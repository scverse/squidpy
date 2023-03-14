from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, List, Mapping, Union

import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import seaborn as sns
from anndata import AnnData
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from squidpy.pl._utils import save_fig

__all__ = ["exp_dist"]


def exp_dist(
    adata: AnnData,
    design_matrix_key: str,
    var: str,
    anchor_key: str | List[str],
    show_scatter: bool = True,
    figsize: tuple[float, float] | None = None,
    dpi: int | None = None,
    save: str | Path | None = None,
    legend_kwargs: Mapping[str, Any] = MappingProxyType({}),
    **kwargs: Any,
) -> Union[Figure, Axes, None]:
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
    save
        Whether to save the plot.

    Returns
    -------

    """
    sc.settings.set_figure_params(dpi=dpi, facecolor="white")
    adata.obsm[design_matrix_key][var] = np.array(adata[:, var].X)
    df = adata.obsm[design_matrix_key]
    fig, ax = plt.subplots(1, 1)
    sns.regplot(
        data=df, x=anchor_key, y=var, color="blue", order=6, scatter=show_scatter, ax=ax, scatter_kws={"color": "grey"}
    )
    if save is not None:
        save_fig(fig, path=save, transparent=False)
