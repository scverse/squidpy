from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from spatialdata._logging import logger as logg

from squidpy.exp.im._qc import SHARPNESS_METRICS


def qc_sharpness_metrics(
    sdata,
    image_key: str,
    metrics: SHARPNESS_METRICS | list[SHARPNESS_METRICS] | None = None,
    figsize: tuple[int, int] | None = None,
    return_fig: bool = False,
    **kwargs,
) -> plt.Figure | None:
    """
    Plot a summary view of raw sharpness metrics from qc_sharpness results.

    Automatically scans adata.uns for calculated metrics and plots the raw sharpness values.
    Creates a multi-panel plot: one panel per calculated sharpness metric.
    Each panel shows: spatial view, histogram, and statistics.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing QC results.
    image_key : str
        Image key used in qc_sharpness function.
    metrics : SHARPNESS_METRICS or list of SHARPNESS_METRICS, optional
        Specific metrics to plot. If None, plots all calculated sharpness metrics.
        Use SHARPNESS_METRICS enum values.
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.
    return_fig : bool
        Whether to return the figure object. Default is False.
    **kwargs
        Additional arguments passed to render_shapes().

    Returns
    -------
    fig : matplotlib.Figure or None
        The matplotlib figure object if return_fig=True, otherwise None.
    """
    import matplotlib.pyplot as plt

    # Expected keys
    table_key = f"qc_img_{image_key}_sharpness"
    shapes_key = f"qc_img_{image_key}_sharpness_grid"

    if table_key not in sdata.tables:
        raise ValueError(f"No QC data found for image '{image_key}'. Run sq.exp.im.qc_sharpness() first.")

    adata = sdata.tables[table_key]

    # Check if qc_sharpness metadata exists
    if "qc_sharpness" not in adata.uns:
        raise ValueError("No qc_sharpness metadata found. Run sq.exp.im.qc_sharpness() first.")

    # Get calculated metrics from metadata
    calculated_metrics = adata.uns["qc_sharpness"]["metrics"]

    if not calculated_metrics:
        raise ValueError("No sharpness metrics found in metadata.")

    # Filter for specific metrics if requested
    if metrics is not None:
        # Convert metrics to list if single metric provided
        metrics_list = metrics if isinstance(metrics, list) else [metrics]
        # Convert enum to string names using the same logic as main function
        metrics_to_plot = []
        for metric in metrics_list:
            metric_name = metric.name.lower() if isinstance(metric, SHARPNESS_METRICS) else metric
            if metric_name not in calculated_metrics:
                raise ValueError(f"Metric '{metric_name}' not found. Available: {calculated_metrics}")
            metrics_to_plot.append(metric_name)
    else:
        metrics_to_plot = calculated_metrics

    logg.info(f"Plotting {len(metrics_to_plot)} sharpness metrics: {metrics_to_plot}")

    # Create subplots: 3 columns, one row per metric
    n_metrics = len(metrics_to_plot)
    ncols = 3  # spatial, histogram, stats
    nrows = n_metrics

    if figsize is None:
        figsize = (12, 4 * nrows)  # 12 width for 3 columns, 4 height per row

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always 2D array for consistent indexing
    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    # Plot each metric
    for i, metric_name in enumerate(metrics_to_plot):
        # Find the metric in adata.var_names and get raw values
        var_name = f"sharpness_{metric_name}"
        if var_name not in adata.var_names:
            logg.warning(f"Variable '{var_name}' not found in adata.var_names. Skipping.")
            continue

        # Get metric index and raw values
        metric_idx = list(adata.var_names).index(var_name)
        raw_values = adata.X[:, metric_idx]

        # Get axes for this metric (row i, columns 0, 1, 2)
        ax_spatial = axes[i, 0]
        ax_hist = axes[i, 1]
        ax_stats = axes[i, 2]

        # Panel 1: Spatial plot
        try:
            (
                sdata.pl.render_shapes(shapes_key, color=var_name, **kwargs).pl.show(
                    ax=ax_spatial, title=f"{metric_name.replace('_', ' ').title()}"
                )
            )
        except Exception as e:
            logg.warning(f"Error plotting spatial view for {metric_name}: {e}")
            ax_spatial.text(
                0.5, 0.5, f"Error plotting\n{metric_name}", ha="center", va="center", transform=ax_spatial.transAxes
            )
            ax_spatial.set_title(f"{metric_name.replace('_', ' ').title()}")

        # Panel 2: Histogram
        ax_hist.hist(raw_values, bins=50, alpha=0.7, edgecolor="black")
        ax_hist.set_xlabel(f"{metric_name.replace('_', ' ').title()}")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title("Distribution")
        ax_hist.grid(True, alpha=0.3)

        # Panel 3: Statistics
        ax_stats.axis("off")
        stats_text = f"""
        Raw {metric_name.replace("_", " ").title()} Statistics:
        
        Count: {len(raw_values):,}
        Mean: {np.mean(raw_values):.4f}
        Std: {np.std(raw_values):.4f}
        Min: {np.min(raw_values):.4f}
        Max: {np.max(raw_values):.4f}
        
        Percentiles:
        5%: {np.percentile(raw_values, 5):.4f}
        25%: {np.percentile(raw_values, 25):.4f}
        50%: {np.percentile(raw_values, 50):.4f}
        75%: {np.percentile(raw_values, 75):.4f}
        95%: {np.percentile(raw_values, 95):.4f}
        
        Non-zero: {np.count_nonzero(raw_values):,}
        Zero: {np.sum(raw_values == 0):,}
        """

        ax_stats.text(
            0.05,
            0.95,
            stats_text.strip(),
            transform=ax_stats.transAxes,
            fontsize=9,
            verticalalignment="top",
            fontfamily="monospace",
        )

    plt.tight_layout()

    return fig if return_fig else None
