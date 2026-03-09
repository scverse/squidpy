from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from spatialdata import SpatialData
from spatialdata._logging import logger as logg

from squidpy.experimental.im._qc_metrics import QCMetric


def qc_image(
    sdata: SpatialData,
    image_key: str,
    metrics: QCMetric | list[QCMetric] | None = None,
    figsize: tuple[int, int] | None = None,
    return_ax: bool = False,
    **kwargs: Any,
) -> np.ndarray | None:
    """
    Plot a summary view of QC metrics from qc_image results.

    Automatically scans adata.uns for calculated metrics and plots the values.
    Creates a multi-panel plot: one panel per calculated metric.
    Each panel shows: spatial view, KDE plot, and statistics.

    Parameters
    ----------
    sdata : SpatialData
        SpatialData object containing QC results.
    image_key : str
        Image key used in qc_image function.
    metrics : QCMetric or list of QCMetric, optional
        Specific metrics to plot. If None, plots all calculated metrics.
    figsize : tuple, optional
        Figure size (width, height). Auto-calculated if None.
    return_ax : bool
        Whether to return the axes array. Default is False.
    **kwargs
        Additional arguments passed to render_shapes().

    Returns
    -------
    axes : numpy.ndarray of matplotlib.axes.Axes or None
        The axes array if return_ax=True, otherwise None.
    """
    table_key = f"qc_img_{image_key}"
    shapes_key = f"qc_img_{image_key}_grid"

    if table_key not in sdata.tables:
        raise ValueError(f"No QC data found for image '{image_key}'. Run sq.exp.im.qc_image() first.")

    adata = sdata.tables[table_key]

    if "qc_image" not in adata.uns:
        raise ValueError("No qc_image metadata found. Run sq.exp.im.qc_image() first.")

    calculated_metrics = adata.uns["qc_image"]["metrics"]

    if not calculated_metrics:
        raise ValueError("No QC metrics found in metadata.")

    if metrics is not None:
        metrics_list = metrics if isinstance(metrics, list) else [metrics]
        metrics_to_plot = []
        for metric in metrics_list:
            metric_name = metric.value if isinstance(metric, QCMetric) else metric
            if metric_name not in calculated_metrics:
                raise ValueError(f"Metric '{metric_name}' not found. Available: {calculated_metrics}")
            metrics_to_plot.append(metric_name)
    else:
        metrics_to_plot = calculated_metrics

    logg.info(f"Plotting {len(metrics_to_plot)} QC metrics: {metrics_to_plot}")

    n_metrics = len(metrics_to_plot)
    ncols = 3
    nrows = n_metrics

    if figsize is None:
        figsize = (12, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    if nrows == 1:
        axes = axes.reshape(1, -1)
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for i, metric_name in enumerate(metrics_to_plot):
        var_name = f"qc_{metric_name}"
        if var_name not in adata.var_names:
            logg.warning(f"Variable '{var_name}' not found in adata.var_names. Skipping.")
            continue

        metric_idx = list(adata.var_names).index(var_name)
        raw_values = adata.X[:, metric_idx]

        ax_spatial = axes[i, 0]
        ax_hist = axes[i, 1]
        ax_stats = axes[i, 2]

        # Panel 1: Spatial plot
        try:
            (
                sdata.pl.render_shapes(shapes_key, color=var_name, table_name=table_key, **kwargs).pl.show(
                    ax=ax_spatial, title=f"{metric_name.replace('_', ' ').title()}"
                )
            )
        except (ValueError, KeyError, AttributeError) as e:
            logg.warning(f"Error plotting spatial view for {metric_name}: {e}")
            ax_spatial.text(
                0.5, 0.5, f"Error plotting\n{metric_name}", ha="center", va="center", transform=ax_spatial.transAxes
            )
            ax_spatial.set_title(f"{metric_name.replace('_', ' ').title()}")

        # Panel 2: KDE plot
        x_min, x_max = float(np.min(raw_values)), float(np.max(raw_values))

        if x_min < x_max:
            x_range = np.linspace(x_min, x_max, 200)

            if "is_tissue" in adata.obs:
                is_tissue = adata.obs["is_tissue"].astype(str) == "True"
                tissue_values = raw_values[is_tissue]
                background_values = raw_values[~is_tissue]

                if len(background_values) > 1 and np.std(background_values) > 0:
                    kde_background = gaussian_kde(background_values)
                    density_background = kde_background(x_range)
                    ax_hist.plot(x_range, density_background, label="Background", alpha=0.7)
                    ax_hist.fill_between(x_range, density_background, alpha=0.3)

                if len(tissue_values) > 1 and np.std(tissue_values) > 0:
                    kde_tissue = gaussian_kde(tissue_values)
                    density_tissue = kde_tissue(x_range)
                    ax_hist.plot(x_range, density_tissue, label="Tissue", alpha=0.7)
                    ax_hist.fill_between(x_range, density_tissue, alpha=0.3)

                ax_hist.legend()

            elif len(raw_values) > 1 and np.std(raw_values) > 0:
                kde = gaussian_kde(raw_values)
                density = kde(x_range)
                ax_hist.plot(x_range, density, alpha=0.7)
                ax_hist.fill_between(x_range, density, alpha=0.3)
        else:
            ax_hist.text(
                0.5,
                0.5,
                f"Constant value: {x_min:.4f}",
                ha="center",
                va="center",
                transform=ax_hist.transAxes,
            )

        ax_hist.set_xlabel(f"{metric_name.replace('_', ' ').title()}")
        ax_hist.set_ylabel("Density")
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

    return axes if return_ax else None
