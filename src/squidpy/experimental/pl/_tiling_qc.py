"""Diagnostic plot for tiling segmentation QC."""

from __future__ import annotations

from typing import Literal

import spatialdata as sd

__all__ = ["tiling_qc"]


def tiling_qc(
    sdata: sd.SpatialData,
    labels_key: str,
    qc_key: str | None = None,
    score_col: Literal[
        "nhood_outlier_fraction",
        "smoothed_cut_score",
        "cut_score",
        "max_straight_edge_ratio",
        "cardinal_alignment_score",
        "is_outlier",
    ] = "nhood_outlier_fraction",
    cmap: str = "RdYlGn_r",
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot labels coloured by their tiling-artifact score.

    Uses :mod:`spatialdata_plot` to render the label element coloured
    by the chosen QC score from the linked table.  If tile-boundary
    artifacts are present the tile grid emerges as lines of
    high-scoring cells.

    Parameters
    ----------
    sdata
        SpatialData object (must contain the QC table).
    labels_key
        Key in ``sdata.labels`` with the segmentation mask.
    qc_key
        Key in ``sdata.tables`` with the QC AnnData.  Defaults to
        ``"{labels_key}_qc"``.
    score_col
        Which ``.obs`` column to colour by.  One of
        ``"nhood_outlier_fraction"``, ``"smoothed_cut_score"``,
        ``"cut_score"``, ``"max_straight_edge_ratio"``,
        ``"cardinal_alignment_score"``, ``"is_outlier"``.
    cmap
        Matplotlib colormap name.
    figsize
        Figure size passed to :meth:`spatialdata.SpatialData.pl.show`.
    """
    table_key = qc_key if qc_key is not None else f"{labels_key}_qc"
    if table_key not in sdata.tables:
        raise ValueError(
            f"QC table '{table_key}' not found in sdata.tables. "
            f"Run calculate_tiling_qc(sdata, labels_key='{labels_key}') first."
        )

    adata = sdata.tables[table_key]
    if score_col not in adata.obs.columns:
        raise ValueError(
            f"Score column '{score_col}' not found in .obs. "
            f"Available: {[c for c in adata.obs.columns if c not in ('region', 'label_id')]}"
        )

    import spatialdata_plot  # noqa: F401  - registers accessor

    _TITLES = {
        "nhood_outlier_fraction": "Neighborhood outlier fraction",
        "smoothed_cut_score": "Smoothed cut score",
        "cut_score": "Cut score",
        "is_outlier": "Outlier flag",
        "max_straight_edge_ratio": "Max straight edge ratio",
        "cardinal_alignment_score": "Cardinal alignment score",
    }

    show_kwargs: dict[str, object] = {"title": _TITLES.get(score_col, score_col)}
    if figsize is not None:
        show_kwargs["figsize"] = figsize

    sdata.pl.render_labels(
        element=labels_key,
        color=score_col,
        table_name=table_key,
        cmap=cmap,
        colorbar=True,
    ).pl.show(**show_kwargs)
