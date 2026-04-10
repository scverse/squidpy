"""Diagnostic plot for tiling segmentation QC."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialdata import SpatialData

__all__ = ["tiling_qc"]


def tiling_qc(
    sdata: SpatialData,
    labels_key: str,
    qc_key: str | None = None,
    score_col: str = "cut_score",
    cmap: str = "Reds",
    figsize: tuple[float, float] | None = None,
    **kwargs: Any,
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
        ``"cut_score"``, ``"max_straight_edge_ratio"``,
        ``"cardinal_alignment_score"``.
    cmap
        Matplotlib colormap name.
    figsize
        Figure size passed to :meth:`spatialdata.SpatialData.pl.show`.
    **kwargs
        Forwarded to :meth:`spatialdata.SpatialData.pl.render_labels`.
    """
    import spatialdata_plot  # noqa: F401  — registers accessor

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

    show_kwargs: dict[str, Any] = {}
    if figsize is not None:
        show_kwargs["figsize"] = figsize

    sdata.pl.render_labels(
        element=labels_key,
        color=score_col,
        table_name=table_key,
        cmap=cmap,
        **kwargs,
    ).pl.show(**show_kwargs)
