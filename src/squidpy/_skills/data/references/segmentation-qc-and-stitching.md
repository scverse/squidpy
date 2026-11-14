# Tiled segmentation QC and stitching — Squidpy 1.8.4

All three stages work with a SpatialData labels element. Do not skip the middle assignment stage. The pipeline identifies and repairs _tile-boundary cuts_; it is not a general-purpose segmentation or cell-tracking system.

## Stage 1: measure possible cut artifacts

```python
sq.experimental.tl.calculate_tiling_qc(
    sdata, labels_key="cell_labels", table_key_added="cell_labels_qc"
)
```

Generates an AnnData QC table in `sdata.tables["cell_labels_qc"]` (default key `f"{labels_key}_qc"`). Per-cell QC fields go in `.obs`; `.X` is empty, and settings in `.uns["tiling_qc"]`. Default scoring uses straight cardinal-aligned edges and neighborhood-based outlier information. Inspect `is_outlier`, `cut_score`, `smoothed_cut_score`, `nhood_outlier_fraction` and relevant QC metadata. `sq.experimental.pl.tiling_qc(...)` can visualize scores.

## Stage 2: identify groups, not stitched images

```python
sq.experimental.tl.assign_stitch_groups(
    sdata, labels_key="cell_labels", qc_table_key="cell_labels_qc"
)
```

Consumes flagged outlier cells, proposes geometric pairings across cut boundaries, and adds group annotations to the QC table. It does **not** alter the labels image. `min_confidence=0.7` is a heuristic threshold, **not** a calibrated probability; inspect false positives and adjust. `max_group_size=4` caps merged group size by default.

## Stage 3: materialize a NEW labels element

```python
sq.experimental.tl.make_stitched_labels(
    sdata,
    labels_key="cell_labels",
    qc_table_key="cell_labels_qc",
    labels_key_added="cell_labels_stitched",
    write_table=True,
    merge_strategy="sum",
)
```

This creates a new labels element; original `"cell_labels"` remains intact. The optional table contains one row per stitch group, with feature columns aggregated. **The `merge_strategy` applies to genuine numeric user feature columns and `.X`**; group annotations, centroid coordinates, and QC scores have their own aggregation rules. `"sum"` is suitable only for additive values; `"mean"` may be more appropriate for non-additive intensity features. Review the scientific meaning before choosing.

With `join_labels=False` (default), fragments can share a label ID without being physically connected. Use `join_labels=True` to close small gaps only when scientifically justified. Existing target labels keys can be overwritten with a warning: select fresh names. With `inplace=False`, the function returns `{"labels": ..., "table": ...}`.

## Checks

- Validate candidate pairs against a small image patch before final materialization.
- Confirm original labels are unchanged and new IDs correspond to stitched groups.
- Confirm the new table's cell count and feature aggregation semantics; don't sum intensity means blindly.
- Inspect seams after `join_labels`; do not erase neighboring cells.

Sources:
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.tl.calculate_tiling_qc.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.tl.assign_stitch_groups.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.tl.make_stitched_labels.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.pl.tiling_qc.html
