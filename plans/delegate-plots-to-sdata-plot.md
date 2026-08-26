# Delegate plots to spatialdata-plot

Tracking issue: scverse/squidpy#912.

## Goal

Replace squidpy's spatial plotting internals with `spatialdata-plot` calls while keeping user-facing signatures unchanged during the deprecation window. Drop the AnnData-input path and the `sq.read.*` readers at v2.0; both are superseded by `spatialdata-io` + `SpatialData` input.

This is a deprecation effort, not a permanent abstraction layer. The AnnData -> SpatialData shim inside the plot wrapper is short-lived and best-effort, not architecture.

## Scope

In scope:
- Deprecate `sq.read.visium`, `sq.read.nanostring`, `sq.read.vizgen`, and any other AnnData-producing readers in `sq.read`.
- Migrate `sq.pl.spatial_scatter` and `sq.pl.spatial_segment` to delegate to `spatialdata-plot >= 0.3.4`.
- Keep public signatures unchanged. Internals route through `render_shapes` / `render_points` / `render_labels` / `render_images` and `show`.
- Accept both AnnData and SpatialData input during the window; emit `DeprecationWarning` on AnnData.

Out of scope for this initiative:
- `sq.pl.nhood_enrichment`, `sq.pl.co_occurrence`, `sq.pl.interaction_matrix`, `sq.pl.centrality_scores`, `sq.pl.ripley`, `sq.pl.var_by_distance`. Statistics plots consume analysis results from `.uns`/`.obsp`/`.obsm` and have no `spatialdata-plot` rendering equivalent today. Separate later milestone if migrated at all.
- `sq.pl.ligrec`. Rank 2 by user engagement (93 historical comments) but `spatialdata-plot` has no cellphoneDB-style dotplot. Decide later whether to upstream or keep native.
- `sq.pl.extract` is a `obsm` -> `obs` data utility, not a plot. Untouched.
- `sq.gr.*` analysis functions. Whether they continue to write results into AnnData or into `sdata.tables['table']` is a separate decision.
- napari integration in `sq.im`/`napari-spatialdata`.

## Plotting surface inventory

Full audit of `sq.pl.*` (10 entries):

| Function | Modality | Classification |
|---|---|---|
| `spatial_scatter` | Coords + optional image, parametric markers | Delegate (Stage 2) |
| `spatial_segment` | Coords + image + raster mask | Delegate (Stage 2) |
| `ligrec` | Dotplot (size + color matrix) | Native, future decision |
| `centrality_scores` | Stat scatter per cluster | Native |
| `interaction_matrix` | Matrix heatmap | Native |
| `nhood_enrichment` | Matrix heatmap | Native |
| `ripley` | Line plot vs distance | Native |
| `co_occurrence` | Per-cluster line plots | Native |
| `var_by_distance` | Seaborn regression plot | Native |
| `extract` | Data utility (not a plot) | N/A |

`spatial_scatter` and `spatial_segment` share ~80% of their kwarg surface. Differentiators: scatter owns `shape`/`size`/`size_key`/`scale_factor`/`outline*`/`connectivity_key`/`edges_*`; segment owns `seg_cell_id`/`seg`/`seg_key`/`seg_contourpx`/`seg_outline`. This justifies a single `Intent` shape with element-existence booleans on `DataIntent` rather than a `ScatterIntent | SegmentIntent` union.

## Intent design (locked)

Internal wrapper structure (not public API):

```
def spatial_scatter(input, **kwargs):
    intent = capture_plotting_intent(mode="scatter", **kwargs)
    intent = resolve_intent(input, intent)  # adds defaults from data
    sdata = input if isinstance(input, SpatialData) else _make_tmp_sdata(input, intent)
    return _render_from_intent(sdata, intent)
```

Four lifecycle buckets:

**DataIntent** (drives `_make_tmp_sdata` and SpatialData element selection)
- Element existence flags: `needs_shapes`, `needs_labels`, `needs_points`, `needs_image`, `needs_graph`
- Element names: `shapes_layer`, `labels_layer`, `image_layer`, `points_layer`, `graph_layer`
- Library selection: `library_ids`, `library_key`
- Coordinate system: `coordinate_system`
- Image source: `img_res_key`, `img_channel`
- Color source resolution: `color`, `use_raw`, `layer`, `alt_var`
- Size source: `size`, `size_key`, `scale_factor` (scatter only)
- Crop: `crop_coord` per library
- Segmentation mapping: `seg_cell_id` (segment only)

**RenderIntent** (per-element kwargs passed to sdata-plot render calls)
- Color encoding: `cmap`, `norm` (vmin/vmax/vcenter folded in at capture), `palette`, `alpha`, `na_color`, `groups`
- Element kind decision: `shape` (drives `render_shapes` vs `render_points`)
- Image styling: `img_alpha`, `img_cmap`
- Mask styling: `contour_px` (translated from `seg_contourpx`), outline alpha (translated from `seg_outline`)
- Outline tuples: `outline`, `outline_color`, `outline_width` -> chain renders the element 3 times (bg, gap, fg) on the same ax
- Graph styling: `edges_width`, `edges_color`, `edges_kwargs` -> passed to `render_graph`

**LayoutIntent** (matplotlib figure setup before render)
- Panel grid: `ncols`, `library_first`, `wspace`, `hspace`
- Figure: `figsize`, `dpi`, `fig`, `ax`, `frameon`
- Return mode: `return_ax`

**PostRenderIntent** (applied to returned axes after `show()`)
- Titles: `title`, `axis_label`
- Legend: `legend_loc` incl. `'on data'` centroid-text interception, `legend_fontsize`, `legend_fontweight`, `legend_fontoutline`, `legend_na`
- Colorbar: `colorbar`
- Scalebar: `scalebar_dx`, `scalebar_units`, `scalebar_kwargs` (passthrough to `matplotlib_scalebar`; sdata-plot v0.3.4 wires the first two through `show()`)
- Save: `save`

### Locked design decisions

1. **Panel expansion happens at capture.** `capture_plotting_intent` flattens `(library_ids x color)` into `Intent.panels: list[PanelIntent]`. Render code is a single loop over panels. Per-library values (`size`, `scalebar_dx`, `crop_coord`) live on `PanelIntent`, not `Intent` root.
2. **Outline effect lives in RenderIntent** as a flag. Render chain renders the element 3 times (bg, gap, fg) on the same ax. No PostRender re-render, no upstream blocker.
3. **Connectivity edges are a sibling render call**, not a PostRender hook. `needs_graph` + `graph_layer` on DataIntent; render chain inserts `render_graph()` ahead of `render_points/shapes` so points sit on top. Replaces squidpy's current pre-image `_plot_edges` call.
4. **`legend_loc='on data'`** is intercepted at capture (sdata-plot rejects it in PR #649). PostRender places centroid text on the returned ax after `show()`.
5. **Element-name ambiguity on SpatialData input**: if multiple shapes/labels elements exist for the selected coordinate system, the wrapper requires the user to pass explicit `shapes_layer=`/`labels_layer=` (new kwargs on the public signature). Mirrors scanpy's `layer=`.
6. **`seg_contourpx=1`** is rejected by sdata-plot PR #645; capture validates and raises with a clear message rather than passing through.

## Version timeline

Current release: `v1.8.1`.

| Version | Action |
|---|---|
| `v1.9.0` | Stage 1. `DeprecationWarning` on every `sq.read.*` function pointing at the `spatialdata-io` equivalent. No removal. Tutorials updated to `spatialdata-io`. |
| `v1.10.0` (or `v1.9.x` if cadence permits) | Stage 2. `spatial_scatter` and `spatial_segment` accept SpatialData natively; AnnData input still accepted with `DeprecationWarning` and routed through the shim. |
| `v2.0.0` | Stage 3 + 4. Remove `sq.read.*`. Remove AnnData input path and shim from `spatial_scatter` / `spatial_segment`. Drop AnnData-side tests. |

Hard rule: no removals before `v2.0.0`. Warnings only during the window.

## Stage 1: deprecate readers (`v1.9.0`)

One PR. Touches `src/squidpy/read/*.py`, docs, tutorials.

Changes per reader:
- At top of function body: `warnings.warn(..., DeprecationWarning, stacklevel=2)` with a message naming the `spatialdata-io` replacement (`spatialdata_io.visium`, `spatialdata_io.nanostring`, etc.) and the removal target (`v2.0.0`).
- Docstring gains a `.. deprecated:: 1.9.0` directive with the same pointer.
- No behavior change.

Docs:
- Migration note in `docs/release_notes.md`.
- Update the "Reading data" section to lead with `spatialdata-io`; reduce `sq.read.*` to a deprecated-reference block.
- Update tutorial notebooks that currently call `sq.read.*` to use `spatialdata-io` instead. Identify these via `grep -rn "sq.read\|squidpy.read" docs/ docs/notebooks/ 2>/dev/null` before the PR.

Tests:
- Add a test per reader asserting `DeprecationWarning` fires.
- Existing reader tests stay green (warning is not an error).

## Stage 2: dual-input plot delegation (`v1.10.0`)

One PR per top function (two PRs total). Land `spatial_scatter` first.

### Adapter (shim)

`src/squidpy/pl/_adata_to_sdata.py` (new, internal, leading underscore in public API).

Single function `_adata_to_sdata(adata) -> SpatialData`. Best-effort. Covers Visium (`adata.uns['spatial']`) and segmentation-table style inputs. For each library:
- Build a `shapes` element from `adata.obsm['spatial']` + `scalefactors[size_key]` so Visium spots arrive as actual circles in data units (resolves the `shape=` question from earlier discussion).
- Build a `table` element wrapping the AnnData.
- Build `images` and `labels` elements from `uns['spatial'][library]['images']` and segmentation if present.
- Set transformations so coordinate systems match per library.

Not polished. Not exposed publicly. Emits one `DeprecationWarning` per call.

### Wrapper translations

For each squidpy kwarg, translate to `spatialdata-plot` call(s):

| Squidpy kwarg | Translation |
|---|---|
| `shape=("circle"\|"square"\|"hex")` | `render_shapes` on the shapes element built by the adapter (or already present in SpatialData input). |
| `shape=None` | `render_points` on a points element derived from `obsm['spatial']`. |
| `vmin` / `vmax` / `vcenter` | Build `Normalize` or `TwoSlopeNorm`, pass `norm=`. |
| `axis_label=[x,y]` | `ax.set_xlabel/set_ylabel` after `show()`. |
| `library_first` | Wrapper owns subplot loop; dispatches `render_*().show(ax=ax_ij)` per cell. |
| `scalebar_dx`, `scalebar_units` | Pass through to `show()` (#648 in sdata-plot). |
| `alt_var` | Rename to `gene_symbols` on render call. |
| `use_raw`, `layer` | Wrapper selects the right `table_layer` or swaps `.X` on a transient SpatialData before the render call. |
| `connectivity_key` | Wrapper composes `render_graph(...).render_points(...).show()`. |
| `seg_outline`, `seg_contourpx` | Translate to `render_labels(contour_px=..., outline_alpha=...)`. Reject `contour_px=1` upstream of the render call (sdata-plot #645). |
| `outline=(c1,c2), outline_width=(w1,w2)` | Two render passes on the same ax. Document as a fallback; consider upstreaming tuple support later. |
| `legend_loc='on data'` | Intercept before `show()`. Render normally, then place text labels at category centroids on the returned ax. |
| `ncols`, `wspace`, `hspace`, multi-library grids, N-gene grids | Wrapper builds the matplotlib grid and dispatches per-cell render chains. |

### Input handling

Function entry:
```
if isinstance(arg, AnnData):
    warnings.warn(..., DeprecationWarning, stacklevel=2)
    sdata = _adata_to_sdata(arg)
elif isinstance(arg, SpatialData):
    sdata = arg
else:
    raise TypeError(...)
```

### Tests

- Parameterize existing `test_spatial_scatter` / `test_spatial_segment` tests over `[adata_input, sdata_input]` for the duration of the window.
- Add a `DeprecationWarning` assertion on the AnnData branch.
- Reference images will shift (sdata-plot rendering does not pixel-match the current matplotlib paths). Follow the reference-image protocol in `tasks/lessons.md` (CI artifacts, not local generation). Refresh baselines once per migrated function in the same PR that lands the migration.

### Risks

- Reference-image churn. Plan for one baseline-refresh commit per top function.
- Visium-HD users at 10^5-10^6 bins: `render_shapes` is per-geometry. Benchmark on a Visium HD fixture before merging Stage 2; if unacceptable, extend `render_points` upstream with a "size in data units" mode rather than densify shapes.
- Non-Visium AnnData-only users (custom readers): the shim must not silently drop their data. Add a clear `NotImplementedError` for unrecognized AnnData layouts pointing at the migration guide.

## Stage 3: remove AnnData input from plots (`v2.0.0`)

- Delete `_adata_to_sdata.py`.
- Function bodies: replace `isinstance(arg, AnnData)` branch with a `TypeError` carrying the migration pointer.
- Drop AnnData-side test parameterizations.
- Signatures unchanged except for the parameter type annotation: `adata: AnnData | SpatialData` -> `sdata: SpatialData` (renaming the kwarg also; accept old name with a `FutureWarning` for one minor if practical, otherwise hard rename and document).

## Stage 4: remove readers (`v2.0.0`)

Same release as Stage 3. Delete `src/squidpy/read/*.py`. Drop reader tests. Migration guide stays.

## Communication plan

Not optional given the surface this touches.

- `v1.9.0` changelog: top-line entry "Readers deprecated, will be removed in v2.0".
- `v1.10.0` changelog: top-line entry "Spatial plots delegate to spatialdata-plot; AnnData input deprecated, will be removed in v2.0".
- Update issue #912 with the timeline at the start of Stage 1.
- Cross-post to the scverse zulip / spatialdata channel at each stage transition.
- Pin a migration guide in `docs/` linked from the package README until v2.0 ships.

## Open questions (resolve before Stage 2)

1. ligrec future: upstream cellphoneDB-style dotplot to sdata-plot, or keep ligrec native and consume `sdata.tables['table']`? Affects whether ligrec's signature also gains SpatialData input in `v1.10`.
2. Statistics plots: in `v2.0`, do they accept SpatialData only, or both? Cleanest is to do them as part of v2.0 in a follow-up PR. Mark separate.
3. Reader replacements that `spatialdata-io` does not yet cover (if any): audit `sq.read` against `spatialdata-io` before Stage 1 to confirm every deprecated reader has a real replacement.
