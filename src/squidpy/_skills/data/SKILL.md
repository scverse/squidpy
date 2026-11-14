---

name: squidpy
description: "Use for Python spatial-omics analysis with Squidpy 1.8.4: building spatial neighbor graphs, cell neighborhood enrichment, spatial autocorrelation, niche discovery (neighborhood, UTAG, CellCharter, SpatialLeiden), or SpatialData-native tissue detection, image QC, tiling, H&E stain normalization, cell image features, segmentation QC, and stitching. Use when debugging Squidpy calls or migrating old Squidpy API names. Do not trigger for general Scanpy preprocessing or plotting with spatialdata-plot alone."
---

# Squidpy 1.8.4

This skill supports **exactly `squidpy==1.8.4`**. Older code may be _migrated_ to 1.8.4, but older releases are not supported. Import as `import squidpy as sq`.

## Before coding

1. Check `sq.__version__`. For tasks needing this skill, require `1.8.4` and do not silently substitute a different version.
2. Check whether the input is `anndata.AnnData` or `spatialdata.SpatialData`. Inspect existing `.obsm`, `.obsp`, `.obs` keys, or `sdata.images`, `sdata.labels`, `sdata.shapes`, `sdata.tables` before choosing element names.
3. For SpatialData, inspect table annotations, linked region/instance IDs, and coordinate systems; never infer the image, label, or table key from examples.
4. Prefer the documented 1.8.4 public API and **keyword arguments**. Consult the relevant reference below before generating unfamiliar experimental calls.
5. Identify writes before executing: several functions mutate AnnData or add/overwrite SpatialData elements; use deliberate output keys.

## Core mechanics you should not guess

- **Graphs:** `sq.gr.spatial_neighbors` is deprecated in 1.8.4. Use `sq.gr.spatial_neighbors_knn`, `_radius`, `_delaunay`, `_grid`, or `_from_builder` by the scientific neighborhood definition. With `key_added="spatial"` and `copy=False`, graph output lives in `adata.obsp["spatial_connectivities"]`, `adata.obsp["spatial_distances"]` and `adata.uns["spatial_neighbors"]`. For a custom `key_added`, metadata goes in `adata.uns[f"{key_added}_neighbors"]`. Some 1.8.4 docstrings incorrectly describe this metadata key; use the runtime contract above. An `AnnData` or a SpatialData table can be modified in place.
- **SpatialData graphs:** pass `table_key` and `elements_to_coordinate_systems` mapping region elements (Shapes/Labels) to coordinate systems. The table must annotate the mapped regions. Verify centroid rows against table region/instance IDs: the 1.8.4 dispatcher can retain shape order when table IDs are shuffled; see the SpatialData reference for explicit alignment. A graph on one sample/library must not connect observations in another; use `library_key` where appropriate.
- **Niches:** `sq.gr.calculate_niche` is deprecated in 1.8.4. Pick an explicit flavor. Niches require an existing spatial graph; SpatialLeiden also requires latent connectivities.
- **Randomness:** use `rng=`, not historical `seed=` or `random_state=`, where the 1.8.4 API exposes it. Permutation results with the same seed may differ from older releases; parallelism should not change the result for the updated permutation implementations.
- **Experimental image workflows:** use `sq.experimental.im.*` (tissue detection, image QC, tiling, stains, features) and `sq.experimental.tl.*` (segmentation tiling QC, stitch grouping, stitched labels). They operate on **SpatialData**, not legacy `ImageContainer` examples. Inspect their current contracts before composing pipelines.
- **Stain workflow:** `fit_stain_reference(...)` **returns a `StainFit` without writing**; then call `fit.transform(...)` for normalized image output or `fit.decompose(...)` for H/E maps. Fitting requires an existing tissue mask; `fit_stain_reference` does not auto-create it.
- **Segmentation stitching:** `calculate_tiling_qc` → `assign_stitch_groups` → `make_stitched_labels`. Group assignment does **not** modify the labels; final materialization creates a new labels element. Validate potential false merges before materializing.
- **Plotting boundaries:** Squidpy includes spatial-statistics plots, but for composed rendering of SpatialData images, labels, and shapes use `spatialdata-plot`; interactive annotation/viewing belongs to `napari-spatialdata`.

## Task routing: read only the relevant reference

| User goal                                                                                                   | Read first                                                                                 |
| ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| SpatialData table/region mapping, coordinate systems, alignment, element changes                            | [references/spatialdata.md](references/spatialdata.md)                                     |
| Spatial graph construction, graph semantics, enrichment, Moran/Geary, interactions, Ripley, ligand–receptor | [references/graphs-and-statistics.md](references/graphs-and-statistics.md)                 |
| Discover or compare spatial niches, UTAG, CellCharter, neighborhood profiles, SpatialLeiden                 | [references/niches.md](references/niches.md)                                               |
| Tissue mask, image QC, grid tiles, Visium spot-based tiles                                                  | [references/image-qc-and-tiling.md](references/image-qc-and-tiling.md)                     |
| Macenko, Vahadane, Reinhard, stain matrices, normalize/decompose H&E                                        | [references/stain-normalization.md](references/stain-normalization.md)                     |
| Extract single-cell shape, intensity, texture, histogram features from labels/shapes                        | [references/feature-extraction.md](references/feature-extraction.md)                       |
| Detect tile-boundary segmentation errors or stitch cell fragments                                           | [references/segmentation-qc-and-stitching.md](references/segmentation-qc-and-stitching.md) |
| Rename/deprecation error, older notebook, moved function, `rng` migration                                   | [references/api-migrations.md](references/api-migrations.md)                               |

Read more than one reference only if the actual task crosses their boundaries. For example, stitching often requires the SpatialData reference; staining and feature extraction may require image alignment information.

## Documentation-checked patterns (runtime testing pending)

```python
import squidpy as sq

# AnnData with physical or image-space coordinates in adata.obsm["spatial"]
sq.gr.spatial_neighbors_knn(adata, n_neighs=6)
assert "spatial_connectivities" in adata.obsp

# On an existing graph, compare spatial adjacency of cell classes.
sq.gr.nhood_enrichment(adata, cluster_key="cell_type", rng=42)
```

```python
import squidpy as sq

# On SpatialData: existing image key "histology" and label key "tissue_mask".
fit = sq.experimental.im.fit_stain_reference(
    sdata, image_key="histology", tissue_mask_key="tissue_mask", method="macenko"
)
fit.transform(sdata, image_key="histology", tissue_mask_key="tissue_mask")
# A new sdata.images["histology_normalized"] is written by default.
```

The names in these snippets are **illustrative**, not assumed keys in an actual user dataset. Examples have been checked against the documented 1.8.4 signatures; they still require runtime testing with real input fixtures.

## Before reporting results

- Check that graph dimensions equal observation counts; inspect isolated cells, self-edges, coordinate units, library separation, and the chosen graph definition.
- For SpatialData, verify coordinate systems and element/table relationships before and after the operation.
- Confirm expected output keys, shapes, dimensions, cell IDs and metadata. Never assume `inplace=False` and `copy=True` have identical effects across functions.
- Inspect derived images/labels visually at a small representative region; do not compute a whole-slide image solely for visual QC.
- Report statistical definitions (graph type, radius/k, permutation count, `rng`), and scientific limitations. Expression-based ligand–receptor scores are not proof of local signaling.

**Authoritative reference:** https://squidpy.readthedocs.io/en/stable/api.html (check the page version; one feature-extraction subpage was indexed as 1.8.3 during preparation). Experimental functionality is explicitly subject to API change; do not generalize these instructions to another Squidpy release.
