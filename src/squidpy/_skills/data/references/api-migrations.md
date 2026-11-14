# Migrating old code TO Squidpy 1.8.4

Only **1.8.4** is supported by this skill. Legacy names are included solely to recognize and rewrite older examples; do not recommend them in new code. Preserve the scientific meaning when migrating: a grid, kNN, radius, and Delaunay graph are not interchangeable.

| Old / deprecated usage                                             | Squidpy 1.8.4 usage                               | Caveat                                                                                  |
| ------------------------------------------------------------------ | ------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `sq.gr.spatial_neighbors(..., coord_type="generic", n_neighs=...)` | `sq.gr.spatial_neighbors_knn(..., n_neighs=...)`  | Confirm original graph construction intent; `coord_type` is not a direct new kwarg      |
| `sq.gr.spatial_neighbors(..., radius=...)`                         | `sq.gr.spatial_neighbors_radius(..., radius=...)` | Preserve distance units and radius interval                                             |
| `sq.gr.spatial_neighbors(..., delaunay=True)`                      | `sq.gr.spatial_neighbors_delaunay(...)`           | Check edge pruning options                                                              |
| `sq.gr.spatial_neighbors(..., coord_type="grid", n_rings=...)`     | `sq.gr.spatial_neighbors_grid(..., n_rings=...)`  | Valid only for lattice-like coordinates                                                 |
| `sq.gr.calculate_niche(..., flavor="neighborhood")`                | `sq.gr.calculate_niche_neighborhood(...)`         | Explicit `groups`, `resolutions`; review altered argument names                         |
| `sq.gr.calculate_niche(..., flavor="utag")`                        | `sq.gr.calculate_niche_utag(...)`                 | Explicit `resolutions`, output naming                                                   |
| `sq.gr.calculate_niche(..., flavor="cellcharter")`                 | `sq.gr.calculate_niche_cellcharter(...)`          | GMM and embedding controls; do not assume bitwise-equivalent results                    |
| `sq.gr.calculate_niche(..., flavor="spatialleiden")`               | `sq.gr.calculate_niche_spatialleiden(...)`        | Requires spatial and latent graphs                                                      |
| `sq.experimental.im.make_stitched_labels(...)` in older examples   | `sq.experimental.tl.make_stitched_labels(...)`    | Target stage 3 only after QC + group assignment                                         |
| `seed=` / `random_state=` in updated stochastic analysis methods   | `rng=`                                            | 1.8.4 may produce different values for an identical integer seed vs historical releases |

Other pitfalls:

- The old `ImageContainer` APIs (`sq.im.calculate_image_features`, `sq.im.process`, `sq.im.segment`) still exist in the published namespace but are **not** drop-in aliases for SpatialData-native `sq.experimental.im.*` methods. Do not blindly rename them without adapting input data, feature arguments, and output semantics.
- `sq.gr.calculate_niche` had a `library_key` argument with per-library behavior; flavor-specific replacements do not provide an identical one-argument migration. Perform per-library subsetting intentionally when needed.
- Some new APIs made formerly positional options keyword-only. Inspect 1.8.4 signatures rather than copying old positional calls.
- The Squidpy napari plugin was moved out; use `napari-spatialdata` for interactive viewing and `spatialdata-plot` for layered static rendering.

Sources:
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.calculate_niche.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.nhood_enrichment.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.tl.make_stitched_labels.html
