# Graphs and spatial statistics — Squidpy 1.8.4

## Choose the graph scientifically

| Geometry / question                                | Public constructor                     | What to check                                                                      |
| -------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------------------------------- |
| Continuous spatial coordinates, fixed neighbors    | `sq.gr.spatial_neighbors_knn`          | `n_neighs` (default 6), potential long edges in sparse areas                       |
| Meaningful distance threshold                      | `sq.gr.spatial_neighbors_radius`       | `radius` in coordinate units; tuple defines an annulus                             |
| Geometry-adaptive triangulation                    | `sq.gr.spatial_neighbors_delaunay`     | Long edges at boundaries; optional edge pruning                                    |
| Approximately regular lattice (e.g., Visium-style) | `sq.gr.spatial_neighbors_grid`         | `n_neighs`, `n_rings`; do not apply lattice logic to irregular Xenium point clouds |
| Custom graph builders                              | `sq.gr.spatial_neighbors_from_builder` | Builder and postprocessor contracts                                                |

The generic `sq.gr.spatial_neighbors` is deprecated in 1.8.4. Pass keyword arguments, including `table_key` and `elements_to_coordinate_systems` for SpatialData. With `copy=False` and default `key_added="spatial"`, outputs are in `adata.obsp["spatial_connectivities"]`, `adata.obsp["spatial_distances"]`, and `adata.uns["spatial_neighbors"]`. For a custom `key_added`, the metadata key is `f"{key_added}_neighbors"`. Some 1.8.4 docstrings incorrectly say `adata.uns["spatial"]`; do not rename or duplicate the runtime's metadata to match those docstrings. With `copy=True`, a `SpatialNeighborsResult` is returned.

Do not mix `scanpy.pp.neighbors` (latent-expression neighborhood graph) with a Squidpy spatial graph. They answer different questions and normally use different `.obsp` keys.

## Downstream choice

- `sq.gr.nhood_enrichment`: permutation-based over-/under-representation of categorical neighbors; pass `cluster_key`, use `rng=`; review rare classes and undefined z-scores instead of blindly filling NaNs.
- `sq.gr.interaction_matrix`: observed counts of class-to-class spatial adjacency; not a permutation test.
- `sq.gr.co_occurrence`: probability of observing classes at a distance from a reference class; interpret distance scales carefully.
- `sq.gr.centrality_scores`: network-derived centrality per annotation group.
- `sq.gr.spatial_autocorr`: global Moran's I (`mode="moran"`) or Geary's C (`mode="geary"`); requires spatial connectivities, and `rng=` if permutation testing. Geary's normality variance was corrected in 1.8.2.
- `sq.gr.ripley`: point-pattern F/G/L statistics, not synonymous with graph enrichment.
- `sq.gr.ligrec`: expression-based ligand–receptor analysis; does not by itself prove physical cell-to-cell communication.
- `sq.gr.sepal`: spatial expression pattern modeling/ranking.
- `sq.gr.mask_graph`: restrict an existing graph, rather than automatically constructing a new spatial neighborhood definition.

## 1.8.4 reproducibility

In permutation-based routines updated in 1.8.4, `seed` and `random_state` were renamed to `rng` (SPEC 7). Implementations spawn independent random generators; changing `n_jobs` should not change the updated permutation outcome. Numbers with an identical seed can differ from older Squidpy releases. Report `rng`, graph settings, and `n_perms` where relevant.

## Checks

- Verify `.obsp` shape, edge counts and unexpected isolated observations.
- In multi-library datasets, use `library_key` where the graph should be disjoint by sample.
- Ensure coordinate scale is physically interpretable before using a radius.
- For tests, compare scientific properties rather than hard-coding older-version permutation results.

Sources:
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors_knn.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors_radius.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors_grid.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.nhood_enrichment.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_autocorr.html
