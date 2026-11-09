# Spatial niche methods — Squidpy 1.8.4

The generic `sq.gr.calculate_niche` is **deprecated as of 1.8.4**. Use the explicit flavor functions below and pass each required parameter by keyword. All methods expect a precomputed spatial graph except that SpatialLeiden additionally expects latent connectivities.

| Scientific goal                                          | Method                                | Key data requirements                                                          | Default `.obs` output columns      |
| -------------------------------------------------------- | ------------------------------------- | ------------------------------------------------------------------------------ | ---------------------------------- |
| Clusters of similar local cell-type composition          | `sq.gr.calculate_niche_neighborhood`  | `groups` in `.obs`, spatial graph, `resolutions`                               | `"nhood_niche_res=<resolution>"`   |
| Local molecular programs via spatial feature aggregation | `sq.gr.calculate_niche_utag`          | Features in `.X` / layer, spatial graph, `resolutions`                         | `"utag_niche_res=<resolution>"`    |
| Ring-aggregated latent embeddings clustered with a GMM   | `sq.gr.calculate_niche_cellcharter`   | Spatial graph; `use_rep` recommended (e.g., scVI), else PCA fallback           | `"cellcharter_niche"`              |
| Joint spatial + latent network community detection       | `sq.gr.calculate_niche_spatialleiden` | **Both** `"spatial_connectivities"` and `"connectivities"`, plus `resolutions` | `"spatialleiden_res=<resolution>"` |

Important details:

- For the Leiden-based methods, `key_added` is the column prefix, and each requested resolution gets a separate column. For example, neighborhood profiles at `resolutions=0.5` write `nhood_niche_res=0.5`, not `nhood_niche`. CellCharter's default column is the unsuffixed `cellcharter_niche`.

- Neighborhood-profile method: `groups` is the categorical cell-type column; `distance` sets graph-hop reach; `abs_nhood` chooses counts vs proportions; `n_hop_weights` can weight rings; the embedding encodes neighborhood _composition_.
- UTAG: aggregates expression/feature representation over the spatial graph, then PCA + Leiden. Its niche embedding is not merely a cell-type histogram.
- CellCharter-like method: aggregates an embedding by disjoint graph-hop rings and uses a scikit-learn GMM, **not CellCharter's torchgmm**. Do not claim its partitions must match the original CellCharter implementation. Set `use_rep` to a suitable representation when available.
- SpatialLeiden: requires a latent graph _in addition to_ the spatial graph; `layer_ratio` balances layers.
- `table_key` is available for SpatialData inputs. Inspect where each method stores labels and embeddings before overwriting a pre-existing key.
- `library_key` on the old generic niche dispatcher is not a drop-in option on flavor functions. If per-library niches are requested, subset and fit each library deliberately; labels across separately fitted libraries are not directly comparable.
- Prefer the method that matches the **scientific definition** of a niche. Do not default to a method merely because its name resembles the input assay.

Sources:
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.calculate_niche.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.calculate_niche_neighborhood.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.calculate_niche_utag.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.calculate_niche_cellcharter.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.calculate_niche_spatialleiden.html
