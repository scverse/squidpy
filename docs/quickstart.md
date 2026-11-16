# Quickstart

Squidpy works on an {class}`~anndata.AnnData` object with spatial coordinates in `adata.obsm["spatial"]`: {mod}`squidpy.gr` builds a spatial graph and computes statistics on it, and {mod}`squidpy.pl` plots the results.
The example below builds the spatial graph of a seqFISH mouse embryo, tests which cell types neighbor each other more than by chance, and ranks genes by spatial autocorrelation.

```python
import squidpy as sq

adata = sq.datasets.seqfish()

sq.gr.spatial_neighbors_delaunay(adata)
sq.gr.nhood_enrichment(adata, cluster_key="celltype_mapped_refined")
sq.gr.co_occurrence(adata, cluster_key="celltype_mapped_refined")
sq.gr.spatial_autocorr(adata, mode="moran")

sq.pl.spatial_scatter(adata, color="celltype_mapped_refined", shape=None)
sq.pl.nhood_enrichment(adata, cluster_key="celltype_mapped_refined")
```

The genes ranked by Moran’s I end up in `adata.uns["moranI"]`.
The {doc}`notebooks/tutorials/index` apply these steps to other technologies, and {doc}`api` lists every function.
