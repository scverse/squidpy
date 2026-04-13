# Extensibility

## Custom graph builders

The `squidpy.gr.neighbors` module exposes two builder base classes:

- {class}`~squidpy.gr.neighbors.GraphBuilder` is the generic builder pipeline.
  Use it when you want to plug in a custom coordinate type or sparse-matrix backend.
- {class}`~squidpy.gr.neighbors.GraphBuilderCSR` is the CSR-specialized builder used
  by the built-in graph construction strategies. Use it when your builder returns
  {class}`~scipy.sparse.csr_matrix` objects and should reuse Squidpy's CSR-specific
  postprocessors, sparse warning suppression, and multi-library combination.
- Reusable postprocessors such as
  {class}`~squidpy.gr.neighbors.DistanceIntervalPostprocessor`,
  {class}`~squidpy.gr.neighbors.PercentilePostprocessor`, and
  {class}`~squidpy.gr.neighbors.TransformPostprocessor` are also exposed for
  custom builder composition.

### What to override

| Base class | Method / property | Required | Purpose |
|---|---|---|---|
| {class}`~squidpy.gr.neighbors.GraphBuilder` | {meth}`~squidpy.gr.neighbors.GraphBuilder.build_graph` | yes | Construct and return ``(adj, dst)`` using the coordinate and matrix types of your custom backend. |
| {class}`~squidpy.gr.neighbors.GraphBuilder` | {meth}`~squidpy.gr.neighbors.GraphBuilder.postprocessors` | no | Return post-build processing steps for ``(adj, dst)``. You can either override this or pass ``postprocessors=...`` to ``super().__init__()``. |
| {class}`~squidpy.gr.neighbors.GraphBuilder` | {meth}`~squidpy.gr.neighbors.GraphBuilder.combine` | no | Combine per-library results when using ``library_key``. If you do not need ``library_key`` support, leaving this unimplemented is fine. |


The generic builder only defines the pipeline. The CSR-specialized builder adds
multi-library ``library_key`` combination and
{class}`~scipy.sparse.SparseEfficiencyWarning` suppression, while built-in and
custom CSR builders can compose the public reusable postprocessors for
distance-interval pruning, percentile filtering, and adjacency transforms.

Here ``adj`` and ``dst`` are square sparse matrices of shape ``(n_obs, n_obs)``
with matching sparsity structure:

- ``adj`` is the connectivity / adjacency matrix. Non-zero entries mark edges in
  the graph, and built-in builders typically use ``1.0`` for present edges.
- ``dst`` is the distance matrix for those same edges. For generic graphs this is
  usually the Euclidean edge length. For grid builders it may instead encode
  graph-distance semantics such as ring number.
- When subclassing {class}`~squidpy.gr.neighbors.GraphBuilderCSR`, both should be
  returned as {class}`~scipy.sparse.csr_matrix`.
- For CSR-based builders, ``adj`` often behaves like a boolean or indicator
  matrix describing whether an edge is present, even if it is stored with a
  numeric dtype such as ``float32``. ``dst`` stores edge-associated values such
  as distances and will often use a floating-point dtype. The exact dtype choice
  is left to the builder implementation and may depend on performance, memory,
  and numerical accuracy requirements.
- By convention, ``dst`` should have a zero diagonal, and ``adj`` should only
  have a non-zero diagonal when ``set_diag=True``.

### Example: fast radius search with SNN

The built-in {class}`~squidpy.gr.neighbors.RadiusBuilder` uses scikit-learn's
``NearestNeighbors``. The [snnpy](https://github.com/nla-group/snn) library
provides a faster exact fixed-radius search based on PCA-based pruning. The
example below swaps the backend while keeping full compatibility with the rest
of the Squidpy graph pipeline:

```python
import numpy as np
from scipy.sparse import csr_matrix
from snnpy import build_snn_model

from squidpy.gr.neighbors import GraphBuilderCSR


class SNNRadiusBuilder(GraphBuilderCSR):
    """Radius graph using the SNN fixed-radius search backend."""

    def __init__(self, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    def build_graph(self, coords):
        N = coords.shape[0]
        model = build_snn_model(coords, verbose=0)
        indices, dists = model.batch_query_radius(
            coords, self.radius, return_distance=True,
        )

        row = np.repeat(np.arange(N), [len(idx) for idx in indices])
        col = np.concatenate(indices)
        d = np.concatenate(dists).astype(np.float64)

        adj = csr_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(N, N),
        )
        dst = csr_matrix((d, (row, col)), shape=(N, N))

        adj.setdiag(1.0 if self.set_diag else adj.diagonal())
        dst.setdiag(0.0)
        return adj, dst
```

Use it like any other builder:

```python
import squidpy as sq

sq.gr.spatial_neighbors_from_builder(adata, SNNRadiusBuilder(radius=100.0))
```
