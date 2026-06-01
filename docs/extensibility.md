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

### Example: approximate kNN search with pynndescent

The built-in {class}`~squidpy.gr.neighbors.KNNBuilder` uses scikit-learn's
``NearestNeighbors``. The [pynndescent](https://github.com/lmcinnes/pynndescent)
library provides an approximate nearest-neighbor search backend that is often
faster on larger datasets. The example below swaps the backend while keeping
the Squidpy graph pipeline contract intact.

To run the example, install the optional backend into the same environment as
Squidpy:

```bash
python -m pip install pynndescent
```

```python
# Following code is illustrative and requires ``pynndescent`` to be installed.
import numpy as np
from scipy.sparse import csr_matrix
from pynndescent import NNDescent

from squidpy.gr.neighbors import GraphBuilderCSR


class PynndescentKNNBuilder(GraphBuilderCSR):
  """KNN graph using the pynndescent approximate nearest-neighbor backend."""

  def __init__(self, n_neighs: int = 6, **kwargs):
    super().__init__(**kwargs)
    self.n_neighs = n_neighs

  def uns_params(self):
    return {
      "n_neighbors": self.n_neighs,
      "set_diag": self.set_diag,
    }

  def build_graph(self, coords):
    n_obs = coords.shape[0]
    model = NNDescent(coords, metric="euclidean")
    indices, dists = model.query(coords, k=self.n_neighs)

    row_indices = np.repeat(np.arange(n_obs), self.n_neighs)
    col_indices = indices.reshape(-1)
    dists = dists.reshape(-1).astype(np.float64)

    adj = csr_matrix(
      (np.ones_like(row_indices, dtype=np.float32), (row_indices, col_indices)),
      shape=(n_obs, n_obs),
    )
    dst = csr_matrix((dists, (row_indices, col_indices)), shape=(n_obs, n_obs))

    adj.setdiag(1.0 if self.set_diag else adj.diagonal())
    dst.setdiag(0.0)
    return adj, dst
```

Use it like any other builder:

```python
import squidpy as sq

sq.gr.spatial_neighbors_from_builder(adata, PynndescentKNNBuilder(n_neighs=6))
```
