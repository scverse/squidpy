# Extensibility

## Custom graph builders

The `squidpy.gr.neighbors` module exposes a {class}`~squidpy.gr.neighbors.GraphBuilder`
base class that all built-in graph construction strategies inherit from.
You can implement your own strategy by subclassing it.

### What to override

| Method / property | Required | Purpose |
|---|---|---|
| {attr}`~squidpy.gr.neighbors.GraphBuilder.coord_type` | yes | Return the {class}`~squidpy._constants._constants.CoordType` this builder supports. |
| {meth}`~squidpy.gr.neighbors.GraphBuilder.build_graph` | yes | Construct and return ``(adj, dst)`` as {class}`~scipy.sparse.csr_matrix` pair. |
| {meth}`~squidpy.gr.neighbors.GraphBuilder.apply_filters` | no | Post-processing on the raw ``adj``/``dst`` (e.g. radius-interval pruning). Called before percentile filtering and transform. |

The base class handles percentile filtering, adjacency transforms, and
{class}`~scipy.sparse.SparseEfficiencyWarning` suppression automatically.

Here ``adj`` and ``dst`` are square sparse matrices of shape ``(n_obs, n_obs)``
with matching sparsity structure:

- ``adj`` is the connectivity / adjacency matrix. Non-zero entries mark edges in
  the graph, and built-in builders typically use ``1.0`` for present edges.
- ``dst`` is the distance matrix for those same edges. For generic graphs this is
  usually the Euclidean edge length. For grid builders it may instead encode
  graph-distance semantics such as ring number.
- Both should be returned as {class}`~scipy.sparse.csr_matrix`.
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

from squidpy._constants._constants import CoordType
from squidpy.gr.neighbors import GraphBuilder


class SNNRadiusBuilder(GraphBuilder):
    """Radius graph using the SNN fixed-radius search backend."""

    def __init__(self, radius: float, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius

    @property
    def coord_type(self) -> CoordType:
        return CoordType.GENERIC

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

sq.gr.spatial_neighbors(adata, builder=SNNRadiusBuilder(radius=100.0))
```
