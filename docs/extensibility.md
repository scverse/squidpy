# Extensibility

## Custom graph builders

The `squidpy.gr.neighbors` module exposes a {class}`~squidpy.gr.neighbors.GraphBuilder`
base class that all built-in graph construction strategies inherit from.
You can implement your own strategy by subclassing it.

### Minimal example

```python
import numpy as np
from scipy.sparse import csr_matrix
from squidpy.gr.neighbors import GraphBuilder
from squidpy._constants._constants import CoordType


class MyBuilder(GraphBuilder):
    """Example: connect every point to its single nearest neighbor."""

    @property
    def coord_type(self) -> CoordType:
        return CoordType.GENERIC

    def _build_graph(self, coords):
        from sklearn.neighbors import NearestNeighbors

        N = coords.shape[0]
        tree = NearestNeighbors(n_neighbors=2, metric="euclidean")
        tree.fit(coords)
        dists, indices = tree.kneighbors()

        # drop self-neighbor (index 0), keep only the closest other point
        row = np.arange(N)
        col = indices[:, 1]
        d = dists[:, 1]

        Adj = csr_matrix((np.ones(N, dtype=np.float32), (row, col)), shape=(N, N))
        Dst = csr_matrix((d, (row, col)), shape=(N, N))
        return Adj, Dst
```

### Using a custom builder

Pass an instance to any of the spatial-neighbor functions via the ``builder``
argument:

```python
import squidpy as sq

sq.gr.spatial_neighbors_knn(adata)           # built-in KNN mode
sq.gr.spatial_neighbors(adata, builder=MyBuilder())  # custom builder
```

### What to override

| Method / property | Required | Purpose |
|---|---|---|
| {attr}`~squidpy.gr.neighbors.GraphBuilder.coord_type` | yes | Return the {class}`~squidpy._constants._constants.CoordType` this builder supports. |
| {meth}`~squidpy.gr.neighbors.GraphBuilder._build_graph` | yes | Construct and return ``(Adj, Dst)`` as {class}`~scipy.sparse.csr_matrix` pair. |
| {meth}`~squidpy.gr.neighbors.GraphBuilder._apply_filters` | no | Post-processing on the raw ``Adj``/``Dst`` (e.g. radius-interval pruning). Called before percentile filtering and transform. |

The base class handles percentile filtering ({meth}`~squidpy.gr.neighbors.GraphBuilder._apply_percentile`),
adjacency transforms ({meth}`~squidpy.gr.neighbors.GraphBuilder._apply_transform`), and
{class}`~scipy.sparse.SparseEfficiencyWarning` suppression automatically.
