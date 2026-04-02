# Extensibility

## Custom graph builders

The `squidpy.gr.neighbors` module exposes a {class}`~squidpy.gr.neighbors.GraphBuilder`
base class that all built-in graph construction strategies inherit from.
You can implement your own strategy by subclassing it.

### What to override

| Method / property | Required | Purpose |
|---|---|---|
| {attr}`~squidpy.gr.neighbors.GraphBuilder.coord_type` | yes | Return the {class}`~squidpy._constants._constants.CoordType` this builder supports. |
| {meth}`~squidpy.gr.neighbors.GraphBuilder.build_graph` | yes | Construct and return ``(Adj, Dst)`` as {class}`~scipy.sparse.csr_matrix` pair. |
| {meth}`~squidpy.gr.neighbors.GraphBuilder.apply_filters` | no | Post-processing on the raw ``Adj``/``Dst`` (e.g. radius-interval pruning). Called before percentile filtering and transform. |

The base class handles percentile filtering, adjacency transforms, and
{class}`~scipy.sparse.SparseEfficiencyWarning` suppression automatically.

### Example: mutual KNN

The built-in {class}`~squidpy.gr.neighbors.KNNBuilder` keeps all k-nearest
edges regardless of reciprocity. A mutual KNN graph retains an edge only when
both endpoints consider each other a neighbor, producing a sparser and more
conservative graph:

```python
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from squidpy._constants._constants import CoordType
from squidpy.gr.neighbors import GraphBuilder


class MutualKNNBuilder(GraphBuilder):
    """KNN graph keeping only mutual (reciprocal) edges."""

    def __init__(self, n_neighs: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.n_neighs = n_neighs

    @property
    def coord_type(self) -> CoordType:
        return CoordType.GENERIC

    def build_graph(self, coords):
        N = coords.shape[0]
        tree = NearestNeighbors(n_neighbors=self.n_neighs, metric="euclidean")
        tree.fit(coords)
        dists, indices = tree.kneighbors()

        row = np.repeat(np.arange(N), self.n_neighs)
        col = indices.reshape(-1)
        d = dists.reshape(-1)

        Adj = csr_matrix((np.ones_like(d, dtype=np.float32), (row, col)), shape=(N, N))
        Dst = csr_matrix((d, (row, col)), shape=(N, N))

        # keep only mutual edges
        mutual = Adj.multiply(Adj.T)
        Adj = mutual.tocsr()
        Dst = Dst.multiply(mutual).tocsr()

        Adj.setdiag(1.0 if self.set_diag else Adj.diagonal())
        Dst.setdiag(0.0)
        return Adj, Dst
```

Use it like any other builder:

```python
import squidpy as sq

sq.gr.spatial_neighbors(adata, builder=MutualKNNBuilder(n_neighs=6))
```
