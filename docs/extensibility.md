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

        Adj = csr_matrix(
            (np.ones(len(row), dtype=np.float32), (row, col)),
            shape=(N, N),
        )
        Dst = csr_matrix((d, (row, col)), shape=(N, N))

        Adj.setdiag(1.0 if self.set_diag else Adj.diagonal())
        Dst.setdiag(0.0)
        return Adj, Dst
```

Use it like any other builder:

```python
import squidpy as sq

sq.gr.spatial_neighbors(adata, builder=SNNRadiusBuilder(radius=100.0))
```
