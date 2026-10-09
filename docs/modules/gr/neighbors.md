# Neighbors `neighbors`

See the {doc}`extensibility guide </extensibility>` for how to implement a custom graph
builder. ``GraphMatrixT`` is the type variable those interfaces are generic over; it is
documented here rather than beside the ``gr`` functions, where a bare type variable read as
public API.

```{eval-rst}
.. module:: squidpy.gr.neighbors
.. currentmodule:: squidpy.gr
.. autosummary::
    :toctree: ../../api

    neighbors.GraphBuilder
    neighbors.GraphBuilderCSR
    neighbors.GraphPostprocessor
    neighbors.DistanceIntervalPostprocessor
    neighbors.PercentilePostprocessor
    neighbors.TransformPostprocessor
    neighbors.KNNBuilder
    neighbors.RadiusBuilder
    neighbors.DelaunayBuilder
    neighbors.GridBuilder
    neighbors.GraphMatrixT
```
