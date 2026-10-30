# Types `types`

The parameter bags, protocols and result tuples, collected by kind rather than by domain.

## Parameters

All {class}`~typing.TypedDict`s: pass a plain `dict` literal or build one with the class. Every key is optional and falls back to the default shown with it.

```{eval-rst}
.. module:: squidpy.types
.. currentmodule:: squidpy
.. autosummary::
    :toctree: ../api

    types.FelzenszwalbParams
    types.WekaParams
    types.ReinhardParams
    types.MacenkoParams
    types.VahadaneParams
```

## Protocols

What a ``clusterer`` argument has to implement.

```{eval-rst}
.. currentmodule:: squidpy
.. autosummary::
    :toctree: ../api

    types.Clusterer
    types.KClusterer
```

## Results

```{eval-rst}
.. currentmodule:: squidpy
.. autosummary::
    :toctree: ../api

    types.SpatialNeighborsResult
    types.NhoodEnrichmentResult
```
