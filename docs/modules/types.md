# Types `types`

The parameter bags and result tuples, collected by kind rather than by domain.

## Parameters

All {class}`~typing.TypedDict`s: pass a plain `dict` literal or build one with the class. Every key is optional and falls back to the default shown with it.

```{eval-rst}
.. module:: squidpy.types
.. currentmodule:: squidpy
.. autosummary::
    :toctree: ../api

    types.StalignObsParams
    types.StalignImageParams
    types.StalignVolumeParams
    types.FelzenszwalbParams
    types.WekaParams
    types.ReinhardParams
    types.MacenkoParams
    types.VahadaneParams
```

## Results

```{eval-rst}
.. currentmodule:: squidpy
.. autosummary::
    :toctree: ../api

    types.SpatialNeighborsResult
    types.NhoodEnrichmentResult
```
