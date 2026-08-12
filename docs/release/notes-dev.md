# Squidpy dev (the-future)

## Features

- Add an experimental, opt-in `spatialdata-plot` delegation backend for {func}`squidpy.pl.spatial_scatter` and {func}`squidpy.pl.spatial_segment`, enabled with the `SQUIDPY_USE_SDATAPLOT=1` environment variable. It accepts native {class}`spatialdata.SpatialData` input (in addition to AnnData) and renders through `spatialdata-plot` instead of the legacy matplotlib path. On SpatialData input, `shapes_layer` / `labels_layer` / `points_layer` / `image_layer` / `table` select the element to render when a coordinate system holds more than one candidate.
- {func}`squidpy.experimental.im.calculate_image_features` now featurizes tiles on a shared dask engine: `n_jobs > 1` runs worker processes via a `dask.distributed.LocalCluster` (or an active `Client`), and per-tile BLAS/OpenMP threads are pinned to avoid oversubscription. This also speeds up the serial path. {func}`squidpy.experimental.tl.calculate_tiling_qc` shares the same engine. Adds `distributed` and `threadpoolctl` as dependencies.
- Fix {func}`squidpy.tl.var_by_distance` behaviour when providing {mod}`numpy` arrays of coordinates as anchor point.
- Update :attr:`squidpy.pl.var_by_distance` to show multiple variables on same plot.
  [@LLehner](https://github.com/LLehner)
  [#929](https://github.com/scverse/squidpy/pull/929)

## Deprecations

- Passing an {class}`anndata.AnnData` to the spatial plotting functions now emits a {class}`DeprecationWarning` under the delegation backend; pass a {class}`spatialdata.SpatialData` instead. AnnData input is slated for removal in squidpy v2.0.
