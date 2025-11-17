# Squidpy 1.1.0 (2021-07-01)

## Bugfixes

- Fix {class}`squidpy.im.ImageContainer` to work with different scaled.
  [#320](https://github.com/scverse/squidpy/pull/320)
  [@hspitzer](https://github.com/hspitzer)

- Fix handling of :attr:`anndata.AnnData.obsm` in `squidpy.im.ImageContainer.interactive`.
  [#335](https://github.com/scverse/squidpy/pull/335)
  [@michalk8](https://github.com/michalk8)

- Fix Z-dimension in `squidpy.im.ImageContainer.interactive`.
  [#351](https://github.com/scverse/squidpy/pull/351)
  [@hspitzer](https://github.com/hspitzer)

- Fix plotting bug in {func}`squidpy.pl.ripley`.
  [#352](https://github.com/scverse/squidpy/pull/352)
  [@giovp](https://github.com/giovp)

- Fix handling of NaNs in {func}`squidpy.gr.ligrec`.
  [#362](https://github.com/scverse/squidpy/pull/362)
  [@michalk8](https://github.com/michalk8)

## Features

- Add many new tutorials and examples.

- Add {func}`squidpy.gr.sepal` :cite:`andersson2021`
  [#313](https://github.com/scverse/squidpy/pull/313)
  [@giovp](https://github.com/giovp)

- Replace ``squidpy.gr.moran`` with {func}`squidpy.gr.spatial_autocorr`, which implements both Moran's I and
  Geary's C.
  [#317](https://github.com/scverse/squidpy/pull/317)
  [@giovp](https://github.com/giovp)

- Add option to compute graph from Delaunay triangulation in {func}`squidpy.gr.spatial_neighbors`.
  [#322](https://github.com/scverse/squidpy/pull/322)
  [@MxMstrmn](https://github.com/MxMstrmn)

- Add lazy computation using :doc:`dask:index` for {mod}`squidpy.im`.
  [#324](https://github.com/scverse/squidpy/pull/324)
  [@michalk8](https://github.com/michalk8)

- Allow Z-dimension shared across all layers in {class}`squidpy.im.ImageContainer`.
  [#329](https://github.com/scverse/squidpy/pull/329)
  [@hspitzer](https://github.com/hspitzer)

- Replace ``squidpy.gr.ripley_k`` with {func}`squidpy.gr.ripley`.
  [#331](https://github.com/scverse/squidpy/pull/331)
  [@giovp](https://github.com/giovp)

- Generalize graph building in {func}`squidpy.gr.spatial_neighbors`.
  [#340](https://github.com/scverse/squidpy/pull/340)
  [@Koncopd](https://github.com/Koncopd)

- Add 3 new example datasets:
  - {func}`squidpy.datasets.merfish`
  - {func}`squidpy.datasets.mibitof`
  - {func}`squidpy.datasets.slideseqv2`
  [#348](https://github.com/scverse/squidpy/pull/348)
  [@giovp](https://github.com/giovp)

- Enable additional layer specification in {func}`squidpy.im.calculate_image_features`.
  [#354](https://github.com/scverse/squidpy/pull/354)
  [@hspitzer](https://github.com/hspitzer)

- Expose ``canvas_only`` in `squidpy.pl.Interactive.screenshot`.
  [#363](https://github.com/scverse/squidpy/pull/363)
  [@giovp](https://github.com/giovp)

- Various minor improvements to the documentation.
  [#356](https://github.com/scverse/squidpy/pull/356)
  [@michalk8](https://github.com/michalk8)

  [#358](https://github.com/scverse/squidpy/pull/358)
  [@michalk8](https://github.com/michalk8)

  [#359](https://github.com/scverse/squidpy/pull/359)
  [@michalk8](https://github.com/michalk8)
