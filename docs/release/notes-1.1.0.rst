Squidpy 1.1.0 (2021-07-01)
==========================

Bugfixes
--------
- Fix :class:`squidpy.im.ImageContainer` to work with different scaled.
  `#320 <https://github.com/scverse/squidpy/pull/320>`__
  `@hspitzer <https://github.com/hspitzer>`__

- Fix handling of :attr:`anndata.AnnData.obsm` in :meth:`squidpy.im.ImageContainer.interactive`.
  `#335 <https://github.com/scverse/squidpy/pull/335>`__
  `@michalk8 <https://github.com/michalk8>`__

- Fix Z-dimension in :meth:`squidpy.im.ImageContainer.interactive`.
  `#351 <https://github.com/scverse/squidpy/pull/351>`__
  `@hspitzer <https://github.com/hspitzer>`__

- Fix plotting bug in :func:`squidpy.pl.ripley`.
  `#352 <https://github.com/scverse/squidpy/pull/352>`__
  `@giovp <https://github.com/giovp>`__

- Fix handling of NaNs in :func:`squidpy.gr.ligrec`.
  `#362 <https://github.com/scverse/squidpy/pull/362>`__
  `@michalk8 <https://github.com/michalk8>`__

Features
--------
- Add many new tutorials and examples.

- Add :func:`squidpy.gr.sepal` :cite:`andersson2021`
  `#313 <https://github.com/scverse/squidpy/pull/313>`__
  `@giovp <https://github.com/giovp>`__

- Replace ``squidpy.gr.moran`` with :func:`squidpy.gr.spatial_autocorr`, which implements both Moran's I and
  Geary's C.
  `#317 <https://github.com/scverse/squidpy/pull/317>`__
  `@giovp <https://github.com/giovp>`__

- Add option to compute graph from Delaunay triangulation in :func:`squidpy.gr.spatial_neighbors`.
  `#322 <https://github.com/scverse/squidpy/pull/322>`__
  `@MxMstrmn <https://github.com/MxMstrmn>`__

- Add lazy computation using :mod:`dask` for :mod:`squidpy.im`.
  `#324 <https://github.com/scverse/squidpy/pull/324>`__
  `@michalk8 <https://github.com/michalk8>`__

- Allow Z-dimension shared across all layers in :class:`squidpy.im.ImageContainer`.
  `#329 <https://github.com/scverse/squidpy/pull/329>`__
  `@hspitzer <https://github.com/hspitzer>`__

- Replace ``squidpy.gr.ripley_k`` with :func:`squidpy.gr.ripley`.
  `#331 <https://github.com/scverse/squidpy/pull/331>`__
  `@giovp <https://github.com/giovp>`__

- Generalize graph building in :func:`squidpy.gr.spatial_neighbors`.
  `#340 <https://github.com/scverse/squidpy/pull/340>`__
  `@Koncopd <https://github.com/Koncopd>`__

- Add 3 new example datasets:
  - :func:`squidpy.datasets.merfish`
  - :func:`squidpy.datasets.mibitof`
  - :func:`squidpy.datasets.slideseqv2`
  `#348 <https://github.com/scverse/squidpy/pull/348>`__
  `@giovp <https://github.com/giovp>`__

- Enable additional layer specification in :func:`squidpy.im.calculate_image_features`.
  `#354 <https://github.com/scverse/squidpy/pull/354>`__
  `@hspitzer <https://github.com/hspitzer>`__

- Expose ``canvas_only`` in :meth:`squidpy.pl.Interactive.screenshot`.
  `#363 <https://github.com/scverse/squidpy/pull/363>`__
  `@giovp <https://github.com/giovp>`__

- Various minor improvements to the documentation.
  `#356 <https://github.com/scverse/squidpy/pull/356>`__
  `@michalk8 <https://github.com/michalk8>`__

  `#358 <https://github.com/scverse/squidpy/pull/358>`__
  `@michalk8 <https://github.com/michalk8>`__

  `#359 <https://github.com/scverse/squidpy/pull/359>`__
  `@michalk8 <https://github.com/michalk8>`__
