Squidpy 1.1.0 (2021-07-01)
==========================

Bugfixes
--------
- Fix :class:`squidpy.im.ImageContainer` to work with different scaled
  `#320 <https://github.com/scverse/squidpy/pull/320>`_.
- Fix handling of :attr:`anndata.AnnData.obsm` in :meth:`squidpy.im.ImageContainer.interactive`
  `#335 <https://github.com/scverse/squidpy/pull/335>`_.
- Fix Z-dimension in :meth:`squidpy.im.ImageContainer.interactive`
  `#351 <https://github.com/scverse/squidpy/pull/351>`_.
- Fix plotting bug in :func:`squidpy.pl.ripley` `#352 <https://github.com/scverse/squidpy/pull/352>`_.
- Fix handling of NaNs in :func:`squidpy.gr.ligrec` `#362 <https://github.com/scverse/squidpy/pull/362>`_.

Features
--------
- Add many new tutorials and examples
- Add :func:`squidpy.gr.sepal` :cite:`andersson2021` `#313 <https://github.com/scverse/squidpy/pull/313>`_.
- Replace ``squidpy.gr.moran`` with :func:`squidpy.gr.spatial_autocorr`, which implements both Moran's I and
  Geary's C `#317 <https://github.com/scverse/squidpy/pull/317>`_.
- Add option to compute graph from Delaunay triangulation in :func:`squidpy.gr.spatial_neighbors`
  `#322 <https://github.com/scverse/squidpy/pull/322>`_.
- Add lazy computation using :mod:`dask` for :mod:`squidpy.im` `#324 <https://github.com/scverse/squidpy/pull/324>`_.
- Allow Z-dimension shared across all layers in :class:`squidpy.im.ImageContainer`
  `#329 <https://github.com/scverse/squidpy/pull/329>`_.
- Replace ``squidpy.gr.ripley_k`` with :func:`squidpy.gr.ripley` `#331 <https://github.com/scverse/squidpy/pull/331>`_
- Generalize graph building in :func:`squidpy.gr.spatial_neighbors`
  `#340 <https://github.com/scverse/squidpy/pull/340>`_.
- Add 3 new example datasets `#348 <https://github.com/scverse/squidpy/pull/348>`_:

  - :func:`squidpy.datasets.merfish`
  - :func:`squidpy.datasets.mibitof`
  - :func:`squidpy.datasets.slideseqv2`

- Enable additional layer specification in :func:`squidpy.im.calculate_image_features`
  `#354 <https://github.com/scverse/squidpy/pull/354>`_.
- Expose ``canvas_only`` in :meth:`squidpy.pl.Interactive.screenshot`
  `#363 <https://github.com/scverse/squidpy/pull/363>`_.
- Various minor improvements to the documentation `#356 <https://github.com/scverse/squidpy/pull/356>`_,
  `#358 <https://github.com/scverse/squidpy/pull/358>`_, `#359 <https://github.com/scverse/squidpy/pull/359>`_.
