.. role:: small

1.1.0 :small:`SOON`
~~~~~~~~~~~~~~~~~~~
This release includes:

.. rubric:: Additions

- Add :func:`squidpy.gr.sepal` :cite:`andersson2021` `PR 313 <https://github.com/theislab/squidpy/pull/313>`_.
- Replace ``squidpy.gr.moran`` with :func:`squidpy.gr.spatial_autocorr`, which implements both Moran's I and
  Geary's C `PR 317 <https://github.com/theislab/squidpy/pull/317>`_.
- Add option to compute graph from Delauney triangulation in :func:`squidpy.gr.spatial_neighbors`
  `PR 322 <https://github.com/theislab/squidpy/pull/322>`_.
- Add lazy computation using :mod:`dask` for :mod:`squidpy.im` `PR 324 <https://github.com/theislab/squidpy/pull/324>`_.
- Allow Z-dimension shared across all layers in :class:`squidpy.im.ImageContainer`
  `PR 329 <https://github.com/theislab/squidpy/pull/329>`_.
- Replace ``squidpy.gr.ripley_k`` with :func:`squidpy.gr.ripley` `PR 331 <https://github.com/theislab/squidpy/pull/331>`_
- Generalize graph building in :func:`squidpy.gr.spatial_neighbors`
  `PR 340 <https://github.com/theislab/squidpy/pull/340>`_.
- Add 3 new example datasets `PR 348 <https://github.com/theislab/squidpy/pull/348>`_:

  - :func:`squidpy.datasets.merfish`
  - :func:`squidpy.datasets.mibitof`
  - :func:`squidpy.datasets.slideseqv2`

- Enable additional layer specification in :func:`squidpy.im.calculate_image_features`
  `PR 354 <https://github.com/theislab/squidpy/pull/354>`_.
- Expose ``canvas_only`` in :meth:`squidpy.im.Interactive.screenshot`
  `PR 363 <https://github.com/theislab/squidpy/pull/363>`_.
- Various minor improvements to the documentation `PR 356 <https://github.com/theislab/squidpy/pull/356>`_,
  `PR 358 <https://github.com/theislab/squidpy/pull/358>`_, `PR 359 <https://github.com/theislab/squidpy/pull/359>`_.

.. rubric:: Bugfixes

- Fix :class:`squidpy.im.ImageContainer` to work with different scaled
  `PR 320 <https://github.com/theislab/squidpy/pull/320>`_
- Fix handling of :attr:`anndata.AnnData.obsm` in :meth:`squidpy.im.ImageContainer.interactive`
  `PR 335 <https://github.com/theislab/squidpy/pull/335>`_.
- Fix Z-dimension in :meth:`squidpy.im.ImageContainer.interactive`
  `PR 351 <https://github.com/theislab/squidpy/pull/351>`_.
- Fix plotting bug in :func:`squidpy.pl.ripley` `PR 352 <https://github.com/theislab/squidpy/pull/352>`_.
- Fix handling of NaNs in :func:`squidpy.gr.ligrec` `PR 362 <https://github.com/theislab/squidpy/pull/362>`_.
