Squidpy 1.1.1 (2021-08-16)
==========================

Features
--------

- Allow defining cylindrical shells in :func:`squidpy.gr.spatial_neighbors` by using the ``radius`` argument.
  Also rename ``n_neigh``, ``n_neigh_grid`` arguments to ``n_neighs``.
  `#393 <https://github.com/scverse/squidpy/pull/393>`__

- Allow specifying gene symbols from :attr:`anndata.AnnData.var` in :func:`squidpy.gr.ligrec`.
  `#395 <https://github.com/scverse/squidpy/pull/395>`__


Bugfixes
--------

- Fix sometimes incorrectly transposing dimensions when reading TIFF files.
  `#390 <https://github.com/scverse/squidpy/pull/390>`__


Miscellaneous
-------------

- Increase performance of Delaunay graph creation in :func:`squidpy.gr.spatial_neighbors`.
  `#381 <https://github.com/scverse/squidpy/pull/381>`__

- Update ``mypy`` type-checking and use PEP 604 for type annotations.
  `#396 <https://github.com/scverse/squidpy/pull/396>`__


Documentation
-------------

- Enable ``towncrier`` for release notes generation.
  `#397 <https://github.com/scverse/squidpy/pull/397>`__
