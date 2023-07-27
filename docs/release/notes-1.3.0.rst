Squidpy 1.3.0 (2023-06-16)
==========================

Features
--------
- Add :func:`squidpy.tl.var_by_distance` to calculate distances to anchor points and store results in a design matrix.
- Add :func:`squidpy.pl.var_by_distance` to visualize variables such as gene expression by distance to an anchor point.
  `@LLehner <https://github.com/LLehner>`__
  `#591 <https://github.com/scverse/squidpy/pull/591>`__


Bugfixes
--------

- Fix :mod:`pandas` inf :func:`squidpy.pl.ligrec`.
  `@michalk8 <https://github.com/michalk8>`__
  `#625 <https://github.com/scverse/squidpy/pull/625>`__

- Remove column assignment to improve compatibility with new cell metadata.
  `@cornhundred <https://github.com/cornhundred>`__
  `#648 <https://github.com/scverse/squidpy/pull/648>`__

- Fix :func:`squidpy.pl.extract` on views.
  `@michalk8 <https://github.com/michalk8>`__
  `#663 <https://github.com/scverse/squidpy/pull/663>`__

- Set coordinates' index type to same as in :attr:`anndata.AnnData.obs` in :func:`squidpy.read.vizgen`
  and :func:`squidpy.read.visium`.
  `@michalk8 <https://github.com/michalk8>`__
  `#665 <https://github.com/scverse/squidpy/pull/665>`__

- Update cell metadata index conversion.
  `@djlee1 <https://github.com/djlee1>`__
  `#679 <https://github.com/scverse/squidpy/pull/679>`__

- Fix previously updated cell metadata index conversion.
  `@dfhannum <https://github.com/dfhannum>`__
  `#692 <https://github.com/scverse/squidpy/pull/692>`__


Miscellaneous
-------------

- Update pre-commits and unpin numba and numpy.
  `@giovp <https://github.com/giovp>`__
  `#643 <https://github.com/scverse/squidpy/pull/643>`__

- Add :attr: option to :func:`squidpy.gr.spatial_autocorr` to select values from :attr:`anndata.AnnData.obs` or :attr:`anndata.AnnData.obsm`.
  `@michalk8 <https://github.com/michalk8>`__
  `#664 <https://github.com/scverse/squidpy/pull/664>`__

- Add :attr:`attr` option to :func:`squidpy.gr.spatial_autocorr` to select values from :attr:`anndata.AnnData.obs`
  or :attr:`anndata.AnnData.obsm.`
  `@michalk8 <https://github.com/michalk8>`__
  `#672 <https://github.com/scverse/squidpy/pull/672>`__

- Add :attr:`percentile` option to :func:`squidpy.gr.spatial_neighbors` to filter neighbor graph using percentile of distances threshold.
  `@LLehner <https://github.com/LLehner>`__
  `#690 <https://github.com/scverse/squidpy/pull/690>`__

- Add :class:`spatialdata.SpatialData` as possible input for graph functions.
  `@LLehner <https://github.com/LLehner>`__
  `#701 <https://github.com/scverse/squidpy/pull/701>`__


Documentation
-------------

- Fix CI badges and tox.
  `@michalk8 <https://github.com/michalk8>`__
  `#627 <https://github.com/scverse/squidpy/pull/627>`__

- Changed tutorial directory structure.
  `@LLehner <https://github.com/LLehner>`__
  `#113 <https://github.com/scverse/squidpy_notebooks/pull/113>`__

- Updated the quality control tutorials for Vizgen, Xenium and Nanostring.
  `@pakiessling <https://github.com/pakiessling>`__
  `#110 <https://github.com/scverse/squidpy_notebooks/pull/110>`__

- Improved example for :func:`squidpy.tl.var_by_distance` and :func:`squidpy.pl.var_by_distance`.
  `@LLehner <https://github.com/LLehner>`__
  `#115 <https://github.com/scverse/squidpy_notebooks/pull/115>`__
