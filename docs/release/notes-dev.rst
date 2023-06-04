Squidpy dev (2023-03-11)
========================

Bugfixes
--------

- Fix :func:`squidpy.pl.extract` on views.
  `@michalk8 <https://github.com/michalk8>`__
  `#663 <https://github.com/scverse/squidpy/pull/663>`__

- Set coordinates' index type to same as in :attr:`anndata.AnnData.obs` in :func:`squidpy.read.vizgen`
  and :func:`squidpy.read.visium`.
  `@michalk8 <https://github.com/michalk8>`__
  `#665 <https://github.com/scverse/squidpy/pull/665>`__


Squidpy dev (2023-04-01)
========================

Miscellaneous
-------------

- Add :attr:`attr` option to :func:`squidpy.gr.spatial_autocorr` to select values from :attr:`anndata.AnnData.obs`
  or :attr:`anndata.AnnData.obsm.`
  `@michalk8 <https://github.com/michalk8>`__
  `#672 <https://github.com/scverse/squidpy/pull/672>`__


Squidpy dev (2023-04-02)
========================

Features
--------

- Add :func:`squidpy.tl.var_by_distance` to calculate distances to user-defined anchor points
  and store the resulting design matrix in :attr:`adata.obsm`.
- Add :func:`squidpy.pl.var_by_distance` to visualize a variable such as expression by distance to an anchor points.
  `@LLehner <https://github.com/LLehner>`__
  `#591 <https://github.com/scverse/squidpy/pull/591>`__
