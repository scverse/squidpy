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

Miscellaneous
-------------

- Add :attr:`attr` option to :func:`squidpy.gr.spatial_autocorr` to select values from :attr:`anndata.AnnData.obs`
  or :attr:`anndata.AnnData.obsm.`
  `@michalk8 <https://github.com/michalk8>`__
  `#672 <https://github.com/scverse/squidpy/pull/672>`__
