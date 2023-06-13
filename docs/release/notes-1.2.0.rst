Squidpy 1.2.0 (2022-04-19)
==========================

Features
--------

- Add :func:`squidpy.pl.spatial_scatter` and :func:`squidpy.pl.spatial_segment` to statically plot
  spatial omics data.
  `@giovp <https://github.com/giovp>`__
  `#437 <https://github.com/scverse/squidpy/pull/437>`__

- Add :func:`squidpy.datasets.visium` to download *10x Genomics* datasets.
  `@dineshpalli <https://github.com/dineshpalli>`__
  `#449 <https://github.com/scverse/squidpy/pull/449>`__

- Add :func:`squidpy.read.visium`, :func:`squidpy.read.vizgen` and :func:`squidpy.read.nanostring` to
  read *Visium*, *Vizgen* and *Nanostring* files, respectively.
  `@dineshpalli <https://github.com/dineshpalli>`__
  `#468 <https://github.com/scverse/squidpy/pull/468>`__

- An optional ``ax`` keyword can now be passed to :func:`squidpy.pl.nhood_enrichment` function. The
  keyword can be used to create a matplotlib figure outside of squidpy and then plot on any matplotlib
  subplot, which could give the user greater flexibility in working with enrichment plots.
  `@jo-mueller <https://github.com/jo-mueller>`__
  `#493 <https://github.com/scverse/squidpy/pull/493>`__

- Add option to load remote *Zarr* store in :class:`squidpy.im.ImageContainer`.
  `@ilan-gold <https://github.com/ilan-gold>`__
  `#500 <https://github.com/scverse/squidpy/pull/500>`__

- Enable specifying diameter in :meth:`squidpy.im.ImageContainer.generate_spot_crops`.
  `@MxMstrmn <https://github.com/MxMstrmn>`__
  `#514 <https://github.com/scverse/squidpy/pull/514>`__

- Add ``library_key`` in :func:`squidpy.gr.spatial_neighbors` to support building graphs across
  multiple slides.
  `@giovp <https://github.com/giovp>`__
  `#516 <https://github.com/scverse/squidpy/pull/516>`__


Bugfixes
--------

- Require ``numba>=0.52.0``.
  `@michalk8 <https://github.com/michalk8>`__
  `#420 <https://github.com/scverse/squidpy/pull/420>`__

- Fix source/target being ``None`` in :func:`squidpy.gr.ligrec`.
  `@michalk8 <https://github.com/michalk8>`__
  `#434 <https://github.com/scverse/squidpy/pull/434>`__

- Do not set edge with in :mod:`napari` since it caused all points to be black.
  `@michalk8 <https://github.com/michalk8>`__
  `#488 <https://github.com/scverse/squidpy/pull/488>`__

- Fix ``use_raw`` in :func:`squidpy.gr.spatial_autocorr`.
  `@michalk8 <https://github.com/michalk8>`__
  `#506 <https://github.com/scverse/squidpy/pull/506>`__

- Include check to be able to load ImageContainer that were generated from another version of squidpy.
  `@MxMstrmn <https://github.com/MxMstrmn>`__
  `#508 <https://github.com/scverse/squidpy/pull/508>`__

- Fix a typo when saving a figure caused a strange directory name to be created.
  `@michalk8 <https://github.com/michalk8>`__
  `#510 <https://github.com/scverse/squidpy/pull/510>`__


Miscellaneous
-------------

- Change imports in the topmost ``__init__.py`` for correct IDE module resolution.
  `@chaichontat <https://github.com/chaichontat>`__
  `#479 <https://github.com/scverse/squidpy/pull/479>`__

- Remove various warnings.
  `@michalk8 <https://github.com/michalk8>`__
  `#489 <https://github.com/scverse/squidpy/pull/489>`__

- Fix missing authors in release notes.
  `@giovp <https://github.com/giovp>`__
  `#520 <https://github.com/scverse/squidpy/pull/520>`__


Documentation
-------------

- Add author to automatically generated news fragment.
  `@michalk8 <https://github.com/michalk8>`__
  `#494 <https://github.com/scverse/squidpy/pull/494>`__
