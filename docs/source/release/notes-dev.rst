Squidpy dev (2022-03-24)
========================

Features
--------

- An optional ``ax`` keyword can now be passed to :func:`squidpy.pl.nhood_enrichment` function. The
  keyword can be used to create a matplotlib figure outside of squidpy and then plot on any matplotlib
  subplot, which could give the user greater flexibility in working with enrichment plots.
  `@jo-mueller <https://github.com/jo-mueller>`__
  `#493 <https://github.com/theislab/squidpy/pull/493>`__


Bugfixes
--------

- Require ``numba>=0.52.0``.
  `#420 <https://github.com/theislab/squidpy/pull/420>`__

- Fix source/target being ``None`` in :func:`squidpy.gr.ligrec`.
  `#434 <https://github.com/theislab/squidpy/pull/434>`__

- Do not set edge with in :mod:`napari` since it caused all points to be black.
  `#488 <https://github.com/theislab/squidpy/pull/488>`__


Miscellaneous
-------------

- Change imports in the topmost ``__init__.py`` for correct IDE module resolution.
  `#479 <https://github.com/theislab/squidpy/pull/479>`__

- Remove various warnings.
  `#489 <https://github.com/theislab/squidpy/pull/489>`__
