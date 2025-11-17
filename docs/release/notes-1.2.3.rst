Squidpy 1.2.3 (2022-10-18)
==========================

Bugfixes
--------

- Fix plotting non-unique categorical colors in :func:`squidpy.pl.spatial_scatter`.
  `@michalk8 <https://github.com/michalk8>`__
  `#561 <https://github.com/scverse/squidpy/pull/561>`__

- Fix :func:`squidpy.read.vizgen`.
  `@giovp <https://github.com/giovp>`__
  `#568 <https://github.com/scverse/squidpy/pull/568>`__

- Convert :attr:`ListedColorMap` to :attr:`Cycler` object.
  `@michalk8 <https://github.com/michalk8>`__
  `#580 <https://github.com/scverse/squidpy/pull/580>`__

- Accomodate latest changes made in spaceranger 2.0
  `@stephenwilliams22 <https://github.com/stephenwilliams22>`__
  `#583 <https://github.com/scverse/squidpy/pull/583>`__

- Fix ligrec from pandas update.
  `@giovp <https://github.com/giovp>`__
  `#609 <https://github.com/scverse/squidpy/pull/609>`__


Miscellaneous
-------------

- Better error message for handling palette in  :func:`squidpy.pl.spatial_scatter`.
  `@giovp <https://github.com/giovp>`__
  `#562 <https://github.com/scverse/squidpy/pull/562>`__

- Update pre-commits and fix CI.
  `@giovp <https://github.com/giovp>`__
  `#587 <https://github.com/scverse/squidpy/pull/587>`__

- Separate linting job on the CI instead as a step. Fix documentation.
  `@michalk8 <https://github.com/michalk8>`__
  `#596 <https://github.com/scverse/squidpy/pull/596>`__

- Update requirements in docs.
  `@michalk8 <https://github.com/michalk8>`__
  `#601 <https://github.com/scverse/squidpy/pull/601>`__

- Change docs theme with Furo, update release note.
  `@giovp <https://github.com/giovp>`__
  `#512 <https://github.com/scverse/squidpy/pull/512>`__

- Fix release notes and misc.
  `@giovp <https://github.com/giovp>`__
  `#617 <https://github.com/scverse/squidpy/pull/617>`__


Documentation
-------------

- Added a squidpy tutorial for Xenium data.
  `@LLehner <https://github.com/LLehner>`__
  `#102 <https://github.com/scverse/squidpy_notebooks/pull/102>`__

- New tutorial for 10x Genomics Xenium data.
  `@LLehner <https://github.com/LLehner>`__
  `#615 <https://github.com/scverse/squidpy/pull/615>`__

- Added tutorial notebook for vizgen mouse liver data.
  `@giovp <https://github.com/giovp>`__
  `#106 <https://github.com/scverse/squidpy_notebooks/pull/106>`__
