"""``align_samples`` family: align two samples' point clouds (STalign).

Importing this package registers the family's estimators into
:data:`~squidpy.experimental._methods._families.ALIGN_SAMPLES`. It stays cheap --
JAX is pulled in lazily, only when an estimator's ``fit`` runs.
"""

from __future__ import annotations

from squidpy.experimental._methods._families import ALIGN_SAMPLES
from squidpy.experimental._methods.align_samples._stalign import fit_stalign
from squidpy.experimental._methods.align_samples._stalign_impl._tools import StalignResult

__all__ = ["ALIGN_SAMPLES", "fit_stalign", "StalignResult"]
