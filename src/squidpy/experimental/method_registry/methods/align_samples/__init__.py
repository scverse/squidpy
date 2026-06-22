"""``align_samples`` family: align two samples' point clouds (STalign).

Importing this package registers the family's estimators into
:data:`~squidpy.experimental.method_registry.registry.ALIGN_SAMPLES`. It stays
cheap -- JAX is pulled in lazily, only when an estimator's ``fit`` runs. Only the
implementations are re-exported here; the registry itself lives in (and is public
from) :mod:`squidpy.experimental.method_registry`.
"""

from __future__ import annotations

from squidpy.experimental.method_registry.methods.align_samples._stalign import StalignResult, fit_stalign

__all__ = ["fit_stalign", "StalignResult"]
