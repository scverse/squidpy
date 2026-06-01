"""Private grouping of the ported STalign solver and its estimator adapter.

Importing this package registers :class:`StalignEstimator` in the ``align``
family and stays cheap -- JAX is pulled in lazily, only when ``fit`` runs.
"""

from __future__ import annotations

from squidpy.experimental._fit._methods._stalign._estimator import StalignEstimator, StalignFitResult

__all__ = ["StalignEstimator", "StalignFitResult"]
