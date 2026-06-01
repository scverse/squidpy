"""Concrete estimators for :mod:`squidpy.experimental.fit`.

Importing this package registers the bundled estimators (e.g. STalign in the
``align`` family) but stays cheap: optional dependencies such as JAX are pulled
in lazily, only when an estimator's ``fit`` is actually called.
"""

from __future__ import annotations

from squidpy.experimental.fit._methods._stalign import ALIGN, StalignEstimator, StalignFitResult

__all__ = ["ALIGN", "StalignEstimator", "StalignFitResult"]
