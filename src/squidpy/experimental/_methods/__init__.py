"""In-memory model-fitting core for experimental squidpy tools.

This subpackage factors the *fit* half of fitting-style tools (alignment,
imputation, ...) out of any particular container or write-back path. Each
family registers its estimators into a flat :class:`Registry` (``name ->
function``). An estimator returns a plain result dataclass that carries the
fitted parameters and a ``transform`` method; turning that into mutations on an
:class:`~anndata.AnnData` / :class:`~spatialdata.SpatialData` is the calling
function's job, not the result's.

Concrete estimators live in per-family subpackages (``align_samples``,
``align_landmarks``, ...) and pull their optional dependencies (e.g. JAX) in
lazily, so importing this package stays cheap.
"""

from __future__ import annotations

from squidpy.experimental._methods._registry import Registry

__all__ = ["Registry"]
