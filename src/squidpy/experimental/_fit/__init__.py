"""In-memory model-fitting core for experimental squidpy tools.

This subpackage factors the *fit* half of fitting-style tools (alignment,
imputation, ...) out of any particular container or write-back path:

- :class:`Estimator` -- a model whose :meth:`~Estimator.fit` takes in-memory
  data and returns a :class:`FitResult` (scikit-learn-style; training is
  assumed to fit in memory).
- :class:`FitResult` -- the fitted parameters plus an optional, pure in-memory
  :meth:`~FitResult.transform`. Container write-back lives in the calling
  function, not here.
- :class:`Registry` -- a flat ``name -> Estimator`` map, one per family.

Concrete estimators live in per-family subpackages (``align_samples``,
``align_landmarks``, ...) and pull their optional dependencies (e.g. JAX) in
lazily, so importing this package stays cheap.

Example
-------
>>> import numpy as np
>>> from squidpy.experimental._fit import Estimator, FitResult, Registry
>>> demo = Registry("demo")
>>> class Shift(FitResult):
...     def __init__(self, delta):
...         self.metadata, self.delta = {}, delta
...
...     def transform(self, x):
...         return np.asarray(x) + self.delta
>>> @demo.register("mean_shift")
... class MeanShift(Estimator):
...     name = "mean_shift"
...
...     def fit(self, ref, query):
...         return Shift(ref.mean(0) - query.mean(0))
>>> est = demo.get("mean_shift")()
>>> result = est.fit(np.ones((3, 2)), np.zeros((3, 2)))
>>> result.transform(np.zeros((3, 2))).tolist()
[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]
"""

from __future__ import annotations

from squidpy.experimental._fit._estimator import Estimator
from squidpy.experimental._fit._registry import Registry
from squidpy.experimental._fit._result import FitResult

__all__ = ["Estimator", "FitResult", "Registry"]
