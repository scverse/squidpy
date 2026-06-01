"""Result contract for fitted estimators."""

from __future__ import annotations

from abc import ABC
from typing import Any


class FitResult(ABC):
    """Base class for the output of :meth:`Estimator.fit`.

    A ``FitResult`` carries the *fitted parameters* of a model -- pointers or
    parametric forms (an affine matrix, a displacement field, neighbour
    weights), not bulk data -- together with free-form :attr:`metadata`.

    It is deliberately container-agnostic. Turning a result into mutations on
    an :class:`~anndata.AnnData` / :class:`~spatialdata.SpatialData`
    (write-back) is the caller's responsibility, not the result's: write-back
    is a general squidpy concern with several unrelated shapes (set a field,
    apply a transformation to a lazy element, ...) and does not belong on the
    fit result.
    """

    metadata: dict[str, Any]

    def transform(self, x: Any) -> Any:
        """Apply the fitted model to in-memory data ``x`` and return the result.

        Optional. Estimators whose fit does not yield a reusable map (or that
        only produce side outputs) need not implement it.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement `transform`.")
