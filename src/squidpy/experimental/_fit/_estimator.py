"""Estimator contract: a pure, in-memory fit."""

from __future__ import annotations

import importlib.util
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from squidpy.experimental.fit._result import FitResult


class Estimator(ABC):
    """Base class for an in-memory model fit.

    Subclasses implement :meth:`fit`, which takes already-extracted in-memory
    data (NumPy arrays, sparse matrices, or an :class:`~anndata.AnnData`) and
    returns a :class:`FitResult`. Following scikit-learn, ``fit`` assumes the
    training data fits in memory -- there is no streaming / out-of-core
    contract (scikit-learn's ``partial_fit`` is the analogous opt-in exception
    and is intentionally not modelled here).

    Reading data out of a container and writing results back are *not* part of
    this contract; both are handled by the calling function.

    Attributes
    ----------
    name
        Stable identifier used as the registry key.
    requires
        Optional third-party packages the estimator needs at fit time (e.g.
        ``("jax",)``). Advertised here so :meth:`check_requirements` can fail
        early with an actionable message before any heavy import fires.
    """

    name: ClassVar[str]
    requires: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def fit(self, *args: Any, **kwargs: Any) -> FitResult:
        """Fit the model on in-memory data and return a :class:`FitResult`."""

    def check_requirements(self) -> None:
        """Raise :class:`ImportError` if any package in :attr:`requires` is missing.

        Estimators should call this at the top of :meth:`fit`, before importing
        their optional backend, so callers get a clean message instead of a
        deep ``ModuleNotFoundError``.
        """
        missing = [pkg for pkg in self.requires if importlib.util.find_spec(pkg) is None]
        if missing:
            verb = "is" if len(missing) == 1 else "are"
            names = ", ".join(repr(p) for p in missing)
            extras = ",".join(missing)
            raise ImportError(
                f"Estimator {self.name!r} requires {names}, which {verb} not installed. "
                f'Install with `pip install "squidpy[{extras}]"`.'
            )
