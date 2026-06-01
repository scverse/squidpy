"""A flat registry mapping method names to estimator classes."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from squidpy.experimental._fit._estimator import Estimator

E = TypeVar("E", bound="type[Estimator]")


class Registry:
    """A flat ``name -> Estimator`` registry for one *family* of methods.

    One :class:`Registry` is created per family (e.g. ``align``, ``impute``),
    so keys are plain method names -- there is no ``(method, mode)`` compound
    key, because the family already pins the rest.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._estimators: dict[str, type[Estimator]] = {}

    def register(self, key: str) -> Callable[[E], E]:
        """Return a class decorator registering an :class:`Estimator` under ``key``."""

        def decorator(cls: E) -> E:
            if key in self._estimators:
                raise ValueError(f"Estimator {key!r} is already registered in the {self.name!r} registry.")
            self._estimators[key] = cls
            return cls

        return decorator

    def get(self, key: str) -> type[Estimator]:
        """Return the :class:`Estimator` class registered under ``key``."""
        try:
            return self._estimators[key]
        except KeyError:
            raise ValueError(f"Unknown {self.name} method {key!r}. Available: {sorted(self._estimators)}.") from None

    def keys(self) -> tuple[str, ...]:
        """Return the registered method names."""
        return tuple(self._estimators)
