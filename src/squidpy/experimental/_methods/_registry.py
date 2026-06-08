"""A flat registry mapping method names to fitting functions."""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Registry:
    """A flat ``name -> function`` registry for one *family* of methods.

    One :class:`Registry` is created per family (e.g. ``align``, ``impute``),
    so keys are plain method names -- there is no ``(method, mode)`` compound
    key, because the family already pins the rest.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, key: str, *, requires: tuple[str, ...] = ()) -> Callable[[F], F]:
        """Return a decorator registering a method/function under ``key``."""

        def decorator(func: F) -> F:
            if key in self._registry:
                raise ValueError(f"Method {key!r} is already registered in the {self.name!r} registry.")

            if requires:
                @functools.wraps(func)
                def wrapped(*args: Any, **kwargs: Any) -> Any:
                    missing = [pkg for pkg in requires if importlib.util.find_spec(pkg) is None]
                    if missing:
                        verb = "is" if len(missing) == 1 else "are"
                        names = ", ".join(repr(p) for p in missing)
                        extras = ",".join(missing)
                        raise ImportError(
                            f"Method {key!r} requires {names}, which {verb} not installed. "
                            f'Install with `pip install "squidpy[{extras}]"`.'
                        )
                    return func(*args, **kwargs)

                self._registry[key] = wrapped
                return wrapped  # type: ignore[return-value]
            else:
                self._registry[key] = func
                return func

        return decorator

    def get(self, key: str) -> Callable[..., Any]:
        """Return the function registered under ``key``."""
        try:
            return self._registry[key]
        except KeyError:
            raise ValueError(f"Unknown {self.name} method {key!r}. Available: {sorted(self._registry)}.") from None

    def keys(self) -> tuple[str, ...]:
        """Return the registered method names."""
        return tuple(self._registry)

