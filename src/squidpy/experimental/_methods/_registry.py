"""A flat registry mapping method names to fitting functions."""

from __future__ import annotations

import functools
import importlib.util
from collections.abc import Callable
from typing import Any, Generic, TypeVar

#: The calling convention a family's registry advertises (returned by :meth:`Registry.get`).
F = TypeVar("F", bound=Callable[..., Any])
#: The concrete function being registered. Kept separate from ``F`` so an estimator may
#: declare specific keyword parameters (e.g. ``config=``) without having to structurally
#: match the family's open-ended ``**kwargs`` calling convention.
RegisteredT = TypeVar("RegisteredT", bound=Callable[..., Any])


class Registry(Generic[F]):
    """A flat ``name -> function`` registry for one *family* of methods.

    One :class:`Registry` is created per family (e.g. ``align``, ``impute``),
    so keys are plain method names -- there is no ``(method, mode)`` compound
    key, because the family already pins the rest.

    The type parameter ``F`` is the family's calling convention (a callable
    :class:`~typing.Protocol`); :meth:`get` returns it, so dispatch sites are
    typed against the family contract rather than ``Callable[..., Any]``.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._registry: dict[str, F] = {}

    def register(self, key: str, *, requires: tuple[str, ...] = ()) -> Callable[[RegisteredT], RegisteredT]:
        """Return a decorator registering a method/function under ``key``."""

        def decorator(func: RegisteredT) -> RegisteredT:
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

                self._registry[key] = wrapped  # type: ignore[assignment]
                return wrapped  # type: ignore[return-value]
            else:
                self._registry[key] = func  # type: ignore[assignment]
                return func

        return decorator

    def get(self, key: str) -> F:
        """Return the function registered under ``key``."""
        try:
            return self._registry[key]
        except KeyError:
            raise ValueError(f"Unknown {self.name} method {key!r}. Available: {sorted(self._registry)}.") from None

    def keys(self) -> tuple[str, ...]:
        """Return the registered method names."""
        return tuple(self._registry)
