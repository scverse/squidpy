"""Shared internal helper for resolving params-TypedDict arguments.

Not part of the public API - symbols here are private and may change
without notice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast, get_type_hints


@dataclass(frozen=True, slots=True)
class Default:
    """The default value of a params key, carried in its ``Annotated`` metadata.

    Keeps the default next to the key and its docstring instead of in a parallel
    mapping -- one source of truth for the resolver and for the docs.
    """

    value: Any


def defaults_of[T: Mapping[str, Any]](spec: type[T]) -> T:
    """Collect the :class:`Default` of every key of a params TypedDict.

    Raises if a key declares no default: with ``total=False`` a type checker
    cannot see a missing entry, so catch it at import time instead.
    """
    defaults = {}
    for key, hint in get_type_hints(spec, include_extras=True).items():
        marker = next((m for m in getattr(hint, "__metadata__", ()) if isinstance(m, Default)), None)
        if marker is None:
            raise TypeError(f"`{spec.__name__}.{key}` is missing a `Default(...)` in its annotation.")
        defaults[key] = marker.value
    return cast("T", defaults)


def resolve_params[T: Mapping[str, Any]](
    params: T | Mapping[str, Any] | None,
    *,
    defaults: T,
    validate: Callable[[dict[str, Any]], None] | None = None,
    arg_name: str = "method_params",
) -> T:
    """Merge a params mapping over ``defaults`` and validate the result.

    ``T`` is a :class:`~typing.TypedDict`, so callers get static key and value
    checking at the call site; this function is the dynamic half. Unknown keys
    are named rather than silently ignored (a plain ``dict`` would accept them),
    and ``validate`` -- which coerces in place and range-checks -- runs on the
    *merged* mapping, so the defaults are checked on every call rather than
    trusted.

    Returns a new mapping; neither ``params`` nor ``defaults`` is mutated.
    """
    if params is not None and not isinstance(params, Mapping):
        raise TypeError(f"`{arg_name}` must be a Mapping or None; got {type(params).__name__}.")
    if params:
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown `{arg_name}` field(s): {sorted(unknown)}; expected from {sorted(defaults)}.")
    merged = {**defaults, **(params or {})}
    if validate is not None:
        validate(merged)
    return cast("T", merged)
