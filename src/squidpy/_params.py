"""Defaults, validation and resolution for the parameter TypedDicts. Private."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from typing import Any, cast, get_type_hints


@dataclass(frozen=True, slots=True)
class Default:
    """A params key's default, declared as ``Annotated[type, Default(value)]``."""

    value: Any


def defaults_of[T: Mapping[str, Any]](spec: type[T]) -> T:
    """The :class:`Default` of every key of *spec*; raises if a key has none."""
    defaults = {}
    for key, hint in get_type_hints(spec, include_extras=True).items():
        marker = next((m for m in getattr(hint, "__metadata__", ()) if isinstance(m, Default)), None)
        if marker is None:
            raise TypeError(f"`{spec.__name__}.{key}` is missing a `Default(...)` in its annotation.")
        defaults[key] = marker.value
    return cast("T", defaults)


# never handed out, only merged from, so caching it is safe
_cached_defaults = cache(defaults_of)

_VALIDATORS: dict[type, Callable[[dict[str, Any]], None]] = {}


def validates[F: Callable[[dict[str, Any]], None]](spec: type) -> Callable[[F], F]:
    """Register the decorated function as *spec*'s validator, run by :func:`resolve_params`.

    It coerces the merged mapping in place and raises on invalid values. Every spec
    passed to :func:`resolve_params` needs one.
    """

    def register(validate: F) -> F:
        _VALIDATORS[spec] = validate
        return validate

    return register


def resolve_params[T: Mapping[str, Any]](
    params: T | Mapping[str, Any] | None,
    spec: type[T],
    *,
    arg_name: str = "method_params",
) -> T:
    """Merge *params* over the defaults of *spec*, then validate the result.

    Unknown keys raise, and so does a *spec* without a :func:`validates` validator. The
    validator runs on the merged mapping, so defaults are checked too. Returns a new mapping.
    """
    defaults = _cached_defaults(spec)
    if params is not None and not isinstance(params, Mapping):
        raise TypeError(f"`{arg_name}` must be a Mapping or None; got {type(params).__name__}.")
    if params:
        unknown = set(params) - set(defaults)
        if unknown:
            raise ValueError(f"Unknown `{arg_name}` field(s): {sorted(unknown)}; expected from {sorted(defaults)}.")
    merged = {**defaults, **(params or {})}
    if (validate := _VALIDATORS.get(spec)) is None:
        raise TypeError(f"`{spec.__name__}` has no validator; register one with `@validates`.")
    validate(merged)
    return cast("T", merged)
