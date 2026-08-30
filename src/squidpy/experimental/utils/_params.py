"""Shared internal helper for resolving params-dataclass arguments.

Not part of the public API - symbols here are private and may change
without notice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, cast, get_type_hints


@dataclass(frozen=True, slots=True, repr=False)
class Default:
    """The default value of a params key, carried in its ``Annotated`` metadata.

    Keeps the default next to the key and its docstring instead of in a parallel
    mapping -- one source of truth for the resolver and for the docs.
    """

    value: Any

    def __repr__(self) -> str:
        """Render as ``Default(0.8)``.

        The generated ``repr`` spells the field name, and Sphinx mangles an ``=`` inside a
        rendered annotation into garbage -- so the marker prints its value positionally.
        """
        return f"Default({self.value!r})"


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


def resolve_params[T](value: T | Mapping[str, Any] | None, cls: type[T], *, label: str) -> T:
    """Normalise a params argument (``None`` / instance / ``Mapping``) to a ``cls`` instance.

    Parameters
    ----------
    value
        ``None`` (use defaults), an instance of ``cls`` (passed through by
        identity), or a ``Mapping`` of field names to values.
    cls
        The params dataclass to construct.
    label
        The user-facing argument name used verbatim in error messages.  Include
        backticks if the caller's convention uses them (e.g. ``"`tiling_qc_params`"``).
    """
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if isinstance(value, Mapping):
        valid = {f.name for f in fields(cls)}
        unknown = set(value) - valid
        if unknown:
            raise ValueError(f"Unknown {label} field(s): {sorted(unknown)}; expected from {sorted(valid)}.")
        return cls(**value)
    raise TypeError(f"{label} must be {cls.__name__}, Mapping, or None; got {type(value).__name__}.")
