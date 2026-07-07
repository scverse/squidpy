"""Shared internal helper for resolving params-dataclass arguments.

Not part of the public API - symbols here are private and may change
without notice.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
from typing import Any


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
