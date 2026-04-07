"""Backend discovery via Python entrypoints."""

from __future__ import annotations

import importlib.metadata
import logging
import warnings
from difflib import get_close_matches
from typing import Any

logger = logging.getLogger(__name__)

_backends: dict[str, Any] = {}  # canonical_name -> instance
_alias_map: dict[str, str] = {}  # alias -> canonical_name
_discovered = False

# Trusted (verified) backends and their known aliases.
# Backends not in this list still work but emit a one-time warning on first use.
# To become trusted, submit a PR adding your backend here and pass the
# conformance test suite (squidpy.testing.backend_conformance).
TRUSTED_BACKENDS: dict[str, dict[str, Any]] = {
    "rapids_singlecell": {
        "aliases": ["rapids-singlecell", "rsc", "cuda", "gpu"],
        "package": "rapids-singlecell",
    },
}

# Build reverse lookup: alias -> canonical_name
_TRUSTED_ALIASES: dict[str, str] = {}
for _canonical, _info in TRUSTED_BACKENDS.items():
    _TRUSTED_ALIASES[_canonical] = _canonical
    for _alias in _info["aliases"]:
        _TRUSTED_ALIASES[_alias] = _canonical


def _ensure_discovered() -> None:
    """Discover and register backends via entrypoints (lazy, runs once).

    All backends are loaded on first call. Untrusted backends (not in
    :data:`TRUSTED_BACKENDS`) emit a warning on first use.
    """
    global _discovered
    if _discovered:
        return
    _discovered = True

    for ep in importlib.metadata.entry_points(group="squidpy.backends"):
        try:
            cls = ep.load()
            instance = cls()
            canonical = instance.name
            _backends[canonical] = instance

            # register aliases
            _alias_map[canonical] = canonical
            for alias in getattr(instance, "aliases", []):
                if alias in _alias_map and _alias_map[alias] != canonical:
                    warnings.warn(
                        f"Backend alias {alias!r} claimed by both "
                        f"{_alias_map[alias]!r} and {canonical!r}. "
                        f"Using {_alias_map[alias]!r}.",
                        stacklevel=2,
                    )
                else:
                    _alias_map[alias] = canonical
        except Exception:  # noqa: BLE001
            logger.debug("Failed to load backend entrypoint %r", ep.name, exc_info=True)

    # Merge backend-specific params into dispatched function signatures
    if _backends:
        from squidpy._backends._dispatch import update_signatures

        update_signatures()


def check_trusted(name: str) -> None:
    """Emit a one-time warning if the backend is not in the trusted list."""
    canonical = _alias_map.get(name, name)
    if canonical not in TRUSTED_BACKENDS and canonical in _backends:
        warnings.warn(
            f"Backend {canonical!r} is not in squidpy's trusted backends list. "
            f"It may not have passed the conformance test suite. "
            f"Trusted backends: {sorted(TRUSTED_BACKENDS)}.",
            stacklevel=3,
        )


def _suggest_backend(name: str) -> str:
    """Build an error message with 'did you mean' suggestions."""
    _ensure_discovered()
    all_names = sorted(set(list(_alias_map.keys()) + list(_TRUSTED_ALIASES.keys())))
    matches = get_close_matches(name, all_names, n=1, cutoff=0.4)
    msg = f"Unknown backend {name!r}."
    if matches:
        msg += f" Did you mean {matches[0]!r}?"
    available = available_backend_names()
    if available:
        msg += f" Available: {available}."
    else:
        msg += " No backends are currently installed."
    return msg


def resolve_backend_name(name: str) -> str | None:
    """Resolve alias to canonical backend name.

    Recognises both loaded backends and trusted (but not yet installed)
    backend aliases.  Returns ``None`` only for completely unknown names.
    """
    _ensure_discovered()
    if name == "cpu":
        return "cpu"
    return _alias_map.get(name) or _TRUSTED_ALIASES.get(name)


def get_backend(name: str) -> Any | None:
    """Get backend instance by name or alias. Returns None for 'cpu'."""
    _ensure_discovered()
    if name == "cpu":
        return None
    canonical = _alias_map.get(name) or _TRUSTED_ALIASES.get(name)
    if canonical is None:
        return None
    return _backends.get(canonical)


def available_backend_names() -> list[str]:
    """Return all registered backend names and aliases."""
    _ensure_discovered()
    return sorted(_alias_map.keys())
