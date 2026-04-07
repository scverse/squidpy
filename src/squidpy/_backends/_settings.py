"""Backend settings with thread-safe context variable."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator

_backend_var: ContextVar[str] = ContextVar("backend", default="cpu")


class _Settings:
    @property
    def backend(self) -> str:
        return _backend_var.get()

    @backend.setter
    def backend(self, value: str) -> None:
        from squidpy._backends._registry import (
            _TRUSTED_ALIASES,
            TRUSTED_BACKENDS,
            _check_trusted,
            available_backend_names,
            get_backend,
            resolve_backend_name,
        )

        if value == "cpu":
            _backend_var.set(value)
            return

        resolved = resolve_backend_name(value)

        # Completely unknown name and not a loaded backend
        if resolved is None:
            raise ValueError(f"Unknown backend {value!r}. Available backends: {available_backend_names() or ['cpu']}.")

        # Trusted alias but backend package not installed
        if resolved in _TRUSTED_ALIASES and get_backend(value) is None:
            canonical = _TRUSTED_ALIASES[resolved]
            package = TRUSTED_BACKENDS[canonical]["package"]
            raise ValueError(
                f"Backend {value!r} ({canonical}) is not installed. Install it with: pip install {package}"
            )

        # Loaded but not installed (untrusted, not in alias map for trusted)
        if get_backend(value) is None:
            raise ValueError(
                f"Backend {value!r} is not installed. Available backends: {available_backend_names() or ['cpu']}."
            )

        # Warn if untrusted
        _check_trusted(value)

        _backend_var.set(value)

    @contextmanager
    def use_backend(self, backend: str) -> Generator[None, None, None]:
        token = _backend_var.set(self.backend)
        try:
            self.backend = backend
            yield
        finally:
            _backend_var.reset(token)


settings = _Settings()
