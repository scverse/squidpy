"""Backend settings with thread-safe context variable."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator

_backend_var: ContextVar[str] = ContextVar("backend", default="cpu")


class _Settings:
    """Global settings for squidpy's backend dispatch system.

    Access via the singleton ``squidpy.settings``.

    Examples
    --------
    >>> import squidpy as sq
    >>> sq.settings.backend = "gpu"
    >>> sq.settings.available_backends()
    ['rapids_singlecell']
    """

    @property
    def backend(self) -> str:
        """The active backend name (default ``'cpu'``).

        Set to a registered backend name or alias (e.g. ``'gpu'``, ``'cuda'``)
        to dispatch supported functions to that backend.
        Aliases are resolved to the canonical name.

        Examples
        --------
        >>> sq.settings.backend = "gpu"
        >>> sq.settings.backend
        'rapids_singlecell'
        """
        return _backend_var.get()

    @backend.setter
    def backend(self, value: str) -> None:
        from squidpy._backends._registry import (
            TRUSTED_BACKENDS,
            _suggest_backend,
            check_trusted,
            get_backend,
            resolve_backend_name,
        )

        if value == "cpu":
            _backend_var.set(value)
            return

        canonical = resolve_backend_name(value)

        # Completely unknown name — suggest alternatives
        if canonical is None:
            raise ValueError(_suggest_backend(value))

        # Trusted but not installed
        if canonical in TRUSTED_BACKENDS and get_backend(canonical) is None:
            package = TRUSTED_BACKENDS[canonical]["package"]
            raise ImportError(
                f"Backend {value!r} ({canonical}) is not installed. Install it with: pip install {package}"
            )

        # Known alias but backend not loaded
        if get_backend(canonical) is None:
            raise ImportError(f"Backend {value!r} is not installed.")

        # Warn if untrusted
        check_trusted(canonical)

        # Always store the canonical name
        _backend_var.set(canonical)

    @contextmanager
    def use_backend(self, backend: str) -> Generator[None, None, None]:
        """Temporarily set the backend within a context.

        Parameters
        ----------
        backend
            The backend to use inside the context.

        Examples
        --------
        >>> with sq.settings.use_backend("gpu"):
        ...     sq.gr.spatial_autocorr(adata)
        """
        token = _backend_var.set(self.backend)
        try:
            self.backend = backend
            yield
        finally:
            _backend_var.reset(token)

    @staticmethod
    def available_backends() -> list[str]:
        """Return canonical names of all discovered backends.

        Examples
        --------
        >>> sq.settings.available_backends()
        ['rapids_singlecell']
        """
        from squidpy._backends._registry import _backends, _ensure_discovered

        _ensure_discovered()
        return sorted(_backends.keys())

    @staticmethod
    def get_backend(name: str) -> Any | None:
        """Look up a backend by name or alias.

        Parameters
        ----------
        name
            Canonical name or alias (e.g. ``'gpu'``, ``'cuda'``,
            ``'rapids_singlecell'``).

        Returns
        -------
        The backend instance, or ``None`` if not found.

        Examples
        --------
        >>> backend = sq.settings.get_backend("gpu")
        >>> backend.name
        'rapids_singlecell'
        """
        from squidpy._backends._registry import get_backend

        return get_backend(name)


settings = _Settings()
