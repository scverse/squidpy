from __future__ import annotations

from importlib import metadata
from importlib.metadata import PackageMetadata

from squidpy import datasets, experimental, gr, im, pl, read, tl
from squidpy._backends import settings

try:
    md: PackageMetadata = metadata.metadata(__name__)
    __version__ = md["Version"] if "Version" in md else ""
    __author__ = md["Author"] if "Author" in md else ""
    __maintainer__ = md["Maintainer-email"] if "Maintainer-email" in md else ""
except ImportError:
    md = None  # type: ignore[assignment]

del metadata, md


def _check_backends() -> None:
    from squidpy._backends._dispatch import update_signatures
    from squidpy._backends._registry import _backends, discover_backends

    discover_backends()
    update_signatures()
    if _backends:
        import logging

        names = ", ".join(_backends.keys())
        logging.getLogger(__name__).info(
            f"GPU backend available ({names}). "
            f"Set squidpy.settings.backend = 'cuda' to enable."
        )


_check_backends()
del _check_backends

__all__ = ["datasets", "experimental", "gr", "im", "pl", "read", "settings", "tl"]
