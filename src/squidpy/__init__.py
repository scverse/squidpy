from __future__ import annotations

from importlib import metadata
from importlib.metadata import PackageMetadata

from squidpy import datasets, gr, im, pl, read, tl

try:
    md: PackageMetadata = metadata.metadata(__name__)
    __version__ = md["Version"] if "Version" in md else ""
    __author__ = md["Author"] if "Author" in md else ""
    __maintainer__ = md["Maintainer-email"] if "Maintainer-email" in md else ""
except ImportError:
    md = None  # type: ignore[assignment]

del metadata, md
