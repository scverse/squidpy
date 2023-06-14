from importlib import metadata

from squidpy import datasets, gr, im, pl, read, tl

try:
    md = metadata.metadata(__name__)
    __version__ = md.get("version", "")
    __author__ = md.get("Author", "")
    __maintainer__ = md.get("Maintainer-email", "")
except ImportError:
    md = None  # type: ignore[assignment]

del metadata, md
