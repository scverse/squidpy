"""Read a squidpy-ports reference bundle: outputs of an original implementation that a port is compared to.

https://github.com/theislab/squidpy-ports documents the format: one AnnData in a zipped zarr store
whose ``uns`` holds the fixture parameters in ``fixtures``, any inputs in ``inputs``, and one
section of outputs per setting, each with its ``provenance``.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class Bundle(NamedTuple):
    fixtures: dict[str, Any]
    inputs: dict[str, Any]
    sections: dict[str, dict[str, Any]]


def read_bundle(url: str, sha256: str) -> Bundle:
    """Download the bundle at ``url`` once, checking its ``sha256``, and read it into memory."""
    import anndata as ad
    import pooch
    from scanpy import settings
    from zarr.storage import ZipStore

    path = pooch.retrieve(url, known_hash=f"sha256:{sha256}", path=settings.datasetdir, progressbar=False)
    with ZipStore(path, mode="r") as store:
        uns = dict(ad.read_zarr(store).uns)
    return Bundle(uns.pop("fixtures"), uns.pop("inputs", {}), uns)
