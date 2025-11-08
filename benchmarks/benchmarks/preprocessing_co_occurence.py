"""Benchmark co-occurrence operations in Squidpy.

API documentation: <https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.co_occurrence.html>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._utils import get_dataset, param_skipper

if TYPE_CHECKING:
    from anndata import AnnData

    from ._utils import Dataset, KeyX

from squidpy.gr import co_occurrence  # type: ignore[attr-defined]

# setup variables


adata: AnnData
cluster_key: str | None


def setup(dataset: Dataset, layer: KeyX, *_) -> None:  # type: ignore[no-untyped-def]
    """Set up global variables before each benchmark."""
    global adata, cluster_key
    adata, cluster_key = get_dataset(dataset, layer=layer)


# ASV suite

params: tuple[list[Dataset], list[KeyX]] = (
    [
        "imc",
    ],
    [None, "off-axis"],
)
param_names = ["dataset", "layer"]

skip_when = param_skipper(param_names, params)


def time_co_occurrence(*_) -> None:  # type: ignore[no-untyped-def]
    co_occurrence(adata, cluster_key=cluster_key)


def peakmem_co_occurrence(*_) -> None:  # type: ignore[no-untyped-def]
    co_occurrence(adata, cluster_key=cluster_key)
