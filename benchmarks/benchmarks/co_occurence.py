"""Benchmark tool operations in Squidpy.

API documentation: <https://squidpy.readthedocs.io/en/stable/api>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import scanpy as sc
import squidpy as sq

from ._utils import pbmc68k_reduced

if TYPE_CHECKING:
    from anndata import AnnData

# setup variables

adata: AnnData


def setup():
    global adata  # noqa: PLW0603
    adata = sq.datasets.imc()


def time_co_occurrence():
    sq.gr.co_occurrence(adata, cluster_key="cell type")


def peakmem_co_occurrence():
    sq.gr.co_occurrence(adata, cluster_key="cell type")

