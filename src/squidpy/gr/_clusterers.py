from __future__ import annotations

from typing import Any, Literal, Protocol, Self, runtime_checkable

import anndata as ad
import pandas as pd
import scanpy as sc
from fast_array_utils.types import HasArrayNamespace as Array
from sklearn.base import BaseEstimator, ClusterMixin

__all__ = ["Clusterer", "LeidenClusterer"]


@runtime_checkable
class Clusterer(Protocol):
    """Assigns one cluster label per observation, and can be re-fitted with a new seed.

    ``get_params``/``set_params`` are part of it because the niche pipeline fits once per
    library: every fit goes to a fresh :func:`~sklearn.base.clone`, so the estimator handed
    in is never mutated, and cloning needs nothing beyond those two.
    """

    def fit_predict(self, X: Array) -> Array:
        """Cluster *X*, observations as rows, and return one label per row."""
        ...

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """The constructor parameters, as :func:`~sklearn.base.clone` reads them."""
        ...

    def set_params(self, *, random_state: int) -> Self:
        """Seed the next fit; returns the estimator."""
        ...


class LeidenClusterer(ClusterMixin, BaseEstimator):
    """Leiden partition of an embedding, at one resolution.

    One resolution per clusterer, so a resolution sweep is a mapping of them rather than a
    loop inside one. ``X`` is the embedding; the neighborhood graph is built here.

    Parameters
    ----------
    n_neighbors
        Neighbors used to build the graph the partition runs on.
    resolution
        Resolution of the partition. Higher gives more, smaller niches.
    flavor
        Leiden backend.
    n_iterations
        Iterations passed to the backend.
    random_state
        Seeds the partition.
    """

    def __init__(
        self,
        *,
        n_neighbors: int = 15,
        resolution: float = 1.0,
        flavor: Literal["igraph", "leidenalg"] = "igraph",
        n_iterations: int = -1,
        random_state: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.resolution = resolution
        self.flavor = flavor
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X: Array, y: None = None) -> LeidenClusterer:
        """Partition *X*, leaving the labels in ``labels_``."""
        shell = ad.AnnData(X=X, obs=pd.DataFrame(index=[str(i) for i in range(X.shape[0])]))
        # no dimensionality reduction, matching the original implementation
        sc.pp.neighbors(shell, n_neighbors=self.n_neighbors, use_rep="X")

        # Default to the igraph backend so niche labels are reproducible across versions;
        # leidenalg is deprecated in scanpy and unstable on small graphs. See
        # scverse/squidpy#1260.
        kwargs: dict[str, Any] = {"flavor": self.flavor, "n_iterations": self.n_iterations}
        # scanpy's igraph backend only supports undirected graphs and errors if `directed`
        # is left at the leidenalg default of True, so pin it to False.
        if self.flavor == "igraph":
            kwargs["directed"] = False
        sc.tl.leiden(
            shell,
            resolution=self.resolution,
            key_added="niche",
            random_state=self.random_state,
            **kwargs,
        )

        self.labels_ = shell.obs["niche"].to_numpy()
        return self
