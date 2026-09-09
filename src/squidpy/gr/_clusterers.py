"""The clusterers behind the niche flavors.

scikit-learn estimators, so that a flavor's clusterer slot takes squidpy's own and
:mod:`sklearn.cluster`'s interchangeably -- see :class:`~squidpy.types.Clusterer`.
:class:`~sklearn.base.BaseEstimator` supplies ``get_params``/``set_params``, so every fit
can go to a fresh :func:`~sklearn.base.clone`, and :class:`~sklearn.base.ClusterMixin`
supplies ``fit_predict`` on top of ``fit``.

One label per observation is all :meth:`~squidpy.types.Clusterer.fit_predict` returns, so
two optional fitted attributes carry what a niche flavor needs beyond it:
``niche_columns_`` for further label columns, and ``niche_uns_`` for diagnostics. Neither
is part of the protocol -- a plain scikit-learn clusterer has neither, and the pipeline
just writes its one column.

The stability sweep :class:`_AutoKClusterer` wraps is in :mod:`squidpy.gr._autok`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import anndata as ad
import pandas as pd
import scanpy as sc
from fast_array_utils.types import HasArrayNamespace as Array
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClusterMixin
from spatialdata._logging import logger as logg

from squidpy._docs import d
from squidpy.gr._autok import _gmm, expand_n_clusters, sweep_auto_k, to_uns
from squidpy.types import ClusterAutoKResult, SweepableClusterer

__all__ = ["_LeidenClusterer", "_AutoKClusterer"]


@d.dedent
class _LeidenClusterer(ClusterMixin, BaseEstimator):
    """Leiden partition of a neighborhood graph, at one resolution.

    A wrapper over :func:`scanpy.tl.leiden`, which is handed the graph directly, so
    nothing here builds one and no :class:`~anndata.AnnData` is read: scanpy needs one
    only as the vessel it writes the labels into. ``X`` is therefore a graph, as it is for
    :class:`~sklearn.cluster.SpectralClustering` with ``affinity="precomputed"``, and the
    graph belongs to whatever produced it -- see ``_LatentGraphEmbedder``. One graph, any
    number of resolutions, one clusterer each.

    Parameters
    ----------
    resolution
        Resolution of the partition. Higher gives more, smaller niches.
    %(niche_leiden_backend)s
    random_state
        Seeds the partition.
    """

    def __init__(
        self,
        *,
        resolution: float = 1.0,
        flavor: Literal["igraph", "leidenalg"] = "igraph",
        n_iterations: int = -1,
        random_state: int | None = None,
    ):
        self.resolution = resolution
        self.flavor = flavor
        self.n_iterations = n_iterations
        self.random_state = random_state

    def fit(self, X: csr_matrix, y: None = None) -> _LeidenClusterer:
        """Partition the graph *X*, leaving the labels in ``labels_``."""
        # `sc.tl.leiden` writes into `adata.obs`, and with `adjacency` given that is the
        # only thing it uses the object for -- so an index-only shell is enough
        shell = ad.AnnData(obs=pd.DataFrame(index=[str(i) for i in range(X.shape[0])]))

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
            adjacency=X,
            resolution=self.resolution,
            key_added="niche",
            random_state=self.random_state,
            **kwargs,
        )

        self.labels_ = shell.obs["niche"].to_numpy()
        # scanpy computes it anyway, and it is the one number that says whether this
        # resolution found any structure
        self.modularity_ = shell.uns["niche"]["modularity"]
        return self


@d.dedent
class _AutoKClusterer(ClusterMixin, BaseEstimator):
    """A clusterer whose number of clusters is chosen by the stability of its labeling.

    Every candidate K is fitted ``max_runs`` times and scored by how stably its labeling
    reproduces across runs; ``labels_`` is the labeling of the most stable K.

    Parameters
    ----------
    n_clusters
        Candidate numbers of clusters. A ``(min, max)`` tuple gains a ``+-1`` halo, see
        :func:`~squidpy.gr._autok.expand_n_clusters`.
    clusterer
        The :class:`~squidpy.types.SweepableClusterer` fitted at every K. Defaults to a
        :class:`~sklearn.mixture.GaussianMixture`, which is what CellCharter selects over.
    max_runs
        Maximum number of repetitions per K.
    convergence_tol
        Stop early once the mean stability curve settles, see
        :func:`~squidpy.gr.sweep_auto_k`.
    store_labels
        Also keep the labeling of every fitted K, as ``k{K}`` entries in
        ``niche_columns_``.
    uns_key
        Key of the per-K diagnostics in ``niche_uns_``.
    model_params
        Extra parameters for the default mixture; unused when *clusterer* is given.
    random_state
        Base seed of the sweep. Every fit is seeded from it, its run and its K.
    """

    def __init__(
        self,
        *,
        n_clusters: tuple[int, int] | Sequence[int],
        clusterer: SweepableClusterer | None = None,
        max_runs: int = 10,
        convergence_tol: float = 1e-2,
        store_labels: bool = False,
        uns_key: str = "niche_autok",
        model_params: Mapping[str, Any] | None = None,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.clusterer = clusterer
        self.max_runs = max_runs
        self.convergence_tol = convergence_tol
        self.store_labels = store_labels
        self.uns_key = uns_key
        self.model_params = model_params
        self.random_state = random_state

    def fit(self, X: Array, y: None = None) -> _AutoKClusterer:
        """Sweep the candidate K values over *X*, leaving the selected labeling in ``labels_``."""
        # expanded in `fit`, not `__init__`: a scikit-learn estimator hands back the
        # parameters it was given, so expanding early would make `clone` lose the halo
        candidates = expand_n_clusters(self.n_clusters)
        logg.info(
            f"Selecting the number of clusters over K={candidates} with up to {self.max_runs} runs each "
            f"({len(candidates) * self.max_runs} fits at most)"
        )
        result = sweep_auto_k(
            X,
            candidates,
            clusterer=self.clusterer if self.clusterer is not None else _gmm(self.model_params),
            max_runs=self.max_runs,
            convergence_tol=self.convergence_tol,
            seed=self.random_state,
        )
        logg.info(f"Selected K={result.best_k} after {result.n_runs} runs")

        self.result_: ClusterAutoKResult = result
        self.labels_ = result.labels[result.best_k]
        self.best_k_ = result.best_k
        self.niche_uns_: dict[str, Any] = {self.uns_key: to_uns(result)}
        if self.store_labels:
            self.niche_columns_: dict[str, Array] = {f"k{k}": labeling for k, labeling in result.labels.items()}
        return self
