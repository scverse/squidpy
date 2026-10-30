"""Public entry point for the stability-based selection of the number of clusters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from spatialdata import SpatialData
from spatialdata._logging import logger as logg

from squidpy._docs import d
from squidpy._utils import RNGLike, SeedLike
from squidpy._validators import assert_isinstance, assert_key_in_adata
from squidpy.gr._autok import check_k_clusterer, expand_n_clusters, sweep_auto_k
from squidpy.gr._clusterers import KClusterer
from squidpy.gr._utils import _save_data, extract_adata_if_sdata

__all__ = ["cluster_auto_k"]


@d.dedent
def cluster_auto_k(
    data: AnnData | SpatialData,
    *,
    use_rep: str,
    n_clusters: tuple[int, int] | Sequence[int] = (2, 10),
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    clusterer: KClusterer | None = None,
    rng: SeedLike | RNGLike | None = None,
    store_labels: bool = False,
    key_added: str = "cluster_auto_k",
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Select the number of clusters (K) by clustering stability.

    Every candidate K is fitted ``max_runs`` times, and each K is scored by how similar its
    labeling is to the labeling of the adjacent K across runs (Fowlkes-Mallows). The most
    stable K is the one whose partition is least sensitive to adding a cluster.

    Reimplements the K selection of CellCharter's ``ClusterAutoK``, using
    :class:`~sklearn.mixture.GaussianMixture` for the fits unless *clusterer* says otherwise.

    Parameters
    ----------
    %(adata)s
    use_rep
        Key in :attr:`anndata.AnnData.obsm` holding the representation to cluster, such as the
        embedding :func:`~squidpy.gr.calculate_niche_cellcharter` stores, or ``'X'`` for
        :attr:`~anndata.AnnData.X`.
    n_clusters
        Numbers of clusters to select from: a ``(min, max)`` tuple for every K in that range,
        or a sequence of K values, which need not be consecutive. As in CellCharter, each K
        is scored by its agreement with K-1 and K+1, so those are fitted too. K < 2 cannot
        be scored and is dropped with a warning.
    max_runs
        Maximum number of repetitions per candidate K. Must be at least 2, since a single
        run leaves stability undefined.
    convergence_tol
        Stop early once the mean absolute percentage error between the mean stability curves
        of consecutive runs falls below this value.
    clusterer
        Fitted at every candidate K instead of the default mixture. Any
        :class:`~squidpy.types.KClusterer` will do (scikit-learn's
        :class:`~sklearn.cluster.KMeans`, cuML's, or your own) as long as it takes a
        number of clusters and a ``random_state``, both of which this sets per fit. It is
        cloned for every fit, so the one passed in is never touched.
    %(rng)s
        Seeds every individual fit.
    store_labels
        Also keep the labeling of every fitted K, the selected one included, as
        ``{key_added}_k{K}``. The sweep fits them all anyway, so this is how a runner-up K is
        inspected without refitting.
    key_added
        Name of the labeling added to ``adata.obs``, and of the diagnostics added to
        ``adata.uns``.
    %(copy)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with the additions below; otherwise ``adata``
    is modified in place and ``None`` is returned. Either way it gains the following keys:

        - :attr:`anndata.AnnData.obs` ``['{key_added}']`` - the labeling at the selected K,
          plus one ``['{key_added}_k{K}']`` per fitted K if ``store_labels``. Categories are
          ``'0'`` to ``'K-1'``, including any component left empty.
        - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - a dict with the selected
          ``best_k``, the per-K diagnostics ``table``, the raw ``stability`` values as a
          DataFrame indexed by the scored K, ``n_runs`` and ``converged``.

    To select the number of niches, run this on the embedding
    :func:`~squidpy.gr.calculate_niche_cellcharter` stores, via ``use_rep``. With a
    ``cluster_mask`` there, pass the same subset here as a copy, ``adata[mask].copy()``,
    since results written to a view are lost.

    Notes
    -----
    The sweep costs up to ``max_runs x len(K)`` mixture fits, and keeps the labelings of all
    previous runs in memory, since every run is compared against all of them.
    """
    assert_isinstance(data, (AnnData, SpatialData), name="data")
    assert_isinstance(copy, bool, name="copy")
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)

    assert_isinstance(use_rep, str, name="use_rep")
    if use_rep != "X":
        assert_key_in_adata(orig_adata, use_rep, attr="obsm")

    assert_isinstance(max_runs, int, name="max_runs")
    assert_isinstance(convergence_tol, (float, int), name="convergence_tol")

    X = orig_adata.X if use_rep == "X" else orig_adata.obsm[use_rep]
    if X is None:
        raise ValueError("'adata.X' is None. Pass a representation in 'adata.obsm' via 'use_rep'.")
    if clusterer is None:
        clusterer = GaussianMixture()
    check_k_clusterer(clusterer)
    if issparse(X) and isinstance(clusterer, GaussianMixture | BayesianGaussianMixture):
        raise TypeError(
            f"'{type(clusterer).__name__}' does not support sparse input. Pass a dense representation via "
            "'use_rep', or densify with 'adata.X = adata.X.toarray()'."
        )

    candidates, scored = expand_n_clusters(n_clusters)
    adata = orig_adata.copy() if copy else orig_adata
    logg.info(
        f"Selecting the number of clusters over K={candidates} with up to {max_runs} runs each "
        f"({len(candidates) * max_runs} {type(clusterer).__name__} fits at most)"
    )

    result = sweep_auto_k(
        X,
        candidates,
        scored=scored,
        max_runs=max_runs,
        convergence_tol=convergence_tol,
        clusterer=clusterer,
        rng=rng,
    )
    logg.info(f"Selected K={result.best_k} after {result.n_runs} runs")

    columns = {key_added: result.best_k}
    if store_labels:
        columns |= {f"{key_added}_k{k}": k for k in result.labels}
    for column, k in columns.items():
        # string categories in numeric order, like `sc.tl.leiden`; all K of them, as CellCharter keeps
        labels = pd.Categorical(result.labels[k].astype(str), categories=np.arange(k).astype(str))
        _save_data(
            adata, attr="obs", key=column, data=pd.Series(labels, index=adata.obs_names), prefix=column == key_added
        )
    uns = {key: value for key, value in result._asdict().items() if key != "labels"}  # labels are in obs
    # one column per run comparison; string names, as h5ad stores them
    uns["stability"] = pd.DataFrame(result.stability, index=pd.Index(scored, name="k")).rename(columns=str)
    _save_data(adata, attr="uns", key=key_added, data=uns)

    return adata if copy else None
