"""Public entry point for the stability-based selection of the number of clusters."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.metrics import fowlkes_mallows_score
from spatialdata import SpatialData
from spatialdata._logging import logger as logg

from squidpy._docs import d
from squidpy._utils import RNGLike, SeedLike
from squidpy._validators import assert_isinstance, assert_key_in_adata
from squidpy.gr._autok import _stability_frame, expand_n_clusters, sweep_auto_k, to_uns
from squidpy.gr._autok import cluster_stability as _cluster_stability
from squidpy.gr._utils import _save_data, extract_adata_if_sdata

__all__ = ["cluster_auto_k", "cluster_stability"]


@d.dedent
def cluster_auto_k(
    data: AnnData | SpatialData,
    n_clusters: tuple[int, int] | Sequence[int] = (2, 10),
    *,
    use_rep: str | None = None,
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    model_params: Mapping[str, Any] | None = None,
    rng: SeedLike | RNGLike | None = None,
    keep_all_labels: bool = False,
    key_added: str = "cluster_auto_k",
    inplace: bool = True,
    table_key: str | None = None,
) -> AnnData | None:
    """Select the number of clusters (K) of a Gaussian mixture by clustering stability.

    Every candidate K is fitted ``max_runs`` times, and each K is scored by how similar its
    labeling is to the labeling of the adjacent K across runs (Fowlkes-Mallows). The most
    stable K is the one whose partition is least sensitive to adding a cluster.

    Reimplements the K selection of CellCharter's ``ClusterAutoK``, using
    :class:`~sklearn.mixture.GaussianMixture` for the fits.

    Parameters
    ----------
    %(adata)s
    n_clusters
        Candidate numbers of clusters. A ``(min, max)`` tuple also fits ``min - 1`` and
        ``max + 1``, because stability is only defined on interior K values and the bounds
        would otherwise not be selectable. A sequence of candidates is taken literally and
        gets no such halo, so its first and last entries can never be selected.
    use_rep
        Key in ``adata.obsm`` holding the representation to cluster. If ``None``, uses
        ``adata.X``.
    max_runs
        Maximum number of repetitions per candidate K. Must be at least 2, since a single
        run leaves stability undefined.
    convergence_tol
        Stop early once the mean absolute percentage error between the mean stability curves
        of consecutive runs falls below this value.
    model_params
        Extra keyword arguments for :class:`~sklearn.mixture.GaussianMixture`. The mapping is
        never modified. ``n_components`` and ``random_state`` are controlled by ``n_clusters``
        and ``rng`` and are rejected here.
    %(rng)s
        Seeds every individual fit.
    keep_all_labels
        Also keep the labeling of every *other* fitted K, as ``{key_added}_k{K}``. The sweep
        fits them all anyway, so this is how a runner-up K is inspected without refitting.
    key_added
        Name of the labeling added to ``adata.obs``, and of the diagnostics added to
        ``adata.uns``.
    %(niche_inplace)s
    %(table_key)s

    Returns
    -------
    If ``inplace = True``, modifies ``adata`` in place and returns ``None``. Otherwise, returns
    a copy of ``adata`` with the same additions. Either way it gains the following keys:

        - :attr:`anndata.AnnData.obs` ``['{key_added}']`` - the labeling at the selected K,
          plus one ``['{key_added}_k{K}']`` per fitted K if ``keep_all_labels``.
        - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - a :class:`~squidpy.gr.ClusterAutoKUns`
          with the selected ``best_k``, the per-K diagnostics ``table`` and the full
          ``stability`` matrix.

    See :func:`~squidpy.gr.calculate_niche_cellcharter` to run the same sweep as part of a
    niche pipeline.

    Notes
    -----
    The sweep costs up to ``max_runs x len(K)`` mixture fits, and keeps the labelings of all
    previous runs in memory, since every run is compared against all of them.
    """
    assert_isinstance(data, (AnnData, SpatialData), name="data")
    assert_isinstance(inplace, bool, name="inplace")
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)
    adata = orig_adata if inplace else orig_adata.copy()

    if use_rep is not None:
        assert_isinstance(use_rep, str, name="use_rep")
        assert_key_in_adata(adata, use_rep, attr="obsm")

    assert_isinstance(max_runs, int, name="max_runs")
    assert_isinstance(convergence_tol, (float, int), name="convergence_tol")

    X = adata.obsm[use_rep] if use_rep is not None else adata.X
    if issparse(X):
        raise TypeError(
            "'GaussianMixture' does not support sparse input. Pass a dense representation via "
            "'use_rep', or densify with 'adata.X = adata.X.toarray()'."
        )

    candidates = expand_n_clusters(n_clusters)
    logg.info(
        f"Selecting the number of clusters over K={candidates} with up to {max_runs} runs each "
        f"({len(candidates) * max_runs} mixture fits at most)"
    )

    result = sweep_auto_k(
        X,
        candidates,
        max_runs=max_runs,
        convergence_tol=convergence_tol,
        model_params=model_params,
        rng=np.random.default_rng(rng),
    )
    logg.info(f"Selected K={result['best_k']} after {result['n_runs']} runs")

    # the selected K is the answer, so it gets the bare `key_added`; the rest are suffixed,
    # which keeps the primary column name predictable for downstream code
    labels = pd.DataFrame({key_added: pd.Categorical(result["labels"][result["best_k"]])}, index=adata.obs_names)
    if keep_all_labels:
        for k, labeling in result["labels"].items():
            labels[f"{key_added}_k{k}"] = pd.Categorical(labeling)

    for column in labels:
        _save_data(adata, attr="obs", key=column, data=labels[column], prefix=column == key_added)
    _save_data(adata, attr="uns", key=key_added, data=to_uns(result))

    return None if inplace else adata


@d.dedent
def cluster_stability(
    data: AnnData | SpatialData,
    cluster_keys: Mapping[int, Sequence[str]],
    *,
    score_fn: Callable[[Any, Any], float] = fowlkes_mallows_score,
    key_added: str = "cluster_stability",
    inplace: bool = True,
    table_key: str | None = None,
) -> AnnData | None:
    """Score existing clusterings by how stably each number of clusters reproduces across runs.

    Where :func:`~squidpy.gr.cluster_auto_k` fits the mixtures itself, this reads labelings that
    are already in ``adata.obs``, so they may come from any clusterer as long as every K was run
    repeatedly. Each K is scored by how similar its labeling is to the labeling of the adjacent K
    across runs; see :func:`~squidpy.gr.cluster_auto_k` for the reasoning.

    Parameters
    ----------
    %(adata)s
    cluster_keys
        Mapping of a number of clusters to the ``adata.obs`` columns holding that K's labelings,
        one column per run. Every K needs the same number of runs and at least two of them, and
        at least 3 K values are needed since stability is only defined on interior K. The lowest
        and highest K are therefore a halo: compared against, but never selectable.
    score_fn
        Similarity of two labelings. Any ``(labels_true, labels_pred) -> float`` works, e.g.
        :func:`~sklearn.metrics.adjusted_rand_score`.
    key_added
        Key in ``adata.uns`` under which the diagnostics are stored.
    %(niche_inplace)s
    %(table_key)s

    Returns
    -------
    If ``inplace = True``, modifies ``adata`` in place and returns ``None``. Otherwise, returns
    a copy of ``adata`` with the same addition. Either way it gains the following key:

        - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - per-K stability indexed by K, with
          ``NaN`` on the two halo rows that are never scored. The most stable K is
          ``df["stability_mean"].idxmax()``.
    """
    assert_isinstance(data, (AnnData, SpatialData), name="data")
    assert_isinstance(inplace, bool, name="inplace")
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)
    adata = orig_adata if inplace else orig_adata.copy()

    for columns in cluster_keys.values():
        for column in columns:
            assert_key_in_adata(adata, column, attr="obs")

    labels = {int(k): [adata.obs[c].to_numpy() for c in columns] for k, columns in cluster_keys.items()}
    interior, stability = _cluster_stability(labels, score_fn=score_fn)
    logg.info(f"Scored the stability of K={sorted(labels)} over {len(next(iter(labels.values())))} runs each")
    df = _stability_frame(sorted(labels), interior, stability)

    _save_data(adata, attr="uns", key=key_added, data=df)

    return None if inplace else adata
