"""Public entry point for the stability-based selection of the number of clusters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from anndata import AnnData
from scipy.sparse import issparse
from spatialdata import SpatialData
from spatialdata._logging import logger as logg

from squidpy._docs import d
from squidpy._validators import assert_isinstance, assert_key_in_adata
from squidpy.gr._autok import ClusterAutoKResult, expand_n_clusters, sweep_auto_k
from squidpy.gr._utils import extract_adata_if_sdata

__all__ = ["cluster_auto_k"]


@d.dedent
def cluster_auto_k(
    data: AnnData | SpatialData,
    n_clusters: tuple[int, int] | Sequence[int] = (2, 10),
    *,
    use_rep: str | None = None,
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    model_params: Mapping[str, Any] | None = None,
    seed: int | None = None,
    table_key: str | None = None,
) -> ClusterAutoKResult:
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
        and ``seed`` and are rejected here.
    %(seed)s
        Seeds every individual fit.
    %(table_key)s

    Returns
    -------
    A :class:`~squidpy.gr.ClusterAutoKResult` with the selected ``best_k``, the runner-up
    ``peaks``, the full per-K ``stability`` matrix and the labeling of every fitted K.
    ``data`` is not modified; see :func:`~squidpy.gr.calculate_niche_cellcharter` to run the
    same sweep as part of a niche pipeline and store the diagnostics in ``adata.uns``.

    Notes
    -----
    The sweep costs up to ``max_runs x len(K)`` mixture fits, and keeps the labelings of all
    previous runs in memory, since every run is compared against all of them.
    """
    assert_isinstance(data, (AnnData, SpatialData), name="data")
    adata = extract_adata_if_sdata(data, table_key=table_key)

    if use_rep is not None:
        assert_isinstance(use_rep, str, name="use_rep")
        assert_key_in_adata(adata, use_rep, attr="obsm")

    assert_isinstance(max_runs, int, name="max_runs")
    assert_isinstance(convergence_tol, (float, int), name="convergence_tol")
    if seed is not None:
        assert_isinstance(seed, int, name="seed")

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
        rng=np.random.default_rng(seed),
    )
    logg.info(f"Selected K={result.best_k} after {result.n_runs} runs (peaks at K={result.peaks})")
    return result
