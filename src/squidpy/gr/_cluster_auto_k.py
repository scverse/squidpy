"""Public entry point for the stability-based selection of the number of clusters."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.mixture import GaussianMixture
from spatialdata import SpatialData
from spatialdata._logging import logger as logg

from squidpy._docs import d
from squidpy._utils import RNGLike, SeedLike
from squidpy._validators import assert_isinstance, assert_key_in_adata
from squidpy.gr._autok import expand_n_clusters, sweep_auto_k
from squidpy.gr._clusterers import Clusterer, k_param
from squidpy.gr._utils import _save_data, extract_adata_if_sdata

__all__ = ["cluster_auto_k"]


@d.dedent
def cluster_auto_k(
    data: AnnData | SpatialData,
    *,
    use_rep: str,
    param: str,
    values: tuple[int, int] | Sequence[float],
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    clusterer: Clusterer | None = None,
    rng: SeedLike | RNGLike | None = None,
    store_labels: bool = False,
    key_added: str = "cluster_auto_k",
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Select the number of clusters (K), or another clusterer parameter, by clustering stability.

    Every candidate value is fitted ``max_runs`` times, and each is scored by how similar its
    labeling is to the labeling of the adjacent values across runs (Fowlkes-Mallows). The most
    stable K is the one whose partition is least sensitive to adding a cluster.

    For K, this reimplements the K selection of CellCharter's ``ClusterAutoK``::

        sq.gr.cluster_auto_k(adata, use_rep="niche_embedding", param="n_components", values=(2, 10))

    The same score selects any other parameter of *clusterer*, such as the resolution of
    :class:`~squidpy.gr.LeidenClusterer`.

    Parameters
    ----------
    %(adata)s
    use_rep
        Key in :attr:`anndata.AnnData.obsm` holding the representation to cluster, such as the
        embedding :func:`~squidpy.gr.calculate_niche_cellcharter` stores, or ``'X'`` for
        :attr:`~anndata.AnnData.X`.
    param
        The parameter of *clusterer* to select: its number of clusters, ``'n_components'`` for
        the default :class:`~sklearn.mixture.GaussianMixture` and ``'n_clusters'`` for
        :class:`~sklearn.cluster.KMeans`, or any other, such as ``'resolution'``.
    values
        The values of *param* to select from. For the number of clusters, as in CellCharter: a
        ``(min, max)`` tuple for every K in that range, or a sequence of K values, which need
        not be consecutive; each K is scored by its agreement with K-1 and K+1, so those are
        fitted too, and K < 2 cannot be scored and is dropped with a warning. For any other
        parameter, at least 3 values, each scored by its agreement with its neighbours once
        sorted, so the smallest and the largest are fitted but not scored.
    max_runs
        Maximum number of repetitions per candidate K. Must be at least 2, since a single
        run leaves stability undefined.
    convergence_tol
        Stop early once the mean absolute percentage error between the mean stability curves
        of consecutive runs falls below this value.
    clusterer
        Fitted at every candidate instead of the default mixture, as a template: it is cloned
        for every fit, with *param* and ``random_state`` set, so the one passed in is never
        touched and its own values of those are ignored. Any :class:`~squidpy.types.Clusterer`
        taking *param* and a ``random_state`` will do: scikit-learn's, cuML's, or your own.
    %(rng)s
        Seeds every individual fit.
    store_labels
        Also keep the labeling of every fitted value, the selected one included, as
        ``{key_added}_{param}={value}``, e.g. ``cluster_auto_k_n_components=5``. The sweep fits
        them all anyway, so this is how a runner-up is inspected without refitting.
    key_added
        Name of the labeling added to ``adata.obs``, and of the diagnostics added to
        ``adata.uns``.
    %(copy)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with the additions below; otherwise ``adata``
    is modified in place and ``None`` is returned. Either way it gains the following keys:

        - :attr:`anndata.AnnData.obs` ``['{key_added}']`` - the labeling at the selected value,
          plus one ``['{key_added}_{param}={value}']`` per fitted value if ``store_labels``.
          Categories are ``'0'`` to ``'K-1'``, including any component left empty.
        - :attr:`anndata.AnnData.uns` ``['{key_added}']`` - a dict with the selected value
          ``best``, the per-value diagnostics ``table`` indexed by the swept parameter, the raw
          ``stability`` values as a DataFrame indexed by the scored values, ``n_runs`` and
          ``converged``.

    To select the number of niches, run this on the embedding
    :func:`~squidpy.gr.calculate_niche_cellcharter` stores, via ``use_rep``. With a
    ``cluster_mask`` there, pass the same subset here as a copy, ``adata[mask].copy()``,
    since results written to a view are not written back to the parent ``adata``.

    Notes
    -----
    The sweep costs up to ``max_runs x len(values)`` fits, and keeps the labelings of all
    previous runs in memory, since every run is compared against all of them.
    :class:`~squidpy.gr.LeidenClusterer` rebuilds the neighbourhood graph on every fit.
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
    is_k = param == k_param(clusterer)
    if is_k:
        candidates, scored = expand_n_clusters(values)
    elif len(set(values)) < 3:
        raise ValueError(f"'values' needs at least 3 distinct values of {param!r} to score one, got {values!r}")
    else:
        candidates = sorted(set(values))
        scored = candidates[1:-1]
    logg.info(
        f"Selecting {param} over {candidates} with up to {max_runs} runs each "
        f"({len(candidates) * max_runs} fits at most)"
    )

    result = sweep_auto_k(
        X,
        candidates,
        param=param,
        scored=scored,
        max_runs=max_runs,
        convergence_tol=convergence_tol,
        clusterer=clusterer,
        rng=rng,
    )
    logg.info(f"Selected {param}={result.best} after {result.n_runs} runs")

    # copied only now, so a request the sweep rejects never pays for it
    adata = orig_adata.copy() if copy else orig_adata
    columns = {key_added: result.best}
    if store_labels:
        columns |= {f"{key_added}_{param}={v}": v for v in result.labels}
    for column, v in columns.items():
        codes = result.labels[v]
        # string categories in numeric order, like `sc.tl.leiden`; all K of them, as CellCharter keeps
        n_categories = v if is_k else codes.max() + 1
        labels = pd.Categorical(codes.astype(str), categories=np.arange(n_categories).astype(str))
        _save_data(
            adata, attr="obs", key=column, data=pd.Series(labels, index=adata.obs_names), prefix=column == key_added
        )
    uns = {key: value for key, value in result._asdict().items() if key != "labels"}  # labels are in obs
    # one column per run comparison; string names, as h5ad stores them
    uns["stability"] = pd.DataFrame(result.stability, index=pd.Index(scored, name=param)).rename(columns=str)
    _save_data(adata, attr="uns", key=key_added, data=uns)

    return adata if copy else None
