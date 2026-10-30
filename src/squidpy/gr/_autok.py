"""Stability-based selection of the number of clusters (K).

Reimplements CellCharter's ``ClusterAutoK`` (https://github.com/CSOgroup/cellcharter).

Anything needing :class:`~anndata.AnnData` belongs in :mod:`squidpy.gr._cluster_auto_k`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import numpy as np
import pandas as pd
from fast_array_utils.types import HasArrayNamespace as Array
from sklearn.base import clone
from sklearn.metrics import fowlkes_mallows_score, mean_absolute_percentage_error
from sklearn.mixture import GaussianMixture

from squidpy._utils import RNGLike, SeedLike, legacy_random
from squidpy.gr._clusterers import KClusterer

# The two spellings of the number-of-clusters parameter, in the order they are tried.
# `n_clusters` first: SpectralClustering takes both, and there `n_components` is the
# width of the spectral embedding, not the number of clusters.
_K_PARAMS = ("n_clusters", "n_components")


class ClusterAutoKResult(NamedTuple):
    """A sweep result."""

    #: Per-K diagnostics indexed by K, with the columns ``stability_mean``, ``stability_std``
    #: and ``fit_loss``. Every fitted K has a row, but the ``+-1`` halo is never scored, so its
    #: stability is ``NaN``.
    table: pd.DataFrame

    #: Raw similarity values, of shape ``(n_scored_k, n_comparisons)``. Row ``i`` belongs to
    #: the ``i``-th scored K.
    stability: np.ndarray

    #: The scored K with the highest mean stability.
    best_k: int

    #: Number of runs actually performed, below ``max_runs`` if the sweep converged.
    n_runs: int

    #: Whether the sweep stopped early because the stability curve had settled.
    converged: bool

    #: Labeling of the best fit (lowest ``fit_loss``) per K, for every fitted K.
    labels: dict[int, np.ndarray]


def expand_n_clusters(n_clusters: tuple[int, int] | Sequence[int]) -> tuple[list[int], list[int]]:
    """Resolve the requested K values into the K values to fit and the K values to score.

    As CellCharter's ``ClusterAutoK`` does: a ``(min, max)`` tuple means every K in that
    range, any other sequence the listed K values. Each requested K is scored by its
    agreement with K-1 and K+1, so those are fitted too. K < 2 cannot be scored and is
    dropped with a warning.

    Returns
    -------
    The K values to fit and the K values to score, both sorted.
    """
    if isinstance(n_clusters, tuple):
        if len(n_clusters) != 2:
            raise ValueError(f"'n_clusters' as a tuple must be (min, max), got {n_clusters!r}")
        low, high = n_clusters
        if low > high:
            raise ValueError(f"'n_clusters' must be (min, max) with min <= max, got {n_clusters!r}")
        requested = set(range(low, high + 1))
    else:
        requested = {int(k) for k in n_clusters}

    scored = sorted(k for k in requested if k >= 2)
    if len(scored) < len(requested):
        warnings.warn("stability cannot be computed for K < 2, ignoring those values", UserWarning, stacklevel=3)
    if not scored:
        raise ValueError(f"'n_clusters' must contain at least one value >= 2, got {n_clusters!r}")
    return sorted({k + d for k in scored for d in (-1, 0, 1)}), scored


def mirror_stability(blocks: Sequence[Sequence[float]]) -> np.ndarray:
    """Fold one-directional similarities into a per-K stability matrix.

    Each block holds the similarities of one pair of runs, scored over adjacent K pairs:
    entry ``i`` compares K ``ks[i]`` of one run against ``ks[i + 1]`` of the other. That
    comparison is one-directional, so a given K is only ever compared *upwards*. Mirroring
    repairs the asymmetry by giving each interior K both the ``(K, K+1)`` and the
    ``(K-1, K)`` comparisons.

    Returns
    -------
    Array of shape ``(len(ks) - 2, 2 * len(blocks))``, with row ``i`` belonging to
    ``ks[i + 1]``, i.e. to the interior K values only.
    """
    per_pair = list(zip(*blocks, strict=True))  # index by K pair instead of by run pair
    return np.array([list(per_pair[i]) + list(per_pair[i - 1]) for i in range(1, len(per_pair))])


def _score_block(
    run_a: Mapping[int, Array],
    run_b: Mapping[int, Array],
    pairs: Sequence[tuple[int, int]],
    score_fn: Callable[[Array, Array], float],
) -> list[float]:
    """Similarity of ``run_a`` at each K against ``run_b`` at the next K, one block per run pair."""
    return [score_fn(run_a[low], run_b[high]) for low, high in pairs]


def _stability_frame(n_clusters: Sequence[int], interior: Sequence[int], stability: np.ndarray) -> pd.DataFrame:
    """Per-K stability diagnostics indexed by K, carrying ``NaN`` on the unscored halo rows."""
    per_k_mean = dict(zip(interior, stability.mean(axis=1), strict=True))
    per_k_std = dict(zip(interior, stability.std(axis=1), strict=True))
    return pd.DataFrame(
        {
            "stability_mean": [per_k_mean.get(k, np.nan) for k in n_clusters],
            "stability_std": [per_k_std.get(k, np.nan) for k in n_clusters],
        },
        index=pd.Index(n_clusters, name="k"),
    )


def check_k_clusterer(clusterer: KClusterer) -> str:
    """Name of *clusterer*'s number-of-clusters parameter, rejecting what cannot be swept."""
    if not isinstance(clusterer, KClusterer):
        raise TypeError(
            f"{type(clusterer).__name__} is not a clusterer squidpy can re-fit: that needs "
            "'fit_predict' to assign one label per observation, plus 'get_params' and "
            "'set_params' to be re-fitted with a new number of clusters and seed"
        )
    params = clusterer.get_params()
    k_param = next((name for name in _K_PARAMS if name in params), None)
    if k_param is None:
        raise ValueError(
            f"{type(clusterer).__name__} takes neither 'n_components' nor 'n_clusters', so there "
            "is no number of clusters to sweep"
        )
    if "random_state" not in params:
        raise ValueError(
            f"{type(clusterer).__name__} takes no 'random_state', so every run of a K returns the "
            "same labeling and there is nothing to compare across runs"
        )
    return k_param


def _fit_once(clusterer: KClusterer, k_param: str, X: Array, k: int, random_state: int) -> tuple[np.ndarray, float]:
    """One run: a fresh clone of *clusterer* at ``k``, seeded by ``random_state``."""
    est = clone(clusterer).set_params(**{k_param: k, "random_state": random_state})
    try:
        # transductive clusterers only expose `fit_predict`
        labels = est.fit(X).predict(X) if hasattr(est, "predict") else est.fit_predict(X)
    except ValueError as err:  # a failed fit otherwise aborts the whole sweep opaquely
        hint = (
            " Increase the mixture's 'reg_covar' (e.g. clusterer=GaussianMixture(reg_covar=1e-4)), "
            "pass float64 data, or request a smaller range of K values."
            if "ill-defined empirical covariance" in str(err)
            else ""
        )
        raise ValueError(
            f"the fit at K={k} failed (K-1 and K+1 of every requested K are fitted too): {err}{hint}"
        ) from err
    # every run's labeling is kept for the whole sweep, so store them narrow
    labels = np.asarray(labels)
    if labels.min() < 0 or labels.max() >= k:  # e.g. -1 for noise, or labels counted from 1
        raise ValueError(f"{type(est).__name__} returned labels outside 0..{k - 1} at K={k}")
    labels = labels.astype(np.uint32, copy=False)
    # only picks which run's labeling is kept per K: negative log-likelihood for a mixture,
    # inertia for KMeans, NaN (keep the first run) for clusterers without `score`
    fit_loss = -float(est.score(X)) if hasattr(est, "score") else float("nan")
    return labels, fit_loss


def sweep_auto_k(
    X: Array,
    n_clusters: Sequence[int],
    *,
    scored: Sequence[int] | None = None,
    clusterer: KClusterer | None = None,
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    rng: SeedLike | RNGLike | None = None,
) -> ClusterAutoKResult:
    """Fit every K repeatedly and score each K by the stability of its labeling.

    Parameters
    ----------
    X
        Feature matrix with observations as rows.
    n_clusters
        The K values to fit, already resolved by :func:`expand_n_clusters`.
    scored
        The K values to score, each of which needs its K-1 and K+1 in *n_clusters*.
        Defaults to every K but the first and the last.
    clusterer
        A ``KClusterer``, whose number of clusters and
        ``random_state`` this sets per fit, so its own are ignored. Defaults to
        :class:`~sklearn.mixture.GaussianMixture`.

        Of scikit-learn's own clusterers, that includes
        :class:`~sklearn.mixture.GaussianMixture`,
        :class:`~sklearn.cluster.KMeans`, :class:`~sklearn.cluster.MiniBatchKMeans`,
        :class:`~sklearn.cluster.BisectingKMeans` and
        :class:`~sklearn.cluster.SpectralClustering`; anything else matching the contract
        works too, scikit-learn or not.
    max_runs
        Maximum number of repetitions per K. Must be at least 2: a single run leaves
        stability undefined, since there is nothing to compare against.
    convergence_tol
        Stop early once the mean absolute percentage error between the mean stability
        curves of two consecutive runs falls below this value.
    rng
        Source of randomness; ``None`` draws from OS entropy. Each fit is seeded from
        ``(base, run, K)``, so adding a K leaves the others' fits untouched. CellCharter
        instead shares one seed across all K of a run; only cross-run comparisons enter the
        score, so this does not change the method.

    Returns
    -------
    The sweep result; see :class:`ClusterAutoKResult`.
    """
    if max_runs <= 1:
        raise ValueError(f"stability needs at least 2 runs to compare, got max_runs={max_runs}")

    if clusterer is None:
        clusterer = GaussianMixture()
    k_param = check_k_clusterer(clusterer)

    base = legacy_random(np.random.default_rng(rng))

    ks = list(n_clusters)
    # adjacent pairs rather than `k + 1` arithmetic, so non-contiguous K lists work
    pairs = list(zip(ks[:-1], ks[1:], strict=True))
    interior = ks[1:-1]
    scored = interior if scored is None else list(scored)
    if not set(scored) <= set(interior):
        raise ValueError(f"every scored K needs a fitted neighbour on both sides, got {scored!r} for {ks!r}")
    rows = [interior.index(k) for k in scored]

    labels_per_run: list[dict[int, np.ndarray]] = []
    best_loss: dict[int, float] = {}
    best_labels: dict[int, np.ndarray] = {}
    blocks: list[list[float]] = []
    previous_curve: np.ndarray | None = None
    converged = False

    for run in range(max_runs):
        run_labels: dict[int, np.ndarray] = {}
        for k in ks:
            random_state = int(np.random.SeedSequence([base, run, k]).generate_state(1)[0])
            labels, fit_loss = _fit_once(clusterer, k_param, X, k, random_state)
            run_labels[k] = labels
            if k not in best_loss or fit_loss < best_loss[k]:
                best_loss[k], best_labels[k] = fit_loss, labels

        if labels_per_run:
            # this run's K against every stored run's K+1, one direction only, which
            # `mirror_stability` then folds into a symmetric per-K score
            blocks.extend(_score_block(run_labels, stored, pairs, fowlkes_mallows_score) for stored in labels_per_run)
            curve = mirror_stability(blocks)[rows].mean(axis=1)
            if previous_curve is not None and mean_absolute_percentage_error(previous_curve, curve) < convergence_tol:
                labels_per_run.append(run_labels)
                converged = True
                break
            previous_curve = curve

        labels_per_run.append(run_labels)

    stability = mirror_stability(blocks)[rows]
    table = _stability_frame(ks, scored, stability)
    table.insert(2, "fit_loss", [best_loss[k] for k in ks])
    return ClusterAutoKResult(
        table=table,
        stability=stability,
        best_k=int(table["stability_mean"].idxmax()),
        labels=best_labels,
        n_runs=len(labels_per_run),
        converged=converged,
    )
