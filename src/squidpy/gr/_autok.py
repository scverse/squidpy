"""Stability-based selection of the number of clusters (K), or of any other clusterer parameter.

Reimplements CellCharter's ``ClusterAutoK`` (https://github.com/CSOgroup/cellcharter), and applies
the same score to a grid of any one parameter, such as a Leiden resolution.
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

from squidpy._utils import RNGLike, SeedLike, legacy_random
from squidpy.gr._clusterers import Clusterer, default_mixture, k_param


class ClusterAutoKResult(NamedTuple):
    """A sweep result."""

    #: Per-value ``stability_mean``, ``stability_std`` and ``fit_loss``; unscored values (the
    #: ``+-1`` halo of a K sweep) have ``NaN`` stability.
    table: pd.DataFrame

    #: Raw similarity values, of shape ``(n_scored, n_comparisons)``. Row ``i`` belongs to
    #: the ``i``-th scored value.
    stability: np.ndarray

    #: The scored value with the highest mean stability.
    best: float

    #: Number of runs actually performed, below ``max_runs`` if the sweep converged.
    n_runs: int

    #: Whether the sweep stopped early because the stability curve had settled.
    converged: bool

    #: Labeling of the best fit (lowest ``fit_loss``) per value, for every fitted value.
    labels: dict[float, np.ndarray]


def expand_n_clusters(n_clusters: tuple[int, int] | Sequence[int]) -> tuple[list[int], list[int]]:
    """Resolve the requested K values into the K values to fit and the K values to score.

    As CellCharter: a ``(min, max)`` tuple is that range, else the listed K. Each K needs K-1 and
    K+1 fitted. K < 2 is dropped with a warning.

    Returns
    -------
    The K values to fit and the K values to score, both sorted.
    """
    if isinstance(n_clusters, tuple):
        if len(n_clusters) != 2:
            raise ValueError(f"'values' as a tuple must be (min, max), got {n_clusters!r}")
        low, high = n_clusters
        if low > high:
            raise ValueError(f"'values' must be (min, max) with min <= max, got {n_clusters!r}")
        requested = set(range(low, high + 1))
    elif any(k != int(k) for k in n_clusters):
        raise ValueError(f"'values' must be whole numbers of clusters, got {n_clusters!r}")
    else:
        requested = {int(k) for k in n_clusters}

    scored = sorted(k for k in requested if k >= 2)
    if len(scored) < len(requested):
        warnings.warn("stability cannot be computed for K < 2, ignoring those values", UserWarning, stacklevel=3)
    if not scored:
        raise ValueError(f"'values' must contain at least one value >= 2, got {n_clusters!r}")
    return sorted({k + d for k in scored for d in (-1, 0, 1)}), scored


def mirror_stability(blocks: Sequence[Sequence[float]]) -> np.ndarray:
    """Fold one-directional similarities into a per-K stability matrix.

    Each block, one pair of runs, compares each K against K+1. Mirroring also gives each
    interior K its ``(K-1, K)`` comparison.

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


def _stability_frame(
    values: Sequence[float],
    interior: Sequence[float],
    stability: np.ndarray,
    fit_loss: Mapping[float, float],
    name: str,
) -> pd.DataFrame:
    """Per-value diagnostics indexed by *name*, carrying ``NaN`` stability on the unscored rows."""
    per_k_mean = dict(zip(interior, stability.mean(axis=1), strict=True))
    per_k_std = dict(zip(interior, stability.std(axis=1), strict=True))
    return pd.DataFrame(
        {
            "stability_mean": [per_k_mean.get(v, np.nan) for v in values],
            "stability_std": [per_k_std.get(v, np.nan) for v in values],
            "fit_loss": [fit_loss[v] for v in values],
        },
        index=pd.Index(values, name=name),
    )


def check_sweep_clusterer(clusterer: Clusterer, param: str) -> None:
    """Reject a *clusterer* that cannot be re-fitted at every value of *param*."""
    if not isinstance(clusterer, Clusterer):
        raise TypeError(f"{type(clusterer).__name__} must implement fit_predict, get_params and set_params")
    params = clusterer.get_params()
    if param not in params:
        raise TypeError(f"{type(clusterer).__name__} has no {param!r} to sweep")
    if "random_state" not in params:
        raise TypeError(f"{type(clusterer).__name__} has no 'random_state', so its runs would not differ")


def _fit_once(
    clusterer: Clusterer, name: str, X: Array, value: float, random_state: int, *, is_k: bool
) -> tuple[np.ndarray, float]:
    """One run: a fresh clone of *clusterer* at ``name=value``, seeded by ``random_state``."""
    est = clone(clusterer).set_params(**{name: value, "random_state": random_state})
    labels = np.asarray(est.fit_predict(X))
    # e.g. -1 for noise, labels counted from 1, or strings
    allowed = f"0..{value - 1}" if is_k else "0 or above"
    if not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or (is_k and labels.max() >= value):
        raise TypeError(f"{type(est).__name__} returned labels that are not integers in {allowed} at {name}={value}")
    labels = labels.astype(np.uint32, copy=False)  # every run is kept: store narrow
    # picks the kept labeling per value; NaN (no `score`) keeps the first run
    fit_loss = -float(est.score(X)) if hasattr(est, "score") else float("nan")
    return labels, fit_loss


def sweep_auto_k(
    X: Array,
    values: Sequence[float],
    *,
    param: str,
    scored: Sequence[float] | None = None,
    clusterer: Clusterer | None = None,
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    rng: SeedLike | RNGLike | None = None,
) -> ClusterAutoKResult:
    """Fit every value repeatedly and score each by the stability of its labeling.

    Parameters
    ----------
    X
        Feature matrix with observations as rows.
    values
        The values to fit, in order: each scored value is compared with its neighbours here.
        For K, already resolved by :func:`expand_n_clusters`.
    param
        The parameter of *clusterer* to sweep. Its number of clusters (see :func:`k_param`)
        gets K's handling: seeded by K, and labels checked to be below K.
    scored
        The values to score, each of which needs a neighbour on both sides in *values*.
        Defaults to every value but the first and the last.
    clusterer
        Cloned per fit with *param* and ``random_state`` set. Defaults to
        :class:`~sklearn.mixture.GaussianMixture` with ``reg_covar=1e-4``.
    max_runs
        Maximum repetitions per value, at least 2. A run with a failed fit is skipped.
    convergence_tol
        Stop early once the mean absolute percentage error between the mean stability
        curves of two consecutive runs falls below this value.
    rng
        Source of randomness; ``None`` draws from OS entropy. Each fit is seeded from
        ``(base, run, K)``, so adding a K leaves the others untouched; any other parameter by
        its position in *values*.

    Returns
    -------
    The sweep result; see :class:`ClusterAutoKResult`.
    """
    if max_runs <= 1:
        raise ValueError(f"stability needs at least 2 runs to compare, got max_runs={max_runs}")

    if clusterer is None:
        clusterer = default_mixture()
    check_sweep_clusterer(clusterer, param)
    is_k = param == k_param(clusterer)

    base = legacy_random(np.random.default_rng(rng))

    ks = list(values)
    if is_k and max(ks) > X.shape[0]:
        raise ValueError(f"{param}={max(ks)} is more clusters than the {X.shape[0]} observations")
    # adjacent pairs rather than `k + 1` arithmetic, so non-contiguous grids work
    pairs = list(zip(ks[:-1], ks[1:], strict=True))
    interior = ks[1:-1]
    scored = interior if scored is None else list(scored)
    if not set(scored) <= set(interior):
        raise ValueError(f"every scored value needs a fitted neighbour on both sides, got {scored!r} for {ks!r}")
    rows = [interior.index(k) for k in scored]

    labels_per_run: list[dict[float, np.ndarray]] = []
    best_loss: dict[float, float] = {}
    best_labels: dict[float, np.ndarray] = {}
    blocks: list[list[float]] = []
    previous_curve: np.ndarray | None = None
    converged = False
    error: ValueError | None = None

    for run in range(max_runs):
        run_labels: dict[float, np.ndarray] = {}
        for i, k in enumerate(ks):
            random_state = int(np.random.SeedSequence([base, run, k if is_k else i]).generate_state(1)[0])
            try:
                labels, fit_loss = _fit_once(clusterer, param, X, k, random_state, is_k=is_k)
            except ValueError as err:  # e.g. a mixture component that collapsed at this seed
                warnings.warn(f"skipping run {run}: the fit at {param}={k} failed: {err}", UserWarning, stacklevel=3)
                error = err
                break
            run_labels[k] = labels
            if k not in best_loss or fit_loss < best_loss[k]:
                best_loss[k], best_labels[k] = fit_loss, labels
        if len(run_labels) < len(ks):
            continue

        if labels_per_run:
            # one direction; mirror_stability symmetrizes
            blocks.extend(_score_block(run_labels, stored, pairs, fowlkes_mallows_score) for stored in labels_per_run)
            curve = mirror_stability(blocks)[rows].mean(axis=1)
            if previous_curve is not None and mean_absolute_percentage_error(previous_curve, curve) < convergence_tol:
                labels_per_run.append(run_labels)
                converged = True
                break
            previous_curve = curve

        labels_per_run.append(run_labels)

    if len(labels_per_run) < 2:
        msg = f"only {len(labels_per_run)} of {max_runs} runs fitted every value, and stability needs 2"
        raise ValueError(msg) from error
    stability = mirror_stability(blocks)[rows]
    table = _stability_frame(ks, scored, stability, best_loss, param)
    return ClusterAutoKResult(
        table=table,
        stability=stability,
        best=table["stability_mean"].idxmax().item(),
        labels=best_labels,
        n_runs=len(labels_per_run),
        converged=converged,
    )
