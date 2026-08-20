"""Stability-based selection of the number of clusters (K) for a Gaussian mixture.

Reimplements the K-selection procedure of CellCharter's ``ClusterAutoK``
(https://github.com/CSOgroup/cellcharter), which fits every candidate K repeatedly and
scores each K by how stably its labeling reproduces across runs.

This module deliberately imports nothing from :mod:`squidpy`, so that downstream projects
(e.g. ``rapids_singlecell``, which vendors squidpy helpers by copying them) can take it as a
single file. Keep it that way: anything needing :class:`~anndata.AnnData`, squidpy's docstring
machinery or its validators belongs in :mod:`squidpy.gr._cluster_auto_k` instead.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.metrics import fowlkes_mallows_score, mean_absolute_percentage_error
from sklearn.mixture import GaussianMixture

__all__ = [
    "ClusterAutoKResult",
    "ClusterAutoKUns",
    "to_uns",
    "cluster_stability",
    "expand_n_clusters",
    "mirror_stability",
    "sweep_auto_k",
]

# `best_k` is a function of run-to-run variability, and therefore of the initialization.
# Pinned so that the selected K does not silently change with a scikit-learn default.
DEFAULT_INIT_PARAMS = "random_from_data"


def expand_n_clusters(n_clusters: tuple[int, int] | Sequence[int]) -> list[int]:
    """Resolve the requested K values into the list of K values to fit.

    A ``(min, max)`` tuple is expanded to ``range(max(1, min - 1), max + 2)``: stability is
    only defined on interior K values (see :func:`mirror_stability`), so the bounds need a
    +-1 halo to be selectable themselves.

    Any other sequence is taken literally and gets **no** halo, so its first and last entries
    can never be selected -- ``[2, ..., 10]`` can only yield a best K in ``3..9``.
    """
    if isinstance(n_clusters, tuple):
        if len(n_clusters) != 2:
            raise ValueError(f"'n_clusters' as a tuple must be (min, max), got {n_clusters!r}")
        low, high = n_clusters
        if low > high:
            raise ValueError(f"'n_clusters' must be (min, max) with min <= max, got {n_clusters!r}")
        candidates = list(range(max(1, low - 1), high + 2))
    else:
        candidates = list(n_clusters)

    if any(k < 1 for k in candidates):
        raise ValueError(f"every value in 'n_clusters' must be at least 1, got {candidates!r}")
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"'n_clusters' must not contain duplicates, got {candidates!r}")
    if len(candidates) < 3:
        raise ValueError(
            f"stability is only defined on interior K values, so at least 3 K values are needed, got {candidates!r}. "
            "Pass a (min, max) tuple to get the required +-1 halo added automatically."
        )
    return candidates


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
    ``ks[i + 1]`` -- i.e. to the interior K values only.
    """
    per_pair = list(zip(*blocks, strict=True))  # index by K pair instead of by run pair
    return np.array([list(per_pair[i]) + list(per_pair[i - 1]) for i in range(1, len(per_pair))])


def _score_block(
    run_a: Mapping[int, Any],
    run_b: Mapping[int, Any],
    pairs: Sequence[tuple[int, int]],
    score_fn: Callable[[Any, Any], float],
) -> list[float]:
    """Similarity of ``run_a`` at each K against ``run_b`` at the next K, one block per run pair."""
    return [score_fn(run_a[low], run_b[high]) for low, high in pairs]


def cluster_stability(
    labels: Mapping[int, Sequence[Any]],
    *,
    score_fn: Callable[[Any, Any], float] = fowlkes_mallows_score,
) -> tuple[list[int], np.ndarray]:
    """Score already-computed labelings by how stably each K reproduces across runs.

    The batch counterpart to the sweep inside :func:`sweep_auto_k`, for labelings produced
    elsewhere: only the labels are compared, so any clusterer will do.

    Parameters
    ----------
    labels
        Mapping of K to that K's labelings, one per run. Every K needs the same number of
        runs, and at least two, since a run can only be scored against another run.
    score_fn
        Similarity of two labelings. Any ``(labels_true, labels_pred) -> float`` works, e.g.
        :func:`~sklearn.metrics.adjusted_rand_score`.

    Returns
    -------
    The interior K values and their stability matrix of shape
    ``(len(interior), n_runs * (n_runs - 1))``, row ``i`` belonging to ``interior[i]``.
    """
    ks = sorted(labels)
    if len(ks) < 3:
        raise ValueError(
            f"stability is only defined on interior K values, so at least 3 K values are needed, got {ks!r}"
        )
    run_counts = {len(labels[k]) for k in ks}
    if len(run_counts) != 1:
        raise ValueError(f"every K needs the same number of runs, got { ({k: len(labels[k]) for k in ks})!r}")
    n_runs = run_counts.pop()
    if n_runs < 2:
        raise ValueError(f"stability needs at least 2 runs to compare, got {n_runs}")

    pairs = list(zip(ks[:-1], ks[1:], strict=True))
    runs = [{k: labels[k][r] for k in ks} for r in range(n_runs)]
    # each unordered pair of runs once, in the same direction as `sweep_auto_k` compares them
    blocks = [_score_block(runs[b], runs[a], pairs, score_fn) for a, b in combinations(range(n_runs), 2)]
    return ks[1:-1], mirror_stability(blocks)


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


class ClusterAutoKUns(TypedDict):
    """The part of a sweep result that can be stored in :attr:`anndata.AnnData.uns`.

    Every field survives an ``h5ad`` round trip with its type intact. Lists would not --
    they come back as arrays -- which is why the fitted and scored K values are read off
    ``table`` rather than repeated as fields.

    Attributes
    ----------
    table
        Per-K diagnostics indexed by K: ``stability_mean``, ``stability_std`` and ``nll``.
        Every fitted K has a row, but the ``+-1`` halo is never scored, so its stability is
        ``NaN``. ``table.index[table["stability_mean"].notna()]`` gives the K values that
        were scored, which is what plots should use as their x-axis.
    stability
        Raw similarity values, of shape ``(n_scored_k, n_comparisons)``. Row ``i`` belongs to
        the ``i``-th scored K. Exposed unaggregated so that distributions can be plotted and
        numerical parity with other implementations can be asserted.
    best_k
        The scored K with the highest mean stability. The reason the sweep exists.
    n_runs
        Number of runs actually performed, which is below ``max_runs`` if the sweep converged.
    converged
        Whether the sweep stopped early because the stability curve had settled.
    """

    table: pd.DataFrame
    stability: np.ndarray
    best_k: int
    n_runs: int
    converged: bool


class ClusterAutoKResult(ClusterAutoKUns):
    """A sweep result: everything in :class:`ClusterAutoKUns`, plus the fits themselves.

    Attributes
    ----------
    labels
        Labeling of the best fit (lowest ``nll``) per K, for every fitted K.

        Labels belong in :attr:`~anndata.AnnData.obs` -- that is where masking,
        ``min_niche_size`` and the per-library merge apply to them, and where plotting expects
        them -- so ``store_labels`` writes them there. This field is what you get when nothing
        was written: the sweep costs up to ``max_runs x len(K)`` fits, too expensive to discard.

        It is also the one field :func:`to_uns` drops. Partly because :mod:`anndata` cannot
        write a dict with non-string keys, but mainly because a copy in ``uns`` would be the
        raw sweep labeling, free to disagree with the post-processed ``obs`` columns.
    """

    labels: dict[int, np.ndarray]


def to_uns(result: ClusterAutoKResult) -> ClusterAutoKUns:
    """The storable part of *result*, for :attr:`anndata.AnnData.uns`."""
    return {key: result[key] for key in ClusterAutoKUns.__annotations__}  # type: ignore[typeddict-item,misc]


def _fit_once(X: Any, k: int, random_state: int, model_params: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """Fit one mixture at ``k`` and return its labeling and negative log-likelihood."""
    gmm = GaussianMixture(n_components=k, random_state=random_state, **model_params)
    try:
        gmm.fit(X)
    except ValueError as err:  # a failed fit otherwise aborts the whole sweep opaquely
        hint = (
            " Pass a stronger regularisation with model_params={'reg_covar': 1e-4}, or request a "
            "smaller range of K values."
            if "ill-defined empirical covariance" in str(err)
            else ""
        )
        raise ValueError(f"the mixture fit at K={k} failed: {err}{hint}") from err
    # labels are in [0, k), and `GaussianMixture` already rejects k > n_samples, so uint32
    # cannot overflow. Every run's labeling is kept for the whole sweep, so the narrower
    # dtype is what keeps that affordable.
    # scikit-learn has no `nll_`; `score` is the mean log-likelihood per observation
    return gmm.predict(X).astype(np.uint32, copy=False), -float(gmm.score(X))


def sweep_auto_k(
    X: Any,
    n_clusters: Sequence[int],
    *,
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    model_params: Mapping[str, Any] | None = None,
    rng: np.random.Generator | None = None,
) -> ClusterAutoKResult:
    """Fit every K repeatedly and score each K by the stability of its labeling.

    Parameters
    ----------
    X
        Feature matrix with observations as rows.
    n_clusters
        The K values to fit, already resolved by :func:`expand_n_clusters`.
    max_runs
        Maximum number of repetitions per K. Must be at least 2: a single run leaves
        stability undefined, since there is nothing to compare against.
    convergence_tol
        Stop early once the mean absolute percentage error between the mean stability
        curves of two consecutive runs falls below this value.
    model_params
        Extra keyword arguments for :class:`~sklearn.mixture.GaussianMixture`. The caller's
        mapping is never modified. ``n_components`` and ``random_state`` are controlled by
        this function and are rejected.
    rng
        Generator seeding every individual fit.

    Returns
    -------
    The sweep result; see :class:`ClusterAutoKResult`.
    """
    if max_runs <= 1:
        raise ValueError(f"stability needs at least 2 runs to compare, got max_runs={max_runs}")

    model_params = dict(model_params or {})
    for owned in ("n_components", "random_state"):
        if owned in model_params:
            raise ValueError(
                f"'{owned}' cannot be set through 'model_params'; it is controlled by "
                f"{'n_clusters' if owned == 'n_components' else 'seed'}"
            )
    model_params.setdefault("init_params", DEFAULT_INIT_PARAMS)

    if rng is None:
        rng = np.random.default_rng()

    ks = list(n_clusters)
    # adjacent pairs rather than `k + 1` arithmetic, so non-contiguous K lists work
    pairs = list(zip(ks[:-1], ks[1:], strict=True))

    labels_per_run: list[dict[int, np.ndarray]] = []
    best_nll: dict[int, float] = {}
    best_labels: dict[int, np.ndarray] = {}
    blocks: list[list[float]] = []
    previous_curve: np.ndarray | None = None
    converged = False

    for _ in range(max_runs):
        run_labels: dict[int, np.ndarray] = {}
        for k in ks:
            # a fresh draw per fit, so runs differ while the whole sweep stays reproducible
            labels, nll = _fit_once(X, k, int(rng.integers(np.iinfo(np.int32).max)), model_params)
            run_labels[k] = labels
            if k not in best_nll or nll < best_nll[k]:
                best_nll[k], best_labels[k] = nll, labels

        if labels_per_run:
            # this run's K against every stored run's K+1 -- one direction only, which
            # `mirror_stability` then folds into a symmetric per-K score
            blocks.extend(_score_block(run_labels, stored, pairs, fowlkes_mallows_score) for stored in labels_per_run)
            curve = mirror_stability(blocks).mean(axis=1)
            if previous_curve is not None and mean_absolute_percentage_error(previous_curve, curve) < convergence_tol:
                labels_per_run.append(run_labels)
                converged = True
                break
            previous_curve = curve

        labels_per_run.append(run_labels)

    stability = mirror_stability(blocks)
    table = _stability_frame(ks, ks[1:-1], stability)
    table.insert(2, "nll", [best_nll[k] for k in ks])
    return ClusterAutoKResult(
        table=table,
        stability=stability,
        best_k=int(table["stability_mean"].idxmax()),
        labels=best_labels,
        n_runs=len(labels_per_run),
        converged=converged,
    )
