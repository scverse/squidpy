"""Stability-based selection of the number of clusters (K) for a Gaussian mixture.

Reimplements CellCharter's ``ClusterAutoK`` (https://github.com/CSOgroup/cellcharter).

Anything needing :class:`~anndata.AnnData` belongs in :mod:`squidpy.gr._cluster_auto_k`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from fast_array_utils.types import HasArrayNamespace as Array
from sklearn.base import clone
from sklearn.metrics import fowlkes_mallows_score, mean_absolute_percentage_error
from sklearn.mixture import GaussianMixture

from squidpy.types import ClusterAutoKResult, SweepableClusterer

# `best_k` is a function of run-to-run variability, and therefore of the initialization.
# Pinned so that the selected K does not silently change with a scikit-learn default.
DEFAULT_INIT_PARAMS = "random_from_data"

# The two spellings of the number-of-clusters parameter, in the order they are tried.
_K_PARAMS = ("n_components", "n_clusters")

# Mixture parameters squidpy sets itself, and the argument that sets each: routing them
# through `model_params` instead would silently override the sweep.
_GMM_OWNED = {"n_components": "n_clusters", "random_state": "rng"}


def expand_n_clusters(n_clusters: tuple[int, int] | Sequence[int]) -> list[int]:
    """Resolve the requested K values into the list of K values to fit.

    A ``(min, max)`` tuple is expanded to ``range(max(1, min - 1), max + 2)``: stability is
    only defined on interior K values (see :func:`mirror_stability`), so the bounds need a
    +-1 halo. ``min=1`` is clamped, so K=1 stays halo and is never selectable.

    Any other sequence is taken literally and gets no halo.
    """
    if isinstance(n_clusters, tuple):
        if len(n_clusters) != 2:
            raise ValueError(f"'n_clusters' as a tuple must be (min, max), got {n_clusters!r}")
        low, high = n_clusters
        if low > high:
            raise ValueError(f"'n_clusters' must be (min, max) with min <= max, got {n_clusters!r}")
        if low < 1:
            raise ValueError(f"'n_clusters' must be (min, max) with min >= 1, got {n_clusters!r}")
        candidates = list(range(max(1, low - 1), high + 2))
    else:
        candidates = list(n_clusters)

    if any(k < 1 for k in candidates):
        raise ValueError(f"every value in 'n_clusters' must be at least 1, got {candidates!r}")
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"'n_clusters' must not contain duplicates, got {candidates!r}")
    if len(candidates) < 3:
        raise ValueError(
            f"stability is only defined on interior K values, so at least 3 K values are needed, got {candidates!r}."
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
    run_a: Mapping[int, Array],
    run_b: Mapping[int, Array],
    pairs: Sequence[tuple[int, int]],
    score_fn: Callable[[Array, Array], float],
) -> list[float]:
    """Similarity of ``run_a`` at each K against ``run_b`` at the next K, one block per run pair."""
    return [score_fn(run_a[low], run_b[high]) for low, high in pairs]


def cluster_stability(
    labels: Mapping[int, Sequence[Array]],
    *,
    score_fn: Callable[[Array, Array], float] = fowlkes_mallows_score,
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


def to_uns(result: ClusterAutoKResult) -> dict[str, object]:
    # don't include labels since they are already in obs
    return {key: value for key, value in result._asdict().items() if key != "labels"}


def label_columns(result: ClusterAutoKResult, key: str, *, all_labels: bool = False) -> dict[str, pd.Categorical]:
    """The label columns a sweep result contributes.

    The selected K takes the bare *key*, which keeps the primary column name predictable
    for downstream code; with ``all_labels`` every fitted K follows as ``{key}_k{K}``.
    """
    columns = {key: pd.Categorical(result.labels[result.best_k])}
    if all_labels:
        columns |= {f"{key}_k{k}": pd.Categorical(labeling) for k, labeling in result.labels.items()}
    return columns


def check_model_params(model_params: Mapping[str, object]) -> None:
    """Reject the mixture parameters squidpy sets itself.

    Separate from :func:`_gmm` so that a caller taking a ``model_params`` argument can
    reject a bad one up front, rather than at whichever point it happens to build a
    mixture.
    """
    for owned, controller in _GMM_OWNED.items():
        if owned in model_params:
            raise ValueError(f"'{owned}' cannot be set through 'model_params'; it is controlled by '{controller}'")


def check_sweepable(clusterer: SweepableClusterer) -> str:
    """Name of *clusterer*'s number-of-clusters parameter, rejecting what cannot be swept.

    Two things are checked, in the only order that works. The methods first, by
    ``isinstance`` -- :class:`~squidpy.types.SweepableClusterer` is runtime-checkable, and
    ``get_params`` has to exist before it can be called. Then the parameters, which no
    protocol can state; see :meth:`~squidpy.types.SweepableClusterer.set_params`. Both
    before any fitting, so an unusable clusterer says so immediately rather than after the
    first fit.
    """
    if not isinstance(clusterer, SweepableClusterer):
        # e.g. the biclustering estimators, which label rows and columns separately rather
        # than assigning one label per observation, and so have no `fit_predict`
        raise ValueError(
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
            "same labeling and its stability is 1 whatever K is. Stability selection needs a "
            "clusterer that varies between runs."
        )
    return k_param


def _gmm(model_params: Mapping[str, object] | None = None, **overrides: object) -> GaussianMixture:
    """The default clusterer: a :class:`~sklearn.mixture.GaussianMixture` with a pinned init.

    The parameters in ``_GMM_OWNED`` are the caller's, not *model_params*': the sweep sets
    them per fit, and a single-fit caller passes them as *overrides*. The caller's mapping
    is never modified.
    """
    check_model_params(model_params or {})
    return GaussianMixture(**{"init_params": DEFAULT_INIT_PARAMS, **(model_params or {})}, **overrides)


def _fit_once(
    clusterer: SweepableClusterer, k_param: str, X: Array, k: int, random_state: int
) -> tuple[np.ndarray, float]:
    """One run: a fresh clone of *clusterer* at ``k``, seeded by ``random_state``."""
    est = clone(clusterer).set_params(**{k_param: k, "random_state": random_state})
    try:
        # `fit` then `predict` where the estimator has one, since that is what a mixture
        # does; transductive clusterers only ever expose `fit_predict`.
        labels = est.fit(X).predict(X) if hasattr(est, "predict") else est.fit_predict(X)
    except ValueError as err:  # a failed fit otherwise aborts the whole sweep opaquely
        hint = (
            " Pass a stronger regularisation with model_params={'reg_covar': 1e-4}, or request a "
            "smaller range of K values."
            if "ill-defined empirical covariance" in str(err)
            else ""
        )
        raise ValueError(f"the fit at K={k} failed: {err}{hint}") from err
    # labels are in [0, k), and the estimator already rejects k > n_samples, so uint32 cannot
    # overflow. Every run's labeling is kept for the whole sweep, so the narrower dtype is
    # what keeps that affordable.
    labels = np.asarray(labels).astype(np.uint32, copy=False)
    # only used to pick which run's labeling is kept for this K; `best_k` never sees it.
    # scikit-learn has no `nll_`, and `score` means different things per estimator, so this
    # is only ever compared within one K. `NaN` for clusterers without one keeps the first run.
    nll = -float(est.score(X)) if hasattr(est, "score") else float("nan")
    return labels, nll


def sweep_auto_k(
    X: Array,
    n_clusters: Sequence[int],
    *,
    clusterer: SweepableClusterer | None = None,
    max_runs: int = 10,
    convergence_tol: float = 1e-2,
    seed: int | None = None,
) -> ClusterAutoKResult:
    """Fit every K repeatedly and score each K by the stability of its labeling.

    Parameters
    ----------
    X
        Feature matrix with observations as rows.
    n_clusters
        The K values to fit, already resolved by :func:`expand_n_clusters`.
    clusterer
        A :class:`~squidpy.types.SweepableClusterer`, whose number of clusters and
        ``random_state`` this sets per fit -- so its own are ignored. Defaults to
        :class:`~sklearn.mixture.GaussianMixture`.

        Being stochastic is a real requirement, not a formality: the sweep scores how much
        a K's labeling moves between runs, so a deterministic clusterer looks perfectly
        stable at every K. Estimators that infer their own K have nothing to sweep. Both
        cases are rejected rather than run.

        Of scikit-learn's own clusterers that leaves
        :class:`~sklearn.mixture.GaussianMixture`,
        :class:`~sklearn.mixture.BayesianGaussianMixture`,
        :class:`~sklearn.cluster.KMeans`, :class:`~sklearn.cluster.MiniBatchKMeans`,
        :class:`~sklearn.cluster.BisectingKMeans` and
        :class:`~sklearn.cluster.SpectralClustering`; the rule is the contract, though, so
        anything else matching it works too, scikit-learn or not.
    max_runs
        Maximum number of repetitions per K. Must be at least 2: a single run leaves
        stability undefined, since there is nothing to compare against.
    convergence_tol
        Stop early once the mean absolute percentage error between the mean stability
        curves of two consecutive runs falls below this value.
    seed
        Base seed. Each fit is seeded from ``(seed, run, K)``, so a fit's seed does not
        depend on its position in ``n_clusters`` and adding a K leaves the others untouched.

    Returns
    -------
    The sweep result; see :class:`~squidpy.types.ClusterAutoKResult`.
    """
    if max_runs <= 1:
        raise ValueError(f"stability needs at least 2 runs to compare, got max_runs={max_runs}")

    if clusterer is None:
        clusterer = _gmm()
    k_param = check_sweepable(clusterer)

    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1)[0])

    ks = list(n_clusters)
    # adjacent pairs rather than `k + 1` arithmetic, so non-contiguous K lists work
    pairs = list(zip(ks[:-1], ks[1:], strict=True))

    labels_per_run: list[dict[int, np.ndarray]] = []
    best_nll: dict[int, float] = {}
    best_labels: dict[int, np.ndarray] = {}
    blocks: list[list[float]] = []
    previous_curve: np.ndarray | None = None
    converged = False

    for run in range(max_runs):
        run_labels: dict[int, np.ndarray] = {}
        for k in ks:
            # keyed by (run, K) rather than drawn in sequence, so appending a candidate K
            # does not re-seed every other K's fits
            random_state = int(np.random.SeedSequence([seed, run, k]).generate_state(1)[0])
            labels, nll = _fit_once(clusterer, k_param, X, k, random_state)
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
