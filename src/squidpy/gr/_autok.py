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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import fowlkes_mallows_score, mean_absolute_percentage_error
from sklearn.mixture import GaussianMixture

__all__ = ["ClusterAutoKResult", "expand_n_clusters", "mirror_stability", "sweep_auto_k"]

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


@dataclass
class ClusterAutoKResult:
    """Outcome of a stability sweep over candidate numbers of clusters.

    Attributes
    ----------
    n_clusters
        Every K that was fitted, including the ``+-1`` halo added for a ``(min, max)`` request.
    interior
        The K values stability is defined on, ``n_clusters[1:-1]``. ``best_k`` and ``peaks``
        are always drawn from these.
    stability
        Raw similarity values, of shape ``(len(interior), n_comparisons)``. Row ``i`` belongs
        to ``interior[i]``. Exposed unaggregated so that distributions can be plotted and
        numerical parity with other implementations can be asserted.
    nll
        Negative log-likelihood of the best fit per K, for **every** K in ``n_clusters``
        including the halo.
    labels
        Labeling of the best fit (lowest ``nll``) per K, for every K in ``n_clusters``.
    n_runs
        Number of runs actually performed, which is below ``max_runs`` if the sweep converged.
    converged
        Whether the sweep stopped early because the stability curve had settled.
    """

    n_clusters: list[int]
    interior: list[int]
    stability: np.ndarray
    nll: dict[int, float]
    labels: dict[int, np.ndarray] = field(repr=False)
    n_runs: int
    converged: bool

    @property
    def stability_mean(self) -> np.ndarray:
        """Mean stability per interior K."""
        return self.stability.mean(axis=1)

    @property
    def best_k(self) -> int:
        """Interior K with the highest mean stability."""
        return self.interior[int(np.argmax(self.stability_mean))]

    @property
    def peaks(self) -> list[int]:
        """Interior K values at a local maximum of the mean stability curve.

        Runner-up candidates worth inspecting: stability is often multi-modal, and a peak
        that is not ``best_k`` can still be the biologically meaningful resolution.
        """
        peak_idx, _ = find_peaks(self.stability_mean)
        return [self.interior[i] for i in peak_idx]

    def to_frame(self) -> pd.DataFrame:
        """Per-K diagnostics as a table indexed by K.

        Halo rows (the K values outside ``interior``) carry ``NaN`` stability: they are
        fitted, and therefore have an ``nll``, but are never scored. Plots should use
        ``interior`` as their x-axis.
        """
        peaks = set(self.peaks)
        best_k = self.best_k
        stability_mean = dict(zip(self.interior, self.stability_mean, strict=True))
        stability_std = dict(zip(self.interior, self.stability.std(axis=1), strict=True))
        return pd.DataFrame(
            {
                "stability_mean": [stability_mean.get(k, np.nan) for k in self.n_clusters],
                "stability_std": [stability_std.get(k, np.nan) for k in self.n_clusters],
                "nll": [self.nll[k] for k in self.n_clusters],
                "is_peak": [k in peaks for k in self.n_clusters],
                "is_best": [k == best_k for k in self.n_clusters],
            },
            index=pd.Index(self.n_clusters, name="k"),
        )


def _fit_once(X: Any, k: int, random_state: int, model_params: Mapping[str, Any]) -> tuple[np.ndarray, float]:
    """Fit one mixture at ``k`` and return its labeling and negative log-likelihood."""
    gmm = GaussianMixture(n_components=k, random_state=random_state, **model_params)
    gmm.fit(X)
    # scikit-learn has no `nll_`; `score` is the mean log-likelihood per observation
    return gmm.predict(X), -float(gmm.score(X))


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
            blocks.extend(
                [fowlkes_mallows_score(run_labels[low], stored[high]) for low, high in pairs]
                for stored in labels_per_run
            )
            curve = mirror_stability(blocks).mean(axis=1)
            if previous_curve is not None and mean_absolute_percentage_error(previous_curve, curve) < convergence_tol:
                labels_per_run.append(run_labels)
                converged = True
                break
            previous_curve = curve

        labels_per_run.append(run_labels)

    return ClusterAutoKResult(
        n_clusters=ks,
        interior=ks[1:-1],
        stability=mirror_stability(blocks),
        nll=best_nll,
        labels=best_labels,
        n_runs=len(labels_per_run),
        converged=converged,
    )
