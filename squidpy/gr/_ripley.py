"""Functions for point patterns spatial statistics."""
from typing import Tuple, Literal, Optional

from numpy.random import default_rng
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
import numpy as np


def _ripley(
    coordinates: np.ndarray,
    mode: Literal["L", "F", "J"],
    metric: str = "euclidean",
    n_neighbors: int = 2,
    n_simulations: int = 100,
    n_observations: int = 1000,
    scale_max: int = 1,
    n_steps: int = 50,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    hull = ConvexHull(coordinates)
    tree = NearestNeighbors(metric=metric, n_neighbors=n_neighbors).fit(coordinates)
    support = _build_support(tree, coordinates, n_steps, n_neighbors, scale_max)

    stat_function = alphabet_dict.get(mode)
    if mode in ("F", "J"):
        random = _ppp(hull, 1, n_observations, seed=seed)
        distances, _ = tree.kneighbors(random, n_neighbors=1)
        if mode == "F":
            distances = distances.squeeze()
        elif mode == "J":
            n_distances, _ = tree.kneighbors(coordinates, n_neighbors=1)
            distances = (n_distances.squeeze(), distances.squeeze())
    else:
        distances, _ = tree.kneighbors(coordinates, n_neighbors=1)
        distances = distances.squeeze()

    bins, obs_stats = stat_function(distances, support)

    sims = np.empty((len(bins), n_simulations)).T
    pvalues = np.ones_like(support)

    for i in range(n_simulations):
        random_i = _ppp(hull, 1, n_observations, seed=seed)
        if mode in ("F", "J"):
            tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neighbors).fit(random_i)
            distances_i, _ = tree_i.kneighbors(random, n_neighbors=1)
            if mode == "F":
                distances_i = distances_i.squeeze()
            elif mode == "J":
                distances_i = (n_distances.squeeze(), distances_i.squeeze())
        else:
            tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neighbors).fit(random_i)
            distances_i, _ = tree_i.kneighbors(coordinates, n_neighbors=1)
            distances_i = distances_i.squeeze()

        _, stats_i = stat_function(distances_i, support)
        pvalues += stats_i >= obs_stats
        sims[i] = stats_i

    pvalues /= n_simulations + 1
    pvalues = np.minimum(pvalues, 1 - pvalues)

    return bins, obs_stats, pvalues, sims


def _f_g_function(distances: np.ndarray, support: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts, bins = np.histogram(distances, bins=support)
    fracs = np.cumsum(counts) / counts.sum()
    return bins, np.asarray([0, *fracs])


def _ppp(hull: ConvexHull, n_simulations: int, n_observations: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate Poisson POint Process on a polygon.

    Parameters
    ----------
    hull
        Convex hull of the area of interest.
    n_simulations
        Number of simulated point processes.
    n_observations
        Number of observations to sample from each simulation
    seed
        Random seed.

    Returns
    -------
        An Array with shape (n_simulation, n_observations, 2).
    """
    rng = default_rng(None if seed is None else seed)
    vxs = hull.points[hull.vertices]
    deln = Delaunay(vxs)

    bbox = np.array([*vxs.min(0), *vxs.max(0)])
    result = np.empty((n_simulations, n_observations, 2))

    for i_sim in range(n_simulations):
        i_obs = 0
        while i_obs < n_observations:
            x, y = (
                rng.uniform(bbox[0], bbox[2]),
                rng.uniform(bbox[1], bbox[3]),
            )
            if deln.find_simplex((x, y)) >= 0:
                result[i_sim, i_obs] = (x, y)
                i_obs += 1

    return result.squeeze()


def _build_support(
    tree: NearestNeighbors, coordinates: np.ndarray, n_steps: int, n_neighbors: int, scale_max: int = 1
) -> np.ndarray:
    max_dist = tree.kneighbors(coordinates, n_neighbors=n_neighbors)[0].max()
    max_dist *= scale_max
    support = np.linspace(0, max_dist, num=n_steps)

    return support


alphabet_dict = {"F": _f_g_function, "G": _f_g_function}
