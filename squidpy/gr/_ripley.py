"""Functions for point patterns spatial statistics."""
from typing import List, Tuple, Union, Literal, Optional

from numpy.random import default_rng
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import pdist
import numpy as np
import pandas as pd


def _ripley(
    coordinates: np.ndarray,
    clusters: np.ndarray,
    mode: Literal["F", "G", "L"],
    metric: str = "euclidean",
    n_neighbors: int = 2,
    n_simulations: int = 100,
    n_observations: int = 1000,
    max_dist: np.float_ = None,
    n_steps: int = 50,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    # prepare support
    N = coordinates.shape[0]
    hull = ConvexHull(coordinates)
    area = hull.volume
    if max_dist is None:
        max_dist = (area / 2) ** 0.5
    support = np.linspace(0, max_dist, n_steps)

    # prepare labels
    le = LabelEncoder().fit(clusters)
    cluster_idx = le.transform(clusters)
    obs_arr = np.empty((le.classes_.shape[0], n_steps))

    for i in np.arange(np.max(cluster_idx) + 1):
        coord_c = coordinates[cluster_idx == i, :]
        if mode == "F":
            random = _ppp(hull, 1, n_observations, seed=seed)
            tree_c = NearestNeighbors(metric=metric, n_neighbors=n_neighbors).fit(coord_c)
            distances, _ = tree_c.kneighbors(random, n_neighbors=n_neighbors)
            bins, obs_stats = _f_g_function(distances.squeeze(), support)
        elif mode == "G":
            tree_c = NearestNeighbors(metric=metric, n_neighbors=n_neighbors).fit(coord_c)
            distances, _ = tree_c.kneighbors(coordinates[cluster_idx != i, :], n_neighbors=n_neighbors)
            bins, obs_stats = _f_g_function(distances.squeeze(), support)
        elif mode == "L":
            distances = pdist(coord_c, metric=metric)
            bins, obs_stats = _l_function(distances, support, N, area)
        obs_arr[i] = obs_stats

    sims = np.empty((n_simulations, len(bins)))
    pvalues = np.ones((le.classes_.shape[0], len(bins)))

    for i in range(n_simulations):
        random_i = _ppp(hull, 1, n_observations, seed=seed)
        if mode == "F":
            tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neighbors).fit(random_i)
            distances_i, _ = tree_i.kneighbors(random, n_neighbors=1)
            _, stats_i = _f_g_function(distances_i.squeeze(), support)
        elif mode == "G":
            tree_i = NearestNeighbors(metric=metric, n_neighbors=n_neighbors).fit(random_i)
            distances_i, _ = tree_i.kneighbors(coordinates, n_neighbors=1)
            _, stats_i = _f_g_function(distances_i.squeeze(), support)
        elif mode == "L":
            distances_i = pdist(random_i, metric=metric)
            _, stats_i = _l_function(distances_i, support, N, area)

        for j in range(obs_arr.shape[0]):
            pvalues[j] += stats_i >= obs_arr[j]
        sims[i] = stats_i

    pvalues /= n_simulations + 1
    pvalues = np.minimum(pvalues, 1 - pvalues)

    obs_df = _reshape_res(obs_arr.T, columns=le.classes_, index=bins, var_name="clusters")
    sims_df = _reshape_res(sims.T, columns=np.arange(n_simulations), index=bins, var_name="simulations")

    return bins, obs_df, pvalues, sims_df


def _reshape_res(
    results: np.ndarray, columns: Union[np.ndarray, List[str]], index: np.ndarray, var_name: str
) -> pd.DataFrame:
    df = pd.DataFrame(results, columns=columns, index=index)
    df.index.set_names(["bins"], inplace=True)
    df = df.melt(var_name=var_name, value_name="stats", ignore_index=False)
    df.reset_index(inplace=True)
    return df


def _f_g_function(distances: np.ndarray, support: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    counts, bins = np.histogram(distances, bins=support)
    fracs = np.cumsum(counts) / counts.sum()
    return bins, np.concatenate((np.zeros((1,), dtype=np.float_), fracs))


def _l_function(
    distances: np.ndarray, support: np.ndarray, n: np.int_, area: np.float_
) -> Tuple[np.ndarray, np.ndarray]:
    n_pairs_less_than_d = (distances < support.reshape(-1, 1)).sum(axis=1)
    intensity = n / area
    k_estimate = ((n_pairs_less_than_d * 2) / n) / intensity
    l_estimate = np.sqrt(k_estimate / np.pi)
    return support, l_estimate


def _ppp(hull: ConvexHull, n_simulations: int, n_observations: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Simulate Poisson Point Process on a polygon.

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
