"""Functions for point patterns spatial statistics."""
from typing import Tuple, Optional

from numpy.random import default_rng
from scipy.spatial import Delaunay, ConvexHull
from sklearn.neighbors import NearestNeighbors
import numpy as np


def _ppp(coords: np.ndarray, n_simulations: int, n_observations: int, seed: Optional[int] = None) -> np.ndarray:
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
    hull = ConvexHull(coords)
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


def _build_support(coords: np.ndarray, n_steps: int) -> np.ndarray:
    # TODO: add kdtree.max() option?
    y_min = int(coords[:, 1].min())
    y_max = int(coords[:, 1].max())
    x_min = int(coords[:, 0].min())
    x_max = int(coords[:, 0].max())
    area = int((x_max - x_min) * (y_max - y_min))
    support = np.linspace(0, (area / 2) ** 0.5, n_steps)

    return support


def _ripley_f(
    coordinates: np.ndarray,
    metric: str = "euclidean",
    n_neighbors: int = 6,
    n_simulations: int = 100,
    n_observations: int = 1000,
    n_steps: int = 50,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:

    tree = NearestNeighbors(coordinates, metric=metric, n_neighbors=n_neighbors).fit(coordinates)
    support = _build_support(coordinates, n_steps)
    random = _ppp(coordinates, n_simulations, n_observations, seed=seed)
    distances, _ = tree.kneighbors(random, k=1)
    distances = distances.squeeze()

    counts, bins = np.histogram(distances, bins=support)
    fracs = np.cumsum(counts) / counts.sum()

    return bins, np.asarray([0, *fracs])
