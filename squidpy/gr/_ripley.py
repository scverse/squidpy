"""Functions for point patterns spatial statistics."""
from typing import Optional

from numpy.random import default_rng
from scipy.spatial import Delaunay, ConvexHull
import numpy as np


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
