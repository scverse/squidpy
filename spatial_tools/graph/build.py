"""
Functions for building graph from spatial coordinates
"""

import numpy as np


def spatial_connectivity(
    adata: "AnnData",
    obsm: str = "spatial",
    key_added: str = "spatial_connectivity",
    n_neigh: int = 6,
    coord_type: int = "visium",
):
    """
    Creates graph from spatial coordinates

    Params
    ------
    adata
        The AnnData object.
    obsm
        Key to spatial coordinates.
    key_added
        Key added to connectivity matrix in obsp.
    n_neigh
        Number of neighborhoods to consider
    coord_type
        Type of coordinate system (Visium vs. general coordinates)
    """

    adata.obsp[key_added] = _build_connectivity(
        adata,
    )


def _build_connectivity():
    """
    Build connectivity matrix from spatial coordinates
    """
    return
