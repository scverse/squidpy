from __future__ import annotations

import numpy as np
from scipy.sparse import spmatrix

from squidpy._utils import NDArrayA


def _spagft(g: spmatrix, vals: NDArrayA) -> NDArrayA:
    """
    SpaGFT: Identify spatially variable genes using graph Fourier transform.
    Returns a score per gene indicating spatial variability.
    """
    from scipy.sparse import csgraph
    from scipy.sparse.linalg import eigsh

    # g: adjacency matrix (n_cells x n_cells)
    # vals: (n_cells x n_genes)
    if vals.shape[0] != g.shape[0]:
        if vals.shape[1] == g.shape[0]:
            vals = vals.T
        else:
            raise ValueError("vals must have shape (n_cells, n_genes), where n_cells == g.shape[0].")
    vals_proc = vals

    # Compute normalized Laplacian
    lap = csgraph.laplacian(g, normed=True)
    # Compute eigenvectors (graph Fourier basis)
    n_eig = min(20, lap.shape[0] - 2)
    if n_eig <= 0:
        from scipy.sparse.linalg import ArpackError

        raise ArpackError("Number of eigenvectors requested must be positive.")
    eigvals, eigvecs = eigsh(lap, k=n_eig, which="SM")

    # Project each gene onto Fourier basis, score by energy in low-frequency components
    scores = []
    for gene in vals_proc.T:
        coeffs = eigvecs.T @ gene
        # SVG score: sum squared coeffs for lowest frequencies (spatially smooth signal)
        lf_energy = np.sum(coeffs[: n_eig // 2] ** 2)
        scores.append(lf_energy)
    return np.array(scores)
