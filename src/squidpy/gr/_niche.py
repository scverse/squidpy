from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from typing import Any, Literal

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from fast_array_utils.conv import to_dense
from fast_array_utils.types import HasArrayNamespace as Array
from scipy.sparse import hstack as sparse_hstack
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture
from spatialdata import SpatialData, sanitize_table
from spatialdata._logging import logger as logg

from squidpy._constants._constants import NicheDefinitions
from squidpy._docs import d, inject_docs
from squidpy._utils import RNGLike, SeedLike, deprecated_randomness_param, legacy_random
from squidpy._validators import assert_isinstance, assert_key_in_adata, assert_one_of
from squidpy.gr._nhood import _nhood_blocks, nhood_aggregate
from squidpy.gr._utils import extract_adata_if_sdata

__all__ = [
    "calculate_niche",
    "calculate_niche_neighborhood",
    "calculate_niche_utag",
    "calculate_niche_cellcharter",
    "calculate_niche_spatialleiden",
]


@d.dedent
@inject_docs(fla=NicheDefinitions)
@deprecated_randomness_param
def calculate_niche(
    data: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter", "spatialleiden"],
    library_key: str | None = None,
    mask: pd.core.series.Series = None,
    groups: str | None = None,
    n_neighbors: int | None = None,
    resolutions: float | tuple[float, float] | list[float | tuple[float, float]] | None = None,
    min_niche_size: int | None = None,
    scale: bool = True,
    abs_nhood: bool = False,
    distance: int | None = None,
    n_hop_weights: list[float] | None = None,
    aggregation: str | None = None,
    n_components: int | None = None,
    spatial_connectivities_key: str = "spatial_connectivities",
    latent_connectivities_key: str = "connectivities",
    layer_ratio: float = 1.0,
    n_iterations: int = -1,
    use_weights: bool | tuple[bool, bool] = True,
    use_rep: str | None = None,
    inplace: bool = True,
    *,
    table_key: str | None = None,
    rng: SeedLike | RNGLike | None = None,
) -> AnnData | None:
    """
    Calculate niches (spatial clusters) based on a user-defined method in 'flavor'.
    The resulting niche labels with be stored in 'adata.obs'.

    .. deprecated:: 1.8.4
        ``calculate_niche`` is deprecated and will be removed in squidpy
        v1.9.0. Use one of the flavor-specific functions instead:

        - :func:`calculate_niche_neighborhood`
        - :func:`calculate_niche_utag`
        - :func:`calculate_niche_cellcharter`
        - :func:`calculate_niche_spatialleiden`

    See Also
    --------
    calculate_niche_neighborhood : Neighborhood-profile flavor with an explicit signature.
    calculate_niche_utag : UTAG flavor with an explicit signature.
    calculate_niche_cellcharter : CellCharter flavor with an explicit signature.
    calculate_niche_spatialleiden : SpatialLeiden flavor with an explicit signature.

    Parameters
    ----------
    %(adata)s
    flavor
        Method to use for niche calculation. Available options are:
            - `{fla.NEIGHBORHOOD.s!r}` - cluster the neighborhood profile.
            - `{fla.UTAG.s!r}` - use utag algorithm (matrix multiplication).
            - `{fla.SPATIALLEIDEN.s!r}` - cluster spatially resolved omics data using Multiplex Leiden.
            - `{fla.CELLCHARTER.s!r}` - a simplified version of CellCharter's approach, using PCA for dimensionality reduction. An arbitrary embedding can be used instead of PCA by setting the `use_rep` parameter which will try to find the embedding in `adata.obsm`.
    %(library_key)s
        If provided, niches will be calculated separately for each unique value in this column.
        Each niche will be prefixed with the library identifier.
    %(table_key)s
    mask
        Boolean array to filter cells which won't get assigned to a niche.
        Note that if you want to exclude these cells during neighborhood calculation already, you should subset your AnnData table before running 'sq.gr.spatial_neigbors'.
        Mask can look like the following. Here, the index values would correspond to adata.obs.index.
        The entries that are False are the ones ignored.
        mask = Series([False, False, True], index = ["a", "b", "c"])
    groups
        Groups based on which to calculate neighborhood profile (E.g. columns of cell type annotations in adata.obs).
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    n_neighbors
        Number of neighbors to use for 'scanpy.pp.neighbors' before clustering using leiden algorithm.
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}` or flavor == `{fla.UTAG.s!r}`.
    resolutions
        List of resolutions to use for leiden clustering.
        In the case of spatialleiden you can pass a tuple. Resolution for the latent space and spatial layer, respectively. A single float applies to both layers.
        Required if flavor == `{fla.NEIGHBORHOOD.s!r}` or flavor == `{fla.UTAG.s!r}`.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    min_niche_size
        Minimum required size of a niche. Niches with fewer cells will be labeled as 'not_a_niche'.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    scale
        If 'True', compute z-scores of neighborhood profiles.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    abs_nhood
        If 'True', calculate niches based on absolute neighborhood profile.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    distance
        n-hop neighbor adjacency matrices to use e.g. [1,2,3] for 1-hop,2-hop,3-hop neighbors respectively or "5" for 1-hop,...,5-hop neighbors. 0 (self) is always included.
        Required if flavor == `{fla.CELLCHARTER.s!r}`.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}`.
    n_hop_weights
        How to weight subsequent n-hop adjacency matrices. E.g. [1, 0.5, 0.25] for weights of 1-hop, 2-hop, 3-hop adjacency matrices respectively.
        Optional if flavor == `{fla.NEIGHBORHOOD.s!r}` and `distance` > 1.
    aggregation
        How to aggregate count matrices. Either 'mean' or 'variance'.
        Required if flavor == `{fla.CELLCHARTER.s!r}`.
    n_components
        Number of components to use for GMM.
        Required if flavor == `{fla.CELLCHARTER.s!r}`.
    %(rng)s
        Optional if flavor == `{fla.CELLCHARTER.s!r}` or flavor == `{fla.SPATIALLEIDEN.s!r}`.
    spatial_connectivities_key
        Key in `adata.obsp` where spatial connectivities are stored.
        Required if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    latent_connectivities_key
        Key in `adata.obsp` where gene expression connectivities are stored.
        Required if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    layer_ratio
        The ratio of the weighting of the layers; latent space vs spatial. A higher ratio will increase relevance of the spatial neighbors and lead to more spatially homogeneous clusters.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    n_iterations
        Number of iterations to run the Leiden algorithm. If the number is negative it runs until convergence.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    use_weights
        Whether to use weights for the edges for latent space and spatial neighbors, respectively. A single bool applies to both layers.
        Optional if flavor == `{fla.SPATIALLEIDEN.s!r}`.
    use_rep
        Key in `adata.obsm` where the embedding is stored. If provided, this embedding will be used instead of PCA for dimensionality reduction.
        Optional if flavor == `{fla.CELLCHARTER.s!r}`.
    inplace
        If 'True', perform the operation in place.
        If 'False', return a new AnnData object with the niche labels.
    """

    warnings.warn(
        "Calling `calculate_niche` is deprecated and will be removed in squidpy "
        "v1.9.0. Use `calculate_niche_neighborhood`, `calculate_niche_utag`, "
        "`calculate_niche_cellcharter`, or `calculate_niche_spatialleiden` instead.",
        FutureWarning,
        stacklevel=2,
    )

    # cellcharter-only defaults stay guarded: filling them for other flavors would trip
    # the "not used for flavor" warning in _check_unnecessary_args
    if flavor == "cellcharter":
        if aggregation is None:
            aggregation = "mean"
        if n_components is None:
            n_components = 10
    if distance is None:
        distance = 3 if flavor == "cellcharter" else 1

    _validate_niche_args(
        data,
        flavor,
        library_key,
        table_key,
        groups,
        n_neighbors,
        resolutions,
        min_niche_size,
        scale,
        abs_nhood,
        distance,
        n_hop_weights,
        aggregation,
        n_components,
        rng,
        spatial_connectivities_key,
        latent_connectivities_key,
        layer_ratio,
        n_iterations,
        use_weights,
        use_rep,
        inplace,
    )

    if resolutions is None:
        resolutions = [0.5]

    if flavor == "neighborhood":
        return calculate_niche_neighborhood(
            data,
            groups=groups,
            resolutions=resolutions,
            n_neighbors=n_neighbors,
            spatial_connectivities_key=spatial_connectivities_key,
            scale=scale,
            distance=distance,
            abs_nhood=abs_nhood,
            n_hop_weights=n_hop_weights,
            embedding_key_added="niche_embedding",
            min_niche_size=min_niche_size,
            mask=mask,
            library_key=library_key,
            copy=not inplace,
            table_key=table_key,
            n_iterations=n_iterations,
            rng=rng,
        )

    elif flavor == "utag":
        return calculate_niche_utag(
            data,
            resolutions=resolutions,
            n_neighbors=n_neighbors,
            use_layer=None,
            spatial_connectivities_key=spatial_connectivities_key,
            embedding_key_added="niche_embedding",
            min_niche_size=min_niche_size,
            mask=mask,
            library_key=library_key,
            copy=not inplace,
            table_key=table_key,
            n_iterations=n_iterations,
            rng=rng,
        )

    elif flavor == "cellcharter":
        return calculate_niche_cellcharter(
            data,
            distance=distance,
            aggregation=aggregation,
            rng=rng,
            spatial_connectivities_key=spatial_connectivities_key,
            n_components=n_components,
            use_rep=use_rep,
            embedding_key_added="niche_embedding",
            min_niche_size=min_niche_size,
            mask=mask,
            library_key=library_key,
            copy=not inplace,
            table_key=table_key,
        )

    elif flavor == "spatialleiden":
        return calculate_niche_spatialleiden(
            data,
            resolutions=resolutions,
            latent_connectivities_key=latent_connectivities_key,
            spatial_connectivities_key=spatial_connectivities_key,
            layer_ratio=layer_ratio,
            n_iterations=n_iterations,
            use_weights=use_weights,
            rng=rng,
            min_niche_size=min_niche_size,
            mask=mask,
            prefix=None,
            library_key=library_key,
            copy=not inplace,
            table_key=table_key,
        )

    return


@d.dedent
def calculate_niche_neighborhood(
    data: AnnData | SpatialData,
    *,
    groups: str,
    resolutions: float | list[float],
    n_neighbors: int = 15,
    spatial_connectivities_key: str = "spatial_connectivities",
    scale: bool = True,
    distance: int = 1,
    abs_nhood: bool = False,
    n_hop_weights: list[float] | None = None,
    embedding_key_added: str = "niche_embedding",
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    copy: bool = False,
    table_key: str | None = None,
    flavor: Literal["igraph", "leidenalg"] = "igraph",
    n_iterations: int = -1,
    rng: SeedLike | RNGLike | None = None,
) -> AnnData | None:
    """Compute spatial niches from local cell-type composition.

    This method represents every observation by a **neighborhood composition
    profile**: a vector describing the cell-type labels observed in its local
    spatial neighborhood. Observations with similar surrounding cell-type
    compositions are then grouped into niches using Leiden clustering.

    A spatial connectivity graph must already be available in
    ``adata.obsp[spatial_connectivities_key]``. The graph defines which
    observations are spatial neighbors; this function does not construct the
    graph itself.

    For each observation ``i``, the method:

    1. Retrieves observations reachable from ``i`` in the spatial graph up to
       ``distance`` hops away.
    2. Counts the values of ``groups`` among those neighboring observations.
    3. Optionally combines profiles from different graph-hop distances using
       ``n_hop_weights``.
    4. Uses either raw counts or normalized cell-type proportions as the
       neighborhood profile.
    5. Optionally z-scores the resulting profile across observations (scale
       argument).
    6. Constructs a k-nearest-neighbor graph from the profile embedding and
       applies Leiden clustering at each requested resolution.

    Thus, a niche is defined by a recurring **local cellular composition**,
    rather than by the expression profile of an individual observation. For
    example, cells surrounded by many immune cells and fibroblasts may be
    assigned to one niche even when those cells themselves have different
    expression profiles or cell-type labels.

    Parameters
    ----------
    %(adata)s
    groups
        Column in ``adata.obs`` containing categorical labels used to define
        neighborhood composition, such as cell types, cell states, clusters,
        or anatomical annotations.
    resolutions
        Resolution parameter(s) for Leiden clustering. A single value produces
        one niche annotation. Supplying multiple values performs Leiden
        clustering separately for each resolution and adds one niche column per
        resolution to ``adata.obs``.
    n_neighbors
        Number of nearest neighbors used to construct the k-nearest-neighbor
        graph on the neighborhood composition embedding before Leiden
        clustering.
    %(niche_spatial_conn_key)s
    scale
        Whether to z-score each neighborhood-profile feature across
        observations before constructing the clustering graph.
    distance
        Maximum number of graph hops used to construct each neighborhood
        profile. ``distance=1`` uses direct spatial neighbors. Larger values
        incorporate increasingly distal observations in the spatial graph.
    abs_nhood
        Whether to use absolute group counts in the neighborhood profile.

        If ``False`` (the default), counts are normalized to proportions, so
        each profile captures relative neighborhood composition and is less
        sensitive to differences in neighborhood size.

        If ``True``, raw counts are retained, so both composition and the
        total number of reachable neighbors can influence the embedding.
    n_hop_weights
        Optional weights used when combining contributions from successive
        graph-hop distances. If provided, the weights determine the relative
        contribution of direct and higher-order neighbors to the final
        neighborhood profile. If not provided, equal weights are used.
    %(niche_common_params)s
    %(table_key)s
    %(niche_leiden_params)s

    Returns
    -------
    If ``copy=True``, returns a copy of ``adata`` with the neighborhood profile
    stored in ``.obsm[embedding_key_added]`` and niche assignments added to
    ``.obs``. Otherwise, modifies ``adata`` in place and returns ``None``.

    Notes
    -----
    This approach is most appropriate when niches are expected to differ
    primarily in local **cellular composition**. In contrast,
    :func:`calculate_niche_utag` and :func:`calculate_niche_cellcharter`
    derive niche embeddings from spatially aggregated molecular or latent
    features rather than from categorical group frequencies.

    """

    embedder = partial(
        _nhood_profile_embedding,
        groups=groups,
        spatial_connectivities_key=spatial_connectivities_key,
        scale=scale,
        distance=distance,
        abs_nhood=abs_nhood,
        n_hop_weights=n_hop_weights,
    )

    # Create instance of _LeidenClusterer using provided inputs
    clusterer = _LeidenClusterer(
        n_neighbors, resolutions, "nhood_niche", flavor=flavor, n_iterations=n_iterations, rng=rng
    )

    return _calculate_niche_custom(
        data,
        embedder,
        clusterer,
        embedding_key_added,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        copy=copy,
        table_key=table_key,
    )


@d.dedent
def calculate_niche_utag(
    data: AnnData | SpatialData,
    *,
    resolutions: float | list[float],
    n_neighbors: int = 15,
    use_layer: str | None = None,
    spatial_connectivities_key: str = "spatial_connectivities",
    embedding_key_added: str = "niche_embedding",
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    copy: bool = False,
    table_key: str | None = None,
    flavor: Literal["igraph", "leidenalg"] = "igraph",
    n_iterations: int = -1,
    rng: SeedLike | RNGLike | None = None,
) -> AnnData | None:
    """Compute spatial niches from UTAG-style feature aggregation.

    Originally adapted from https://github.com/ElementoLab/utag/blob/main/utag/segmentation.py
    This method computes a spatially aggregated feature representation for each
    observation and clusters that representation with Leiden. The resulting
    niches group observations that occur in similar local molecular
    environments.

    A spatial connectivity graph must already be available in
    ``adata.obsp[spatial_connectivities_key]``. The graph determines how
    features are propagated or aggregated across spatially neighboring
    observations; this function does not construct the graph itself.

    The method proceeds as follows:

    1. Selects an input feature matrix from ``adata.X`` or from
       ``adata.layers[use_layer]``.
    2. Performs a normalized (by number of cell-neighbors) aggregation of
       features over the spatial connectivity graph, producing a new feature
       matrix in which each observation reflects information from its local
       spatial neighborhood.
    3. Treats this spatially aggregated matrix as the niche embedding.
    4. Constructs a k-nearest-neighbor graph in the embedding space.
    5. Applies Leiden clustering at each requested resolution.

    If the input contains gene expression values, the embedding describes
    local expression programs rather than merely the categorical composition
    of neighboring cells.

    This differs from :func:`calculate_niche_neighborhood`, which uses counts
    or proportions of a categorical ``groups`` annotation. UTAG-style
    aggregation can identify niches that have similar local expression
    patterns even when their neighborhoods contain different annotated cell
    types, or when cell-type labels are unavailable.

    Parameters
    ----------
    %(adata)s
    resolutions
        Resolution parameter(s) for Leiden clustering. A single value produces
        one niche annotation. Supplying multiple values runs Leiden clustering
        independently at each resolution and adds one niche column per
        resolution to ``adata.obs``.
    n_neighbors
        Number of nearest neighbors used to construct the k-nearest-neighbor
        graph on the spatially aggregated embedding before Leiden clustering.
    use_layer
        Key in ``adata.layers`` containing the feature matrix to aggregate. If
        ``None``, uses ``adata.X``.

        Typically, this should contain a normalized expression matrix or
        another observation-by-feature representation appropriate for local
        aggregation. The selected matrix determines what biological signal is
        used to define niches.
    %(niche_spatial_conn_key)s
    %(niche_common_params)s
    %(table_key)s
    %(niche_leiden_params)s

    Returns
    -------
    If ``copy=True``, returns a copy of ``adata`` with the spatially aggregated
    embedding stored in ``.obsm[embedding_key_added]`` and niche assignments
    added to ``.obs``. Otherwise, modifies ``adata`` in place and returns
    ``None``.

    """

    embedder = partial(_utag_embedding, spatial_connectivities_key=spatial_connectivities_key, use_layer=use_layer)

    clusterer = _LeidenClusterer(
        n_neighbors, resolutions, "utag_niche", flavor=flavor, n_iterations=n_iterations, rng=rng
    )

    return _calculate_niche_custom(
        data,
        embedder,
        clusterer,
        embedding_key_added,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        copy=copy,
        table_key=table_key,
    )


@d.dedent
def calculate_niche_cellcharter(
    data: AnnData | SpatialData,
    *,
    distance: int = 3,
    aggregation: str = "mean",
    rng: SeedLike | RNGLike | None = None,
    spatial_connectivities_key: str = "spatial_connectivities",
    n_components: int = 10,
    use_rep: str | None = None,
    embedding_key_added: str = "niche_embedding",
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute spatial niches using a CellCharter-style embedding and GMM.

    This method identifies niches by clustering an embedding that represents
    each observation together with information from its surrounding spatial
    neighborhood. Unlike :func:`calculate_niche_neighborhood`, which builds
    an embedding from categorical cell-type composition, this approach uses a
    continuous feature representation and a Gaussian mixture model (GMM) for
    clustering.

    Two input modes are supported:

    - If ``use_rep`` is provided, ``adata.obsm[use_rep]`` is used as the input
      representation for niche clustering.
    - If ``use_rep`` is ``None``, a CellCharter-style spatial embedding is
      computed by aggregating features over the precomputed spatial graph,
      including information from multi-hop neighborhoods up to ``distance``.

    A spatial connectivity graph must already be present in
    ``adata.obsp[spatial_connectivities_key]`` when spatial aggregation is
    required. This function does not construct the graph itself.

    When an embedding is computed internally, the method:

    1. Starts from the available observation-level feature representation.
    2. Aggregates neighborhood features over the spatial graph from direct
       neighbors through ``distance`` graph hops.
    3. Combines the aggregated features according to ``aggregation`` to create
       a spatial-context embedding for every observation.
    4. Fits a Gaussian mixture model with ``n_components`` mixture components.
    5. Uses the GMM component assignments as niche labels.

    Consequently, each niche corresponds to a probabilistic cluster in a
    feature space that encodes both an observation's features and its broader
    spatial context. Increasing ``distance`` allows the embedding to reflect
    larger tissue-scale neighborhoods, whereas smaller values emphasize local
    microenvironments.

    Parameters
    ----------
    %(adata)s
    distance
        Maximum number of graph hops included when constructing the
        CellCharter-style spatial embedding. ``distance=1`` emphasizes direct
        neighbors; larger values incorporate progressively more distal
        observations in the spatial graph.
    aggregation
        Aggregation statistic used to summarize features across spatial
        neighborhoods. Typical options include ``"mean"`` and ``"variance"``.

        ``"mean"`` emphasizes the average local feature state, such as the
        average local expression program. ``"variance"`` emphasizes local
        heterogeneity in the feature representation.
    %(rng)s
        Seeds the Gaussian mixture clustering step. When stratifying by ``library_key``,
        every library is fitted with an independent rng derived from it.
    %(niche_spatial_conn_key)s
    n_components
        Number of Gaussian mixture components used to assign niches.
        Therefore, this parameter directly determines the number of niche
        labels produced per library or dataset.
    use_rep
        Key in ``adata.obsm`` containing a precomputed observation-level
        representation to cluster. When provided, this representation is used
        instead of deriving a new spatially aggregated embedding.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``copy=True``, returns a copy of ``adata`` with the embedding stored in
    ``.obsm[embedding_key_added]`` and GMM-based niche assignments added to
    ``.obs``. Otherwise, modifies ``adata`` in place and returns ``None``.

    """

    embedder: NicheEmbedder
    if use_rep is not None:
        embedder = partial(_precomputed_embedding, obsm_key=use_rep)
    else:
        logg.warning(
            "CellCharter recommends to use a dimensionality reduced embedding of the data, e.g. a scVI embedding. Since 'use_rep' is not provided, PCA will be used as proxy - performance may be suboptimal."
        )
        embedder = partial(
            _nhop_pca_embedding,
            distance=distance,
            aggregation=aggregation,
            spatial_connectivities_key=spatial_connectivities_key,
        )

    clusterer = _GMMClusterer(n_components, np.random.default_rng(rng), base_colname="cellcharter_niche")

    return _calculate_niche_custom(
        data,
        embedder,
        clusterer,
        embedding_key_added,
        min_niche_size=min_niche_size,
        mask=mask,
        library_key=library_key,
        copy=copy,
        table_key=table_key,
    )


@d.dedent
def calculate_niche_spatialleiden(
    data: AnnData | SpatialData,
    *,
    resolutions: float | tuple[float, float] | list[float | tuple[float, float]],
    latent_connectivities_key: str = "connectivities",
    spatial_connectivities_key: str = "spatial_connectivities",
    layer_ratio: float = 1.0,
    n_iterations: int = -1,
    use_weights: bool | tuple[bool, bool] = True,
    rng: SeedLike | RNGLike | None = None,
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    prefix: str | None = None,
    library_key: str | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using the SpatialLeiden algorithm.

    This is a wrapper around the `SpatialLeiden <https://github.com/HiDiHlabs/SpatialLeiden>`_
    algorithm that uses :class:`~anndata.AnnData` as input and works with two layers; one latent
    space and one spatial layer.
    Adapted from https://github.com/HiDiHlabs/SpatialLeiden/.

    Parameters
    ----------
    %(adata)s
    latent_connectivities_key
        Key in ``adata.obsp`` containing the latent-space connectivity matrix.
    %(niche_spatial_conn_key)s
    resolutions
        Resolution parameter(s) for the Leiden optimization. Can be a single
        float or a list of floats.
    layer_ratio
        Relative weight assigned to the latent and spatial layers.
    n_iterations
        Number of optimization iterations used by SpatialLeiden.
    use_weights
        Whether to use edge weights during clustering.
    %(rng)s
        Each resolution — and each library when stratifying by ``library_key`` — is
        clustered with an independent rng derived from it.
    %(niche_min_niche_size)s
    %(niche_mask)s
    prefix
        Prefix added to niche labels produced by SpatialLeiden.
        When stratifying by ``library_key``, a library-specific prefix is added
        automatically (something like "lib=").
    %(library_key)s
    %(copy)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

    Notes
    -----
    If ``library_key`` is provided, clustering is performed independently for
    each library and the results are merged back into the parent object.
    """

    try:
        import spatialleiden as sl
    except ImportError as e:
        msg = "Please install the spatialleiden algorithm: `pip install squidpy[leiden]` or `conda install bioconda::spatialleiden` or `pip install spatialleiden`."
        raise ImportError(msg) from e

    # obtain adata if data was of sdata type
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)

    adata = orig_adata.copy() if copy else orig_adata

    # normalise once here; everything below this point works with rngs only
    rng = np.random.default_rng(rng)

    if library_key is not None:
        # first assert that library_key was there in adata.obs, and then, stratify the object according to that library_key and
        # then re-call calculate_niche_spatialleiden for each subpart, with library_key = None and prefix with appropriate information like "lib="
        assert_key_in_adata(adata, library_key, attr="obs")
        logg.info(f"Stratifying by library_key '{library_key}'")

        # each library is an independent clustering problem, so it gets its own rng
        # (indexed by `itr` so that skipped empty libraries don't shift the others)
        library_ids = adata.obs[library_key].unique()
        library_rngs = rng.spawn(len(library_ids))

        # go through each library_id and process the corresponding adata subset
        for itr, lib_id in enumerate(library_ids):
            logg.info(f"Processing library '{lib_id}'")

            lib_indices = adata.obs[adata.obs[library_key] == lib_id].index

            if len(lib_indices) == 0:
                logg.warning(f"Library '{lib_id}' contains no cells, skipping")
                continue

            lib_adata = adata[lib_indices].copy()

            # give prefix appropriate value so that the niche values indicate lib id.
            calculate_niche_spatialleiden(
                lib_adata,
                resolutions=resolutions,
                latent_connectivities_key=latent_connectivities_key,
                spatial_connectivities_key=spatial_connectivities_key,
                layer_ratio=layer_ratio,
                n_iterations=n_iterations,
                use_weights=use_weights,
                rng=library_rngs[itr],
                min_niche_size=min_niche_size,
                mask=mask,
                prefix=f"lib={lib_id}_",
                library_key=None,
                copy=False,  # to save memory
                table_key=table_key,
            )

            # from itr==1 onwards, adata will hold the columns that are being added hence,
            # added_columns will be empty. Hence only obtain added_columns when itr==0
            if itr == 0:
                added_columns = list(set(lib_adata.obs.columns) - set(adata.obs.columns))

            for col in added_columns:
                # ensure that adata has the columns in which we are adding the information
                if col not in adata.obs:
                    adata.obs[col] = "not_a_niche"
                adata.obs.loc[lib_indices, col] = list(lib_adata.obs[col].astype("str"))

    else:
        # Simply call sl.spatialleiden with the provided arguments
        if not isinstance(resolutions, list):
            resolutions = [resolutions]

        # every resolution is a separate clustering run, so seed each one independently
        resolution_rngs = rng.spawn(len(resolutions))

        for res, res_rng in zip(resolutions, resolution_rngs, strict=True):
            sl.spatialleiden(
                adata,
                resolution=res,
                use_weights=use_weights,
                n_iterations=n_iterations,
                layer_ratio=layer_ratio,
                latent_neighbors_key=latent_connectivities_key,
                spatial_neighbors_key=spatial_connectivities_key,
                random_state=legacy_random(res_rng),
                directed=False,
                key_added=f"spatialleiden_res={res}",
            )

        # obtain the result_columns, which are basically the difference in columns in orig_adata and adata
        result_columns = [f"spatialleiden_res={res}" for res in resolutions]

        _postprocess_niche_results(adata, result_columns, mask, min_niche_size, prefix)

    # For SpatialData, the column names shouldn't have = sign. Hence, run sanitize_table.
    # TODO: In future, change the naming standard of any niche columns added to not have '=' to be compatible with spatialdata naming
    if isinstance(data, SpatialData):
        sanitize_table(adata)

    return adata if copy else None


@d.dedent
def _calculate_niche_custom(
    data: AnnData | SpatialData,
    embedder: NicheEmbedder,
    clusterer: _NicheClusterer,
    embedding_key_added: str = "niche_embedding",
    min_niche_size: int | None = None,
    mask: pd.Series | None = None,
    library_key: str | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using user-defined embedding, clustering, and postprocessing.

    This function provides a flexible pipeline where embedding, clustering,
    and postprocessing are decoupled and customizable.

    Parameters
    ----------
    %(adata)s
    embedder
        Any ``(AnnData) -> Array`` callable returning one row per observation.
    clusterer
        Instance of ``_NicheClusterer`` used to assign niches based on the embedding.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

    Notes
    -----
    If ``library_key`` is provided, the computation is performed independently
    for each library and results are merged back into ``adata``.

    See Also
    --------
    calculate_niche_neighborhood : Convenience wrapper for neighborhood flavor niche analysis.
    calculate_niche_utag : Convenience wrapper for utag flavor niche analysis.
    calculate_niche_cellcharter : Convenience wrapper for cellcharter flavor niche analysis.
    calculate_niche_spatialleiden : Convenience wrapper for spatialleiden flavor niche analysis.
    _NicheClusterer : Base class for clustering strategies.
    """

    # obtain adata if data was of sdata type
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)

    adata = orig_adata.copy() if copy else orig_adata

    embedding = embedder(adata)
    adata.obsm[embedding_key_added] = embedding

    if library_key is not None:
        assert_key_in_adata(adata, library_key, attr="obs")
        logg.info(f"Stratifying by library_key '{library_key}'")

        # go through each library_id and process the corresponding adata subset
        for itr, lib_id in enumerate(adata.obs[library_key].unique()):
            logg.info(f"Processing library '{lib_id}'")

            lib_indices = adata.obs[adata.obs[library_key] == lib_id].index

            if len(lib_indices) == 0:
                logg.warning(f"Library '{lib_id}' contains no cells, skipping")
                continue

            lib_adata = adata[lib_indices].copy()

            lib_embedding = lib_adata.obsm[embedding_key_added]
            result_columns = clusterer.cluster(lib_adata, lib_embedding)
            _postprocess_niche_results(lib_adata, result_columns, mask, min_niche_size, prefix=f"lib={lib_id}_")

            # from itr==1 onwards, adata will hold the columns that are being added hence,
            # added_columns will be empty. Hence only obtain added_columns when itr==0
            if itr == 0:
                added_columns = list(set(lib_adata.obs.columns) - set(adata.obs.columns))

            for col in added_columns:
                # ensure that adata has the columns in which we are adding the information
                if col not in adata.obs:
                    adata.obs[col] = "not_a_niche"
                adata.obs.loc[lib_indices, col] = list(lib_adata.obs[col].astype("str"))

    else:
        result_columns = clusterer.cluster(adata, embedding)
        _postprocess_niche_results(adata, result_columns, mask, min_niche_size)

    # For SpatialData, the column names shouldn't have = sign. Hence, run sanitize_table.
    # TODO: In future, change the naming standard of any niche columns added to not have '=' to be compatible with spatialdata naming
    if isinstance(data, SpatialData):
        sanitize_table(adata)

    return adata if copy else None


def _validate_niche_args(
    data: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter", "spatialleiden"],
    library_key: str | None,
    table_key: str | None,
    groups: str | None,
    n_neighbors: int | None,
    resolutions: float | tuple[float, float] | list[float | tuple[float, float]] | None,
    min_niche_size: int | None,
    scale: bool,
    abs_nhood: bool,
    distance: int | None,
    n_hop_weights: list[float] | None,
    aggregation: str | None,
    n_components: int | None,
    # the one internal that sees a raw `rng`: it reports on what the caller passed, and
    # `None` must stay `None` here so the "unused for this flavor" check can spot it
    rng: SeedLike | RNGLike | None,
    spatial_connectivities_key: str,
    latent_connectivities_key: str,
    layer_ratio: float,
    n_iterations: int,
    use_weights: bool | tuple[bool, bool],
    use_rep: str | None,
    inplace: bool,
) -> None:
    """
    Validate whether necessary arguments are provided for a given niche flavor.
    Also warns whether unnecessary optional arguments are supplied.

    Raises
    ------
    ValueError
        If required arguments for the specified flavor are missing or have incorrect values.
    TypeError
        If arguments are of incorrect type.
    """
    assert_isinstance(data, (AnnData, SpatialData), name="data")

    assert_one_of(flavor, ["neighborhood", "utag", "cellcharter", "spatialleiden"], name="flavor")

    if isinstance(data, SpatialData) and table_key is None:
        raise TypeError("missing required keyword-only argument: 'table_key'")

    if library_key is not None:
        assert_isinstance(library_key, str, name="library_key")
        adata = extract_adata_if_sdata(data, table_key=table_key)
        if library_key not in adata.obs.columns:
            raise ValueError(f"'library_key' must be a column in 'adata.obs', got {library_key}")

    if n_neighbors is not None:
        assert_isinstance(n_neighbors, int, name="n_neighbors")

    if resolutions is not None:
        if not isinstance(resolutions, float | tuple | list):
            raise TypeError(
                f"'resolutions' must be a float, a tuple of floats, a list of floats, or a list containing floats and/or tuples of floats, got {type(resolutions).__name__}"
            )

        if isinstance(resolutions, tuple):
            if not all(isinstance(x, float) for x in resolutions):
                raise TypeError("All elements in the tuple 'resolutions' must be floats.")
        elif isinstance(resolutions, list):
            for item in resolutions:
                if not (
                    isinstance(item, float) or (isinstance(item, tuple) and all(isinstance(i, float) for i in item))
                ):
                    raise TypeError("Each item in the list 'resolutions' must be a float or a tuple of floats.")

    if n_hop_weights is not None:
        assert_isinstance(n_hop_weights, list, name="n_hop_weights")

    assert_isinstance(scale, bool, name="scale")

    assert_isinstance(abs_nhood, bool, name="abs_nhood")

    # Define parameters used by each flavor
    flavor_param_specs = {
        "neighborhood": {
            "required": ["groups", "n_neighbors", "resolutions", "spatial_connectivities_key"],
            "optional": [
                "min_niche_size",
                "scale",
                "abs_nhood",
                "distance",
                "n_hop_weights",
                "rng",
                "n_iterations",
            ],
            "unused": [
                "aggregation",
                "n_components",
                "latent_connectivities_key",
                "layer_ratio",
                "use_weights",
                "use_rep",
            ],
        },
        "utag": {
            "required": ["n_neighbors", "resolutions", "spatial_connectivities_key"],
            "optional": ["rng", "n_iterations"],
            "unused": [
                "groups",
                "min_niche_size",
                "scale",
                "abs_nhood",
                "distance",
                "n_hop_weights",
                "aggregation",
                "n_components",
                "latent_connectivities_key",
                "layer_ratio",
                "use_weights",
                "use_rep",
            ],
        },
        "cellcharter": {
            "required": ["distance", "aggregation", "spatial_connectivities_key"],
            # `rng` is optional: `None` is a valid value meaning "draw from OS entropy"
            "optional": ["n_components", "use_rep", "rng"],
            "unused": [
                "groups",
                "min_niche_size",
                "scale",
                "abs_nhood",
                "n_neighbors",
                "resolutions",
                "n_hop_weights",
                "latent_connectivities_key",
                "layer_ratio",
                "n_iterations",
                "use_weights",
            ],
        },
        "spatialleiden": {
            "required": ["latent_connectivities_key", "spatial_connectivities_key"],
            "optional": [
                "resolutions",
                "layer_ratio",
                "n_iterations",
                "use_weights",
                "rng",
            ],
            "unused": ["groups", "min_niche_size", "scale", "abs_nhood", "n_neighbors", "n_hop_weights", "use_rep"],
        },
    }

    for param_name in flavor_param_specs[flavor]["required"]:
        param_value = locals()[param_name]
        if param_value is None:
            raise ValueError(f"'{param_name}' is required for flavor '{flavor}'")

    _check_unnecessary_args(
        flavor,
        {
            "groups": groups,
            "n_neighbors": n_neighbors,
            "resolutions": resolutions,
            "min_niche_size": min_niche_size,
            "scale": scale,
            "abs_nhood": abs_nhood,
            "distance": distance,
            "n_hop_weights": n_hop_weights,
            "aggregation": aggregation,
            "n_components": n_components,
            "rng": rng,
            "use_rep": use_rep,
        },
        flavor_param_specs[flavor],
    )

    # Flavor-specific validations
    if flavor == "neighborhood":
        assert_isinstance(groups, str, name="groups")

        if min_niche_size is not None:
            assert_isinstance(min_niche_size, int, name="min_niche_size")

        if distance is not None and isinstance(distance, int) and distance < 1:
            raise ValueError(f"'distance' must be at least 1, got {distance}")

    elif flavor == "cellcharter":
        if distance is not None:
            assert_isinstance(distance, int, name="distance")
        if distance is not None and distance < 1:
            raise ValueError(f"'distance' must be at least 1, got {distance}")

        if aggregation is not None:
            assert_isinstance(aggregation, str, name="aggregation")
            assert_one_of(aggregation, ["mean", "variance"], name="aggregation")

        assert_isinstance(n_components, int, name="n_components")
        if n_components < 1:
            raise ValueError(f"'n_components' must be at least 1, got {n_components}")

        if use_rep is not None:
            assert_isinstance(use_rep, str, name="use_rep")

        # for mypy
        if resolutions is None:
            resolutions = [0.0]

    elif flavor == "spatialleiden":
        assert_isinstance(latent_connectivities_key, str, name="latent_connectivities_key")
        assert_isinstance(spatial_connectivities_key, str, name="spatial_connectivities_key")

        assert_isinstance(layer_ratio, (float, int), name="layer_ratio")
        assert_isinstance(n_iterations, int, name="n_iterations")
        if not (
            isinstance(use_weights, bool)
            or (
                isinstance(use_weights, tuple)
                and len(use_weights) == 2
                and all(isinstance(x, bool) for x in use_weights)
            )
        ):
            raise TypeError(f"'use_weights' must be a bool or a tuple of two bools, got {use_weights!r}")

        if resolutions is None:
            resolutions = [1.0]

    assert_isinstance(inplace, bool, name="inplace")


def _check_unnecessary_args(flavor: str, param_dict: dict[str, Any], param_specs: dict[str, Any]) -> None:
    """
    Check for unnecessary arguments that were provided but not used by the given flavor.

    Parameters
    ----------
    flavor
        The flavor being used ('neighborhood', 'utag', 'cellcharter', or 'spatialleiden')
    param_dict
        Dictionary of parameter names to their values
    param_specs
        Dictionary with 'required', 'optional', and 'unused' parameter lists for the flavor
    """
    unnecessary_args = []

    for param_name in param_specs["unused"]:
        param_value = param_dict.get(param_name)

        # Special handling for parameters whose default is not None
        if param_name == "scale" and param_value is True:
            continue
        if param_name == "abs_nhood" and param_value is False:
            continue

        if param_value is not None:
            unnecessary_args.append(param_name)

    if unnecessary_args:
        logg.warning(
            f"Parameters {', '.join([f'{arg}' for arg in unnecessary_args])} are not used for flavor '{flavor}'.",
        )


############
### embedder classes
############


NicheEmbedder = Callable[[AnnData], Array]


def _nhood_profile_embedding(
    adata: AnnData,
    *,
    groups: str,
    spatial_connectivities_key: str,
    scale: bool,
    distance: int,
    abs_nhood: bool,
    n_hop_weights: list[float] | None,
) -> Array:
    """Neighborhood composition: how frequent each category is around each observation."""
    profile = nhood_aggregate(
        adata,
        groups=groups,
        connectivity_key=spatial_connectivities_key,
        hops=range(1, distance + 1),
        hop_weights=n_hop_weights,
        aggregation="sum" if abs_nhood else "mean",
    )
    # narrow -- one column per category -- and `sc.pp.scale` densifies anyway, so the
    # one-hot's sparseness is not worth carrying past here
    profile = to_dense(profile)
    return sc.pp.scale(profile, zero_center=True) if scale else profile


def _utag_embedding(adata: AnnData, *, spatial_connectivities_key: str, use_layer: str | None) -> Array:
    """Each observation inherits the mean features of its immediate neighbors."""
    aggregated = nhood_aggregate(adata, layer=use_layer, connectivity_key=spatial_connectivities_key, hops=(1,))
    return sc.pp.pca(aggregated)


def _nhop_pca_embedding(adata: AnnData, *, distance: int, aggregation: str, spatial_connectivities_key: str) -> Array:
    """Disjoint hop rings of aggregated features, concatenated and reduced."""
    # hop 0 is the observation's own features; the rings are disjoint and each keeps its
    # own columns, so this stays sparse when the features are
    blocks = _nhood_blocks(
        adata,
        connectivity_key=spatial_connectivities_key,
        hops=range(distance + 1),
        hop_mode="shell",
        aggregation=aggregation,
    )
    # this is `distance + 1` times the width of the features, so it is the one place
    # densifying costs; keep the container they came in, as CellCharter does
    if all(issparse(block) for block in blocks):
        aggregated = sparse_hstack(blocks, format="csr")
    else:
        aggregated = np.hstack([to_dense(block) for block in blocks])
    return sc.pp.pca(aggregated)


def _precomputed_embedding(adata: AnnData, *, obsm_key: str) -> Array:
    """An embedding that already exists in ``adata.obsm``."""
    assert_key_in_adata(adata, obsm_key, attr="obsm")
    return adata.obsm[obsm_key]


############
### clusterer classes
############


class _NicheClusterer(ABC):
    """Base class for clustering embeddings into niche assignments.

    Subclasses must implement :meth:`cluster`, which assigns cluster labels
    and stores them in ``adata.obs``.
    """

    @abstractmethod
    def cluster(self, adata: AnnData, embedding: Array) -> list[str]:
        """Adds column/s in adata.obs with the clustering done. Returns the names of the columns just added."""


@d.dedent
class _LeidenClusterer(_NicheClusterer):
    """Cluster embeddings using the Leiden algorithm.

    Parameters
    ----------
    n_neighbors
        Number of neighbors used to construct the kNN graph.
    resolutions
        Resolution parameter(s) for Leiden clustering. Can be a single
        float value or list of floats.
    base_colname
        Base name for columns added to ``adata.obs``. Resolution is
        appended to this to unique identify columns for each resolution.
    %(niche_leiden_params)s

    Notes
    -----
    A separate clustering is computed for each resolution, producing multiple
    niche annotation columns.
    """

    def __init__(
        self,
        n_neighbors: int,
        resolutions: float | list[float],
        base_colname: str = "niche_leiden",
        *,
        flavor: Literal["igraph", "leidenalg"] = "igraph",
        n_iterations: int = -1,
        rng: SeedLike | RNGLike | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.resolutions = resolutions if isinstance(resolutions, list) else [resolutions]
        self.base_colname = base_colname
        self.flavor = flavor
        self.n_iterations = n_iterations
        self.rng = np.random.default_rng(rng)

    def cluster(self, adata: AnnData, embedding: Array) -> list:
        # first create an adata object using the embedding provided
        adata_embedding = ad.AnnData(X=embedding, obs=pd.DataFrame(index=adata.obs.index))

        # required for leiden clustering (note: no dim reduction performed in original implementation)
        sc.pp.neighbors(adata_embedding, n_neighbors=self.n_neighbors, use_rep="X")

        # For each resolution, apply leiden on neighborhood profile. Each cluster label equals to a niche label
        niche_keys = []
        # every resolution is a separate clustering run, so seed each one independently
        resolution_rngs = self.rng.spawn(len(self.resolutions))
        for res, res_rng in zip(self.resolutions, resolution_rngs, strict=True):
            niche_key = f"{self.base_colname}_res={res}"
            niche_keys.append(niche_key)

            if niche_key in adata.obs.columns:
                logg.info(f"Overwriting existing column '{niche_key}'")

            # Default to the igraph backend so niche labels are reproducible across
            # versions; leidenalg is deprecated in scanpy and unstable on small graphs.
            # See scverse/squidpy#1260.
            leiden_kwargs: dict[str, Any] = {
                "flavor": self.flavor,
                "n_iterations": self.n_iterations,
                "random_state": legacy_random(res_rng),
            }
            # scanpy's igraph backend only supports undirected graphs and errors if
            # ``directed`` is left at the leidenalg default of True, so pin it to False.
            if self.flavor == "igraph":
                leiden_kwargs["directed"] = False
            sc.tl.leiden(adata_embedding, resolution=res, key_added=niche_key, **leiden_kwargs)

            adata.obs[niche_key] = list(
                adata_embedding.obs[niche_key]
            )  # since constrain all embedders to return embedding with numrows==numcells and in same order, this should be fine

        return niche_keys


@d.dedent
class _GMMClusterer(_NicheClusterer):
    """Cluster embeddings with a Gaussian mixture model.

    Parameters
    ----------
    n_components
        Number of mixture components.
    rng
        rng supplying the seed of every mixture fit.
    base_colname
        Name of the output column added to ``adata.obs``.

    Notes
    -----
    Cluster assignments are stored as categorical niche labels in ``adata.obs``.

    One instance may be reused for several fits (e.g. once per library when stratifying
    by ``library_key``). Each :meth:`cluster` call draws a fresh seed from ``rng``, so the
    fits are seeded independently while remaining reproducible as a sequence.
    """

    def __init__(
        self,
        n_components: int,
        rng: np.random.Generator,
        base_colname: str = "niche_gmm",
    ):
        self.n_components = n_components
        self.rng = rng
        self.base_colname = base_colname

    def cluster(self, adata: AnnData, embedding: Array) -> list:
        """Returns niche labels generated by GMM clustering.
        Compared to cellcharter this approach is simplified by using sklearn's GaussianMixture model without stability analysis.
        """
        # cluster concatenated matrix with GMM, each cluster label equals to a niche label
        gmm = GaussianMixture(
            n_components=self.n_components,
            random_state=legacy_random(self.rng),
            init_params="random_from_data",
        )
        gmm.fit(embedding)
        niches = gmm.predict(embedding)

        if self.base_colname in adata.obs.columns:
            logg.info(f"Overwriting existing column '{self.base_colname}'")

        adata.obs[self.base_colname] = pd.Categorical(niches)
        return [self.base_colname]


############
### postprocessing
############


def _postprocess_niche_results(
    adata: AnnData,
    result_columns: list[str],
    mask: pd.Series | None = None,
    min_niche_size: int | None = None,
    prefix: str | None = None,
) -> None:
    """Refine niche assignments in place, rewriting each column in ``result_columns``.

    Parameters
    ----------
    adata
        Annotated data matrix.
    result_columns
        Columns in ``adata.obs`` holding the niche assignments to refine.
    mask
        Boolean :class:`~pandas.Series` indexed like ``adata.obs``. Observations that
        are ``False`` get the label ``"not_a_niche"``, e.g.
        ``Series([False, False, True], index=["a", "b", "c"])``.
    min_niche_size
        Niches with fewer than this many observations are relabeled ``"not_a_niche"``.
    prefix
        Prepended to every niche label, used to keep labels unique across libraries.

    Notes
    -----
    Columns are modified in place, so the niche column name does not depend on
    which of these options were supplied.
    """
    if mask is None and min_niche_size is None and prefix is None:
        return

    for col in result_columns:
        # str, so that "not_a_niche" and prefixed labels can be assigned regardless of the clusterer's dtype
        labels = adata.obs[col].astype(str)

        if mask is not None:
            aligned = mask[mask.index.isin(adata.obs.index)]
            labels[~aligned] = "not_a_niche"

        if min_niche_size is not None:
            counts = labels.value_counts()
            too_small = counts[counts < min_niche_size].index
            labels[labels.isin(too_small)] = "not_a_niche"

        if prefix is not None:
            labels = prefix + labels

        adata.obs[col] = labels.astype("category")
