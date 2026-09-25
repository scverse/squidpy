from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from numbers import Real
from typing import Any, Literal

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from fast_array_utils.conv import to_dense
from fast_array_utils.types import HasArrayNamespace as Array
from numpy.typing import NDArray
from pandas.api.types import is_bool_dtype
from sklearn.base import clone
from sklearn.mixture import GaussianMixture
from spatialdata import SpatialData, sanitize_table
from spatialdata._logging import logger as logg

from squidpy._constants._constants import NicheDefinitions
from squidpy._docs import d, inject_docs
from squidpy._utils import (
    RNGLike,
    SeedLike,
    deprecated_randomness_param,
    legacy_random,
)
from squidpy._validators import assert_isinstance, assert_key_in_adata, assert_one_of
from squidpy.gr._clusterers import Clusterer, LeidenClusterer
from squidpy.gr._nhood import (
    _aggregate_over,
    _assert_hop_request,
    compute_hop_adjacency_matrices,
    nhood_aggregate,
)
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
def calculate_niche(  # noqa: PLR0917, deprecated and removed in v1.9.0
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
    library_key
        Deprecated, and removed with this function in v1.9.0; no flavor function takes it. Each
        library is fitted on its own and its labels are prefixed ``lib=<id>_``, so a niche in one
        library is unrelated to the same-numbered niche in another.

        Either way, build the graphs per library with :func:`~squidpy.gr.spatial_neighbors`.
        For niches shared across libraries, leave this unset. For niches per library, call a
        flavor function on each library's slice, which is what this does::

            adata.obs["niche"] = "not_a_niche"
            for lib in adata.obs["library"].unique():
                sub = adata[adata.obs["library"] == lib].copy()
                sq.gr.calculate_niche_utag(sub, resolutions=0.5)
                adata.obs.loc[sub.obs_names, "niche"] = f"lib={{lib}}_" + sub.obs["utag_niche_res=0.5"].astype(str)
    %(table_key)s
    mask
        Boolean array to filter cells which won't get assigned to a niche. Spelled
        ``cluster_mask`` on the three flavors that build an embedding, and
        `{fla.SPATIALLEIDEN.s!r}` raises rather than accepting one it cannot honour.
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
        # `calculate_niche` is wrapped by `@deprecated_randomness_param`, so 2 lands in
        # squidpy/_utils.py rather than on the caller the deprecation is addressed to
        stacklevel=3,
    )

    # cellcharter-only defaults stay guarded: filling them for other flavors would trip
    # the "not used for flavor" warning in _check_unnecessary_args
    if mask is not None and flavor == "spatialleiden":
        raise ValueError(
            "'mask' keeps masked observations out of the niche fit, which 'spatialleiden' cannot "
            "do: it clusters the graphs themselves, so an observation either takes part or loses "
            "its edges. It is 'cluster_mask' on the other three flavors."
        )

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

    def run(target: AnnData | SpatialData, *, copy: bool, target_key: str | None) -> AnnData | None:
        if flavor == "neighborhood":
            return calculate_niche_neighborhood(
                target,
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
                cluster_mask=mask,
                copy=copy,
                table_key=target_key,
                n_iterations=n_iterations,
                rng=rng,
            )

        elif flavor == "utag":
            return calculate_niche_utag(
                target,
                resolutions=resolutions,
                n_neighbors=n_neighbors,
                use_layer=None,
                use_rep=use_rep,
                spatial_connectivities_key=spatial_connectivities_key,
                embedding_key_added="niche_embedding",
                cluster_mask=mask,
                min_niche_size=min_niche_size,
                copy=copy,
                table_key=target_key,
                n_iterations=n_iterations,
                rng=rng,
            )

        elif flavor == "cellcharter":
            return calculate_niche_cellcharter(
                target,
                distance=distance,
                aggregation=aggregation,
                rng=rng,
                spatial_connectivities_key=spatial_connectivities_key,
                n_clusters=n_components,
                use_rep=use_rep,
                embedding_key_added="niche_embedding",
                cluster_mask=mask,
                min_niche_size=min_niche_size,
                copy=copy,
                table_key=target_key,
            )

        elif flavor == "spatialleiden":
            return calculate_niche_spatialleiden(
                target,
                resolutions=resolutions,
                latent_connectivities_key=latent_connectivities_key,
                spatial_connectivities_key=spatial_connectivities_key,
                layer_ratio=layer_ratio,
                n_iterations=n_iterations,
                use_weights=use_weights,
                rng=rng,
                min_niche_size=min_niche_size,
                copy=copy,
                table_key=target_key,
            )
        return None

    if library_key is None:
        return run(data, copy=not inplace, target_key=table_key)

    warnings.warn(
        "'library_key' is deprecated and will be removed with `calculate_niche` in squidpy v1.9.0. "
        "It fits a separate model per library, so niche labels are not shared between libraries. "
        "Either way, build the graphs per library with `spatial_neighbors(..., library_key=...)`. "
        "For niches shared across libraries, leave `library_key` unset. For niches per library, "
        "call a flavor function on each library's slice, e.g. "
        "`calculate_niche_utag(adata[adata.obs['library'] == lib].copy(), ...)`, and copy the "
        "labels back; the `calculate_niche` docstring shows how.",
        FutureWarning,
        stacklevel=3,  # the same frame as the deprecation above
    )
    if flavor == "cellcharter":
        columns = ["cellcharter_niche"]
    else:
        stem = {"neighborhood": "nhood_niche", "utag": "utag_niche", "spatialleiden": "spatialleiden"}[flavor]
        columns = [f"{stem}_res={res}" for res in _resolution_values(resolutions, pairs_ok=flavor == "spatialleiden")]
    return _niches_per_library(
        data,
        library_key=library_key,
        columns=columns,
        table_key=table_key,
        copy=not inplace,
        run=lambda lib_adata: run(lib_adata, copy=False, target_key=None),
    )


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
    key_added: str = "nhood_niche",
    min_niche_size: int | None = None,
    cluster_mask: pd.Series | None = None,
    copy: bool = False,
    table_key: str | None = None,
    flavor: Literal["igraph", "leidenalg"] = "igraph",
    n_iterations: int = -1,
    rng: SeedLike | RNGLike | None = None,
) -> AnnData | None:
    """Compute spatial niches from local cell-type composition.

    Follows monkeybread's
    https://github.com/immunitastx/monkeybread/blob/main/src/monkeybread/calc/_neighborhood_profile.py

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
    %(niche_cluster_mask)s
    %(niche_key_added_stem)s
    %(niche_common_params)s
    %(table_key)s
    %(niche_leiden_params)s

    Returns
    -------
    If ``copy=True``, returns a copy of ``adata`` with the neighborhood profile
    stored in ``.obsm[embedding_key_added]`` and niche
    assignments added to ``.obs``. Otherwise, modifies ``adata`` in place and returns ``None``.

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

    clusterers = _leiden_clusterers(
        base_colname=key_added,
        resolutions=resolutions,
        n_neighbors=n_neighbors,
        flavor=flavor,
        n_iterations=n_iterations,
    )

    return calculate_niche_custom(
        data,
        embedder=embedder,
        clusterers=clusterers,
        rng=rng,
        embedding_key_added=embedding_key_added,
        min_niche_size=min_niche_size,
        cluster_mask=cluster_mask,
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
    use_rep: str | None = None,
    key_added: str = "utag_niche",
    min_niche_size: int | None = None,
    cluster_mask: pd.Series | None = None,
    copy: bool = False,
    table_key: str | None = None,
    flavor: Literal["igraph", "leidenalg"] = "igraph",
    n_iterations: int = -1,
    rng: SeedLike | RNGLike | None = None,
) -> AnnData | None:
    """Compute spatial niches from UTAG-style feature aggregation.

    UTAG :cite:`kim2022`, adapted from
    https://github.com/ElementoLab/utag/blob/main/utag/segmentation.py

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
    3. Reduces that matrix with PCA; the scores are the niche embedding.
    4. Constructs a k-nearest-neighbor graph in that embedding space.
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
    %(niche_cluster_mask)s
    use_rep
        Key in :attr:`anndata.AnnData.obsm` holding the representation to aggregate over each
        neighborhood, or ``'X'`` for :attr:`~anndata.AnnData.X`. Taken as already reduced, so the
        PCA of the aggregate is skipped. Rejected together with ``use_layer``.
    %(niche_key_added_stem)s
    %(niche_common_params)s
    %(table_key)s
    %(niche_leiden_params)s

    Returns
    -------
    If ``copy=True``, returns a copy of ``adata`` with the aggregated features in
    ``.obsm[embedding_key_added]``, PCA-reduced unless ``use_rep``
    was given, and niche assignments added to ``.obs``. Otherwise, modifies ``adata`` in place
    and returns ``None``.

    """

    embedder = partial(
        _utag_embedding,
        spatial_connectivities_key=spatial_connectivities_key,
        use_layer=use_layer,
        use_rep=use_rep,
    )

    clusterers = _leiden_clusterers(
        base_colname=key_added,
        resolutions=resolutions,
        n_neighbors=n_neighbors,
        flavor=flavor,
        n_iterations=n_iterations,
    )

    return calculate_niche_custom(
        data,
        embedder=embedder,
        clusterers=clusterers,
        rng=rng,
        embedding_key_added=embedding_key_added,
        min_niche_size=min_niche_size,
        cluster_mask=cluster_mask,
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
    n_clusters: int = 10,
    n_pca_components: int | None = None,
    n_jobs: int | None = None,
    use_rep: str | None = None,
    embedding_key_added: str = "niche_embedding",
    key_added: str = "cellcharter_niche",
    min_niche_size: int | None = None,
    cluster_mask: pd.Series | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute spatial niches using a CellCharter-style embedding and GMM :cite:`varrone2023`.

    CellCharter recommends a dimensionality-reduced embedding such as scVI, passed as
    ``use_rep``; PCA of ``adata.X`` is the fallback when it is not given. The mixture
    model is scikit-learn's, not CellCharter's torchgmm, so partitions will not match theirs.

    This method identifies niches by clustering an embedding that represents
    each observation together with information from its surrounding spatial
    neighborhood. Unlike :func:`calculate_niche_neighborhood`, which builds
    an embedding from categorical cell-type composition, this approach uses a
    continuous feature representation and a Gaussian mixture model (GMM) for
    clustering.

    A spatial connectivity graph must already be present in
    ``adata.obsp[spatial_connectivities_key]``. This function does not construct
    the graph itself.

    The method:

    1. Takes a reduced representation of every observation: ``adata.obsm[use_rep]``
       if given, otherwise the first ``n_pca_components`` principal components of ``adata.X``,
       restricted to ``adata.var["highly_variable"]`` when that column is present.
    2. Aggregates that representation over each disjoint hop ring of the spatial
       graph, from direct neighbors through ``distance`` graph hops, according to
       ``aggregation``.
    3. Concatenates the observation's own representation with every ring's aggregate.
    4. Fits a Gaussian mixture model with ``n_clusters`` mixture components.
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
        Seeds the Gaussian mixture clustering step.
    %(niche_spatial_conn_key)s
    n_clusters
        Number of Gaussian mixture components, and therefore the number of niche
        labels produced per library or dataset.
    n_pca_components
        Number of principal components of ``adata.X`` aggregated over the hop rings. ``None``
        uses 10, the latent width scVI defaults to, or one fewer than the smallest dimension of
        ``adata.X`` when that is narrower. Named for the PCA because
        :class:`~sklearn.mixture.GaussianMixture` spells its cluster count ``n_components``
        too; here that is ``n_clusters``. Rejected together with ``use_rep``, which supplies an
        already reduced embedding and so skips the PCA.
    %(n_jobs_threads)s
    use_rep
        Key in :attr:`anndata.AnnData.obsm` holding a reduced representation, such as an scVI
        latent, or ``'X'`` for :attr:`~anndata.AnnData.X`. Aggregated over the hop rings in place
        of the PCA of ``adata.X``.
    %(niche_cluster_mask)s
    key_added
        Name of the :attr:`anndata.AnnData.obs` column the labels are written to.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``copy=True``, returns a copy of ``adata`` with the embedding stored in
    ``.obsm[embedding_key_added]`` and GMM-based niche
    assignments added to ``.obs``. Otherwise, modifies ``adata`` in place and returns ``None``.

    """

    if use_rep is None:
        logg.warning(
            "CellCharter recommends to use a dimensionality reduced embedding of the data, e.g. a scVI embedding. Since 'use_rep' is not provided, PCA will be used as proxy - performance may be suboptimal."
        )
    elif n_pca_components is not None:
        raise ValueError("'n_pca_components' sizes the PCA, which 'use_rep' replaces; pass one or the other")

    embedder = partial(
        _nhop_pca_embedding,
        distance=distance,
        aggregation=aggregation,
        spatial_connectivities_key=spatial_connectivities_key,
        use_rep=use_rep,
        n_pca_components=n_pca_components,
        n_jobs=n_jobs,
    )

    # `GaussianMixture` is a `Clusterer` as it stands, so this flavor needs no wrapper
    clusterers = {key_added: GaussianMixture(n_components=n_clusters)}

    return calculate_niche_custom(
        data,
        embedder=embedder,
        clusterers=clusterers,
        rng=rng,
        embedding_key_added=embedding_key_added,
        min_niche_size=min_niche_size,
        cluster_mask=cluster_mask,
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
    key_added: str = "spatialleiden",
    min_niche_size: int | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> AnnData | None:
    """Compute niche assignments using the SpatialLeiden algorithm.

    This is a wrapper around SpatialLeiden :cite:`muellerboetticher2025`, which takes
    :class:`~anndata.AnnData` as input and works with two layers; one latent space and one
    spatial layer. Adapted from https://github.com/HiDiHlabs/SpatialLeiden/.

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
        Each resolution is clustered with an independent rng derived from it.
    %(niche_min_niche_size)s
    %(niche_key_added_stem)s
    %(copy)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.
    """
    resolution_list = _resolution_values(resolutions, pairs_ok=True)
    rng = np.random.default_rng(rng)

    def run(adata: AnnData) -> None:
        _spatialleiden_once(
            adata,
            key_added=key_added,
            resolution_list=resolution_list,
            rng=rng,
            latent_connectivities_key=latent_connectivities_key,
            spatial_connectivities_key=spatial_connectivities_key,
            layer_ratio=layer_ratio,
            n_iterations=n_iterations,
            use_weights=use_weights,
            min_niche_size=min_niche_size,
        )

    return _on_table(data, table_key=table_key, copy=copy, work=run)


@d.dedent
def calculate_niche_custom(
    data: AnnData | SpatialData,
    *,
    embedder: NicheEmbedder,
    clusterers: Mapping[str, Clusterer],
    rng: SeedLike | RNGLike | None = None,
    embedding_key_added: str = "niche_embedding",
    min_niche_size: int | None = None,
    cluster_mask: pd.Series | None = None,
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
    clusterers
        The clusterer labelling each ``adata.obs`` column, keyed by name.
    rng
        Seeds every fit.
    %(niche_common_params)s
    %(table_key)s

    Returns
    -------
    If ``copy = True``, returns a copy of ``adata`` with niche annotations added to ``.obs``.
    Otherwise, modifies ``adata`` in place and returns ``None``.

    Notes
    -----
    One embedding and one clustering over everything, so the labels mean the same thing for
    every observation. Per-library fitting lives in :func:`calculate_niche_spatialleiden`.

    See Also
    --------
    calculate_niche_neighborhood : Convenience wrapper for neighborhood flavor niche analysis.
    calculate_niche_utag : Convenience wrapper for utag flavor niche analysis.
    calculate_niche_cellcharter : Convenience wrapper for cellcharter flavor niche analysis.
    calculate_niche_spatialleiden : Convenience wrapper for spatialleiden flavor niche analysis.
    """

    # AnnData accepts `obsm[None]` and only fails at `write_h5ad`
    if not isinstance(embedding_key_added, str) or len(embedding_key_added) == 0:
        raise ValueError(f"'embedding_key_added' must be a non-empty string, got {embedding_key_added!r}")

    rng = np.random.default_rng(rng)

    def run(adata: AnnData) -> None:
        keep = None
        if cluster_mask is not None:
            if not is_bool_dtype(cluster_mask):
                raise TypeError(f"'cluster_mask' must be a boolean Series, got dtype '{cluster_mask.dtype}'")
            if not cluster_mask.index.isin(adata.obs_names).any():
                raise ValueError("'cluster_mask' shares no index value with 'adata.obs', so it masks nothing")
            # observations the mask omits are kept, as the documented three-entry example reads
            keep = cluster_mask.reindex(adata.obs_names, fill_value=True).to_numpy(dtype=bool)
            if not keep.any():
                raise ValueError("'cluster_mask' excludes every observation, so no niche could be assigned")

        # called here, not via a helper: another frame would shift the warnings' stacklevel
        embedding = embedder(adata)
        adata.obsm[embedding_key_added] = embedding
        _fit_clusterers(adata, embedding, clusterers, rng, keep=keep, min_niche_size=min_niche_size)

    return _on_table(data, table_key=table_key, copy=copy, work=run)


def _validate_niche_args(  # noqa: PLR0917, deprecated and removed in v1.9.0
    data: AnnData | SpatialData,
    flavor: Literal["neighborhood", "utag", "cellcharter", "spatialleiden"],
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

    if n_neighbors is not None:
        assert_isinstance(n_neighbors, int, name="n_neighbors")

    if n_hop_weights is not None:
        assert_isinstance(n_hop_weights, list, name="n_hop_weights")

    assert_isinstance(scale, bool, name="scale")

    assert_isinstance(abs_nhood, bool, name="abs_nhood")

    # Define parameters used by each flavor
    flavor_param_specs = {
        "neighborhood": {
            "required": ["groups", "n_neighbors", "resolutions", "spatial_connectivities_key"],
            "optional": [
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
            "optional": ["rng", "n_iterations", "use_rep"],
            "unused": [
                "groups",
                "scale",
                "abs_nhood",
                "distance",
                "n_hop_weights",
                "aggregation",
                "n_components",
                "latent_connectivities_key",
                "layer_ratio",
                "use_weights",
            ],
        },
        "cellcharter": {
            "required": ["distance", "aggregation", "spatial_connectivities_key"],
            # `rng` is optional: `None` is a valid value meaning "draw from OS entropy"
            "optional": ["n_components", "use_rep", "rng"],
            "unused": [
                "groups",
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
            "unused": ["groups", "scale", "abs_nhood", "n_neighbors", "n_hop_weights", "use_rep"],
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


def _has_edge_weights(adata: AnnData, key: str) -> bool:
    """True when the graph carries values other than 0 and 1."""
    weights = adata.obsp[key].data
    return bool(weights.size) and not np.array_equal(weights, weights.astype(bool))


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
    if distance < 1:
        raise ValueError(f"'distance' must be >= 1, got {distance}")
    if n_hop_weights is not None and len(n_hop_weights) != distance:
        raise ValueError(
            f"'n_hop_weights' has {len(n_hop_weights)} value(s) but 'distance' is {distance}. "
            f"Earlier versions padded a short list with its last value; pass all {distance}."
        )

    if _has_edge_weights(adata, spatial_connectivities_key):
        warnings.warn(
            f"'{spatial_connectivities_key}' carries non-binary edge weights. They weight hop 1, "
            "so the profile is the share of connectivity reaching each category rather than the "
            "share of neighboring cells. Set them to 1 for a plain composition. Hops past the "
            "first are unweighted either way, since only edges carry weights.",
            UserWarning,
            # `_on_table`, the pipeline's `run` and the embedder sit in between. Still one short
            # through `calculate_niche`.
            stacklevel=6,
        )

    profile = nhood_aggregate(
        adata,
        groups=groups,
        connectivity_key=spatial_connectivities_key,
        hops=range(1, distance + 1),
        hop_weights=n_hop_weights,
        aggregation="sum" if abs_nhood else "mean",
    )
    # one column per category, and `sc.pp.scale` densifies anyway
    profile = to_dense(profile)
    # monkeybread counts per cell with a `Counter`; this is the same profile as a one-hot
    # product. Scaling follows theirs, see
    # https://monkeybread.readthedocs.io/en/latest/notebooks/tutorial.html#niche-analysis
    return sc.pp.scale(profile, zero_center=True) if scale else profile


def _utag_embedding(
    adata: AnnData, *, spatial_connectivities_key: str, use_layer: str | None, use_rep: str | None
) -> Array:
    """Each observation inherits the mean features of its immediate neighbors."""
    aggregated = nhood_aggregate(
        adata, layer=use_layer, use_rep=use_rep, connectivity_key=spatial_connectivities_key, hops=(1,)
    )
    # already reduced, and on one basis across libraries. This PCA would refit it per library.
    return to_dense(aggregated) if use_rep is not None else sc.pp.pca(aggregated)


def _nhop_pca_embedding(
    adata: AnnData,
    *,
    distance: int,
    aggregation: str,
    spatial_connectivities_key: str,
    use_rep: str | None = None,
    n_pca_components: int | None = None,
    n_jobs: int | None = None,
) -> Array:
    """Reduced features and their ring aggregates, concatenated as in CellCharter."""
    if aggregation not in ("mean", "variance"):
        raise ValueError(f"'aggregation' must be 'mean' or 'variance', got {aggregation!r}")
    if distance < 1:
        raise ValueError(f"'distance' must be >= 1, got {distance}")

    hops = range(distance + 1)
    _assert_hop_request(adata, spatial_connectivities_key, hops)

    if _has_edge_weights(adata, spatial_connectivities_key):
        warnings.warn(
            f"'{spatial_connectivities_key}' carries non-binary edge weights, which this flavor "
            "ignores entirely: the hop rings are boolean, as in CellCharter. The 'neighborhood' "
            "flavor weights hop 1 by them.",
            UserWarning,
            # `_on_table`, the pipeline's `run` and the embedder sit in between. Still one short
            # through `calculate_niche`.
            stacklevel=6,
        )

    # CellCharter aggregates an already reduced representation, so PCA comes first: the rings then
    # aggregate a narrow dense matrix instead of `distance + 1` copies of every gene
    if use_rep is None:
        features = _pca_features(adata, n_pca_components)
    elif use_rep == "X":  # the spelling `scanpy.pp.neighbors` takes
        features = to_dense(adata.X)
    else:
        assert_key_in_adata(adata, use_rep, attr="obsm")
        features = to_dense(adata.obsm[use_rep])

    rings = compute_hop_adjacency_matrices(adata.obsp[spatial_connectivities_key], distance, n_jobs=n_jobs)

    # hop 0 is the observation itself, so it heads the concatenation and every ring follows.
    # Filling a preallocated block keeps the features' dtype and frees each aggregate as it lands
    width = features.shape[1]
    # a ring mean is not a count, so integer features cannot hold the block they are written into
    dtype = np.result_type(features.dtype, np.float32)
    embedding = np.empty((features.shape[0], width * (len(rings) + 1)), dtype=dtype)
    embedding[:, :width] = features
    for position, ring in enumerate(rings, start=1):
        embedding[:, position * width : (position + 1) * width] = _aggregate_over(ring, features, aggregation)
    return embedding


def _pca_features(adata: AnnData, n_pca_components: int | None) -> Array:
    """The PCA that stands in for a reduced representation when ``use_rep`` is not given."""
    # scanpy's `pca(adata)` masks by `highly_variable` but also writes to `adata`; mask here instead
    if "highly_variable" in adata.var:
        X = adata.X[:, adata.var["highly_variable"].to_numpy()]
    else:
        X = adata.X
    ceiling = min(X.shape)
    if ceiling <= 1:
        # a one-marker panel is its own reduction, and PCA cannot return a component here
        return to_dense(X)
    if n_pca_components is None:
        # 10 matches scVI's default latent width, the input CellCharter is designed around
        return sc.pp.pca(X, n_comps=min(10, ceiling - 1))
    if not 1 <= n_pca_components < ceiling:
        raise ValueError(
            f"'n_pca_components' must be between 1 and {ceiling - 1}, the features PCA runs on "
            f"({'highly variable ' if 'highly_variable' in adata.var else ''}genes and observations), "
            f"got {n_pca_components}"
        )
    return sc.pp.pca(X, n_comps=n_pca_components)


def _resolution_values(resolutions: Any, *, pairs_ok: bool) -> list[Any]:
    """The resolution values, one per clustering run and one per labelled column.

    *pairs_ok* admits the ``(latent, spatial)`` pair that only SpatialLeiden takes.
    """
    if isinstance(resolutions, str | tuple) or not isinstance(resolutions, Iterable):
        values = [resolutions]
    else:
        values = list(resolutions)

    expected = "numbers or (latent, spatial) pairs of numbers" if pairs_ok else "numbers"
    for value in values:
        if isinstance(value, tuple):
            if not pairs_ok:
                raise TypeError(f"'resolutions' got the pair {value}, which only the 'spatialleiden' flavor takes")
            if len(value) != 2 or not all(isinstance(x, Real) for x in value):
                raise TypeError(f"'resolutions' got {value!r}, which is not a pair of numbers")
        elif not isinstance(value, Real):
            raise TypeError(f"'resolutions' must be {expected}, got {value!r}")

    if len(values) == 0:
        raise ValueError("'resolutions' is empty, so there is nothing to cluster")
    repeated = sorted({str(v) for v in values if values.count(v) > 1})
    if len(repeated) > 0:
        raise ValueError(f"'resolutions' repeats {', '.join(repeated)}, but each value labels its own column")
    return values


def _leiden_clusterers(
    *,
    base_colname: str,
    resolutions: float | Sequence[float],
    n_neighbors: int,
    flavor: Literal["igraph", "leidenalg"],
    n_iterations: int,
) -> dict[str, Clusterer]:
    """One Leiden clusterer per requested resolution, keyed by the column it labels."""
    values = _resolution_values(resolutions, pairs_ok=False)
    return {
        f"{base_colname}_res={res}": LeidenClusterer(
            n_neighbors=n_neighbors, resolution=res, flavor=flavor, n_iterations=n_iterations
        )
        for res in values
    }


def _fit_clusterers(
    adata: AnnData,
    embedding: Array,
    clusterers: Mapping[str, Clusterer],
    rng: np.random.Generator,
    *,
    keep: NDArray[np.bool_] | None = None,
    min_niche_size: int | None = None,
) -> list[str]:
    """Fit each clusterer on *embedding* and write its labels, returning the column names.

    *keep* restricts the fit to those observations; the rest are labelled ``'not_a_niche'``
    without having taken part in it.
    """
    for column, clusterer in clusterers.items():
        # `isinstance` sees method presence only, so check the one parameter the pipeline sets:
        # a deterministic estimator such as DBSCAN satisfies the protocol and then rejects it
        if not isinstance(clusterer, Clusterer):
            raise TypeError(f"clusterer for '{column}' must implement fit_predict, get_params and set_params")
        if "random_state" not in clusterer.get_params():
            raise TypeError(f"clusterer for '{column}' has no 'random_state', so the pipeline cannot seed its fits")
    # one rng per clusterer, so a resolution sweep is seeded independently of its length
    rngs = rng.spawn(len(clusterers))
    for (column, clusterer), rng in zip(clusterers.items(), rngs, strict=True):
        if column in adata.obs.columns:
            logg.info(f"Overwriting existing column '{column}'")
        # a fresh clone per fit, so the estimator handed in is never mutated
        fit = clone(clusterer).set_params(random_state=legacy_random(rng))
        if keep is None:
            labels = np.asarray(fit.fit_predict(embedding)).astype(str)
        else:
            labels = np.full(adata.n_obs, "not_a_niche", dtype=object)
            labels[keep] = np.asarray(fit.fit_predict(embedding[keep])).astype(str)
        adata.obs[column] = _niche_labels(labels, min_niche_size)
    return list(clusterers)


def _niche_labels(labels: NDArray[np.str_] | NDArray[np.object_], min_niche_size: int | None) -> pd.Categorical:
    """String niche labels as a categorical, niches under *min_niche_size* relabeled ``'not_a_niche'``."""
    if min_niche_size is not None:
        niches, counts = np.unique(labels, return_counts=True)
        labels = np.where(np.isin(labels, niches[counts < min_niche_size]), "not_a_niche", labels)
    # the categories order the legend and the colors, so keep `sc.tl.leiden`'s: 0, 1, 2, ... 10
    return pd.Categorical(labels, categories=sorted(np.unique(labels), key=_label_order))


def _label_order(label: str) -> tuple[bool, int, str]:
    """Numeric labels in numeric order, then any other label as text."""
    try:
        return False, int(label), ""
    except ValueError:
        return True, 0, label


############
### postprocessing
############


# TODO(v1.9.0): delete with `calculate_niche`, its only caller; no flavor function takes `library_key`
def _niches_per_library(
    data: AnnData | SpatialData,
    *,
    library_key: str,
    columns: list[str],
    table_key: str | None,
    copy: bool,
    run: Callable[[AnnData], object],
) -> AnnData | None:
    """v1.8.3's ``library_key``: each library fitted on its own, labels prefixed ``lib=<id>_``."""

    def loop(adata: AnnData) -> None:
        assert_key_in_adata(adata, library_key, attr="obs")
        # fresh per call, so a rerun cannot keep the previous run's labels; an observation with
        # no library keeps "not_a_niche", as in v1.8.3
        labels = {col: np.full(adata.n_obs, "not_a_niche", dtype=object) for col in columns}
        libraries = adata.obs[library_key]
        for lib_id in libraries.dropna().unique():
            at = np.flatnonzero((libraries == lib_id).to_numpy())
            lib_adata = adata[at].copy()
            run(lib_adata)
            for col in columns:
                values = lib_adata.obs[col].astype(str).to_numpy()
                labels[col][at] = [v if v == "not_a_niche" else f"lib={lib_id}_{v}" for v in values]
        for col, values in labels.items():
            adata.obs[col] = pd.Categorical(values)

    return _on_table(data, table_key=table_key, copy=copy, work=loop)


def _on_table(
    data: AnnData | SpatialData,
    *,
    table_key: str | None,
    copy: bool,
    work: Callable[[AnnData], None],
) -> AnnData | None:
    """Run *work* on the table, taking it out of a SpatialData and putting it back."""
    orig_adata = extract_adata_if_sdata(data, table_key=table_key)
    adata = orig_adata.copy() if copy else orig_adata
    work(adata)

    # For SpatialData, the column names shouldn't have = sign. Hence, run sanitize_table.
    # TODO: In future, change the naming standard of any niche columns added to not have '=' to be compatible with spatialdata naming
    if isinstance(data, SpatialData):
        sanitize_table(adata)

    return adata if copy else None


def _spatialleiden_once(
    adata: AnnData,
    *,
    key_added: str,
    resolution_list: list[Any],
    rng: np.random.Generator,
    latent_connectivities_key: str,
    spatial_connectivities_key: str,
    layer_ratio: float,
    n_iterations: int,
    use_weights: bool | tuple[bool, bool],
    min_niche_size: int | None,
) -> None:
    """One SpatialLeiden run per resolution, in place."""
    try:
        import spatialleiden as sl
    except ImportError as e:
        msg = "Please install the spatialleiden algorithm: `pip install squidpy[leiden]` or `conda install bioconda::spatialleiden` or `pip install spatialleiden`."
        raise ImportError(msg) from e

    # every resolution is a separate clustering run, so seed each one independently
    resolution_rngs = rng.spawn(len(resolution_list))
    for res, res_rng in zip(resolution_list, resolution_rngs, strict=True):
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
            key_added=f"{key_added}_res={res}",
        )
        # `sl.spatialleiden` writes its labels into `obs` rather than returning them, so read them
        # back and finish them as every other flavor's are: string labels, small niches relabeled
        column = f"{key_added}_res={res}"
        adata.obs[column] = _niche_labels(adata.obs[column].to_numpy().astype(str), min_niche_size)
