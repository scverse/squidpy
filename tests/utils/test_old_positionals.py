"""Arguments that became keyword-only still accept an old positional call, with a warning (#1288)."""

from __future__ import annotations

import inspect
import warnings
from importlib import import_module

import numpy as np
import pytest
from anndata import AnnData

import squidpy as sq
from squidpy.gr import interaction_matrix

# Every shimmed function with the parameters it took positionally after the first one in v1.8.3.
# fmt: off
V1_8_3_POSITIONALS = [
    ("squidpy.gr._build", "spatial_neighbors", ("spatial_key", "elements_to_coordinate_systems", "table_key", "library_key", "coord_type", "n_neighs", "radius", "delaunay", "n_rings", "percentile", "transform", "set_diag", "key_added", "copy", "n_jobs")),
    ("squidpy.gr._build", "spatial_neighbors_from_builder", ("builder",)),
    ("squidpy.gr._build", "mask_graph", ("table_key", "polygon_mask", "negative_mask", "spatial_key", "key_added", "copy")),
    ("squidpy.gr._ligrec", "ligrec", ("cluster_key", "interactions", "complex_policy", "threshold", "corr_method", "corr_axis", "use_raw", "copy", "key_added", "gene_symbols")),
    ("squidpy.gr._nhood", "nhood_enrichment", ("cluster_key", "library_key", "connectivity_key", "n_perms", "numba_parallel", "seed", "copy", "n_jobs", "backend", "show_progress_bar")),
    ("squidpy.gr._nhood", "centrality_scores", ("cluster_key", "score", "connectivity_key", "copy", "n_jobs", "backend", "show_progress_bar")),
    ("squidpy.gr._nhood", "interaction_matrix", ("cluster_key", "connectivity_key", "normalized", "copy", "weights")),
    ("squidpy.gr._ppatterns", "spatial_autocorr", ("connectivity_key", "genes", "mode", "transformation", "n_perms", "two_tailed", "corr_method", "attr", "layer", "seed", "use_raw", "copy", "n_jobs", "backend", "show_progress_bar")),
    ("squidpy.gr._ppatterns", "co_occurrence", ("cluster_key", "spatial_key", "interval", "copy")),
    ("squidpy.gr._ripley", "ripley", ("cluster_key", "mode", "spatial_key", "metric", "n_neigh", "n_simulations", "n_observations", "max_dist", "n_steps", "seed", "copy")),
    ("squidpy.gr._sepal", "sepal", ("max_neighs", "genes", "n_iter", "dt", "thresh", "connectivity_key", "spatial_key", "layer", "use_raw", "copy", "n_jobs", "show_progress_bar")),
    ("squidpy.im._feature", "calculate_image_features", ("img", "layer", "library_id", "features", "features_kwargs", "key_added", "copy", "n_jobs", "backend", "show_progress_bar")),
    ("squidpy.im._process", "process", ("layer", "library_id", "method", "chunks", "lazy", "layer_added", "channel_dim", "copy", "apply_kwargs")),
    ("squidpy.pl._graph", "centrality_scores", ("cluster_key", "score", "legend_kwargs", "palette", "figsize", "dpi", "save")),
    ("squidpy.pl._graph", "interaction_matrix", ("cluster_key", "annotate", "method", "title", "cmap", "palette", "cbar_kwargs", "figsize", "dpi", "save", "ax")),
    ("squidpy.pl._graph", "nhood_enrichment", ("cluster_key", "mode", "annotate", "method", "title", "cmap", "palette", "cbar_kwargs", "figsize", "dpi", "save", "ax")),
    ("squidpy.pl._graph", "ripley", ("cluster_key", "mode", "plot_sims", "palette", "figsize", "dpi", "save", "ax", "legend_kwargs")),
    ("squidpy.pl._graph", "co_occurrence", ("cluster_key", "palette", "clusters", "figsize", "dpi", "save", "legend_kwargs")),
    ("squidpy.pl._ligrec", "ligrec", ("cluster_key", "source_groups", "target_groups", "means_range", "pvalue_threshold", "remove_empty_interactions", "remove_nonsig_interactions", "dendrogram", "alpha", "swap_axes", "title", "figsize", "dpi", "save")),
    ("squidpy.pl._spatial", "spatial_scatter", ("shape",)),
    ("squidpy.pl._spatial", "spatial_segment", ("seg_cell_id", "seg", "seg_key", "seg_contourpx", "seg_outline")),
    ("squidpy.pl._utils", "extract", ("obsm_key", "prefix")),
    ("squidpy.pl._var_by_distance", "var_by_distance", ("var", "anchor_key", "design_matrix_key", "stack_vars", "covariate", "order", "show_scatter", "color", "line_palette", "scatter_palette", "dpi", "figsize", "save", "title", "axis_label", "return_ax", "regplot_kwargs", "scatterplot_kwargs")),
    ("squidpy.tl._sliding_window", "sliding_window", ("library_key", "window_size", "overlap", "coord_columns", "sliding_window_key", "spatial_key", "drop_partial_windows", "copy")),
    ("squidpy.tl._var_by_distance", "var_by_distance", ("groups", "cluster_key", "library_key", "library_id", "design_matrix_key", "covariates", "metric", "spatial_key", "copy")),
]
# fmt: on


@pytest.mark.parametrize(("module", "name", "old"), V1_8_3_POSITIONALS, ids=lambda x: x if isinstance(x, str) else "")
def test_formerly_positional_args_are_now_keyword_only(module: str, name: str, old: tuple[str, ...]) -> None:
    # The lint rule (PLR0917) only catches functions with too many positional arguments; this checks
    # that every argument that was positional in v1.8.3 is now keyword-only, however short the signature.
    func = getattr(import_module(module), name)
    assert hasattr(func, "__wrapped__"), f"{name} is missing its @old_positionals shim"
    params = inspect.signature(inspect.unwrap(func)).parameters
    # `seed` was renamed to `rng`; names no longer in the signature are deprecated no-ops the shim drops.
    survivors = []
    for name_1_8_3 in old:
        current = "rng" if name_1_8_3 == "seed" and "rng" in params else name_1_8_3
        if current in params:
            survivors.append(current)
    still_positional = [n for n in survivors if params[n].kind is not inspect.Parameter.KEYWORD_ONLY]
    assert not still_positional, f"{name}: expected keyword-only, still positional: {still_positional}"


def test_old_positional_call_warns_and_still_works(nhood_data: AnnData) -> None:
    # `cluster_key`, `connectivity_key`, `normalized` and `copy` were positional before #1288
    with pytest.warns(FutureWarning, match=r"cluster_key.*stops working in squidpy v1\.9\.0"):
        old = interaction_matrix(nhood_data, "leiden", "spatial", False, True)
    new = interaction_matrix(nhood_data, cluster_key="leiden", connectivity_key="spatial", normalized=False, copy=True)
    np.testing.assert_array_equal(old, new)


def test_warnings_inside_shimmed_functions_point_at_the_caller(nhood_data: AnnData) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sq.gr.spatial_neighbors(nhood_data, coord_type="generic", n_neighs=5, radius=3.0)
        sq.gr.nhood_enrichment(
            nhood_data, "leiden", n_perms=5, min_cell_count=nhood_data.n_obs, show_progress_bar=False
        )
    # Each squidpy-emitted warning must exist and point at this caller; unrelated GC noise such as a
    # ResourceWarning collected mid-call is ignored (it does not originate from squidpy's stacklevel).
    for part in ("Calling `spatial_neighbors`", "`n_neighs` is ignored", "no longer positional", "were excluded"):
        matching = [w for w in caught if part in str(w.message)]
        assert matching, part
        assert {w.filename for w in matching} == {__file__}, [(w.filename, str(w.message)) for w in caught]
