"""The plotting module."""

from squidpy.pl._graph import (
    centrality_scores,
    co_occurrence,
    interaction_matrix,
    nhood_enrichment,
    ripley,
)
from squidpy.pl._interactive import Interactive  # type: ignore[attr-defined]
from squidpy.pl._ligrec import ligrec
from squidpy.pl._spatial import spatial_scatter, spatial_segment
from squidpy.pl._utils import extract
from squidpy.pl._var_by_distance import var_by_distance
