from __future__ import annotations

import itertools
import spatialdata as sd
import spatialdata_plot

_ = spatialdata_plot

from squidpy._docs import d


@d.dedent
def _spatial_counts_distribution(
    sdata: sd.SpatialData,
    coordinate_system: list[str] | str | None = None,
    sdata_plot_kwargs: dict = {},
) -> None:
    """
    Plot the distribution of spatial counts.

    EXPERIMENTAL: Only supports the new SpatialData interface.
    """

    if coordinate_system is not None:
        if isinstance(coordinate_system, str):
            coordinate_system = [coordinate_system]
        if not all(isinstance(cs, str) for cs in coordinate_system):
            raise ValueError("All elements in `coordinate_system` must be strings.")
        for cs in coordinate_system:
            pass
