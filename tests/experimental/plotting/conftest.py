from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent


def pytest_collection_modifyitems(items):
    """Mark everything in this directory as requiring spatialdata-plot.

    Every module here imports ``spatialdata_plot`` at module level, so applying the
    marker from the directory keeps it in one place and covers files added later.
    The hook is session-wide even though this conftest is not, hence the path filter.
    """
    for item in items:
        if item.path.is_relative_to(HERE):
            item.add_marker("spatialdata_plot")
