from __future__ import annotations

import sys

import pytest


@pytest.fixture(autouse=True)
def _xfail_internet_if_macos(request: pytest.FixtureRequest) -> None:
    if request.node.get_closest_marker("internet") and sys.platform == "darwin":
        request.applymarker(pytest.mark.xfail("Downloads fail on macOS", strict=False))
