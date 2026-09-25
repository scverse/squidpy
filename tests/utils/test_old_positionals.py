"""Arguments that became keyword-only still accept an old positional call, with a warning (#1288)."""

from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from squidpy.gr import interaction_matrix


def test_old_positional_call_warns_and_still_works(nhood_data: AnnData) -> None:
    # `cluster_key`, `connectivity_key`, `normalized` and `copy` were positional before #1288
    with pytest.warns(FutureWarning, match="cluster_key"):
        old = interaction_matrix(nhood_data, "leiden", "spatial", False, True)
    new = interaction_matrix(nhood_data, cluster_key="leiden", connectivity_key="spatial", normalized=False, copy=True)
    np.testing.assert_array_equal(old, new)
