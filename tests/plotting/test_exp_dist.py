from __future__ import annotations

from typing import Sequence
from functools import partial
import pytest
import platform

from anndata import AnnData
import scanpy as sc

import numpy as np
import pandas as pd

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from squidpy import pl
from squidpy.gr import spatial_neighbors
from tests.conftest import PlotTester, PlotTesterMeta
from squidpy.pl._spatial_utils import _get_library_id
from squidpy._constants._pkg_constants import Key

sc.set_figure_params(dpi=40, color_map="viridis")

# WARNING:
# 1. all classes must both subclass PlotTester and use metaclass=PlotTesterMeta
# 2. tests which produce a plot must be prefixed with `test_plot_`
# 3. if the tolerance needs to be change, don't prefix the function with `test_plot_`, but with something else
#    the comp. function can be accessed as `self.compare(<your_filename>, tolerance=<your_tolerance>)`
#    ".png" is appended to <your_filename>, no need to set it


class TestSpatialStatic(PlotTester, metaclass=PlotTesterMeta):
    def test_single_cluster(self, adata_hne: AnnData):
        pass

    def test_multiple_clusters(self, adata_hne: AnnData):
        pass
