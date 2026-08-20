from __future__ import annotations

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from shapely import Point, Polygon
from spatialdata.transformations import Identity, Scale

from squidpy.tl import ring_density


@pytest.fixture()
def sdata_ring_density():
    contour_df = gpd.GeoDataFrame(
        {
            "classification_name": ["region_a"],
            "assigned_structure": ["region_a"],
            "annotation_source": ["synthetic"],
        },
        index=pd.Index(["contour_a"], name="region_id"),
        geometry=[Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])],
    )

    cells_raw = gpd.GeoDataFrame(
        {"radius": [1.0] * 4},
        index=pd.Index(["cell_0", "cell_1", "cell_2", "cell_3"], name="cell_id"),
        geometry=[
            Point(2.5, 2.5),  # global (5, 5), far inside -> excluded for inward=4
            Point(5.5, 2.5),  # global (11, 5), first outward ring
            Point(6.5, 2.5),  # global (13, 5), second outward ring
            Point(2.5, 4.75),  # global (5, 9.5), first inward ring
        ],
    )

    transcripts_df = pd.DataFrame(
        {
            "x": [2.5, 5.5, 6.5, 2.5],
            "y": [4.75, 2.5, 2.5, 2.5],
            "feature_name": ["B", "A", "A", "A"],
            "cell_id": ["cell_3", "cell_1", "cell_2", "cell_0"],
        }
    )
    transcripts_ddf = dd.from_pandas(transcripts_df, npartitions=2)

    adata = ad.AnnData(np.zeros((len(cells_raw), 1)))
    adata.obs_names = cells_raw.index.astype(str)
    adata.obs["cell_id"] = adata.obs_names.astype(str)
    adata.obs["region"] = pd.Categorical(["cells"] * len(cells_raw))
    adata.uns["spatialdata_attrs"] = {
        "region": "cells",
        "region_key": "region",
        "instance_key": "cell_id",
    }

    return sd.SpatialData.init_from_elements(
        {
            "cells": sd.models.ShapesModel().parse(
                cells_raw, transformations={"global": Scale([2.0, 2.0], axes=("x", "y"))}
            ),
            "contours": sd.models.ShapesModel().parse(contour_df, transformations={"global": Identity()}),
            "transcripts": sd.models.PointsModel().parse(
                transcripts_ddf,
                feature_key="feature_name",
                instance_key="cell_id",
                transformations={"global": Scale([2.0, 2.0], axes=("x", "y"))},
            ),
            "table": sd.models.TableModel().parse(adata),
        }
    )


class TestRingDensity:
    def test_ring_density_cells(self, sdata_ring_density: sd.SpatialData):
        result = ring_density(
            sdata_ring_density,
            contour_key="contours",
            target="cells",
            table_key="table",
            ring_width=2.0,
            inward=4.0,
            outward=4.0,
            copy=True,
        )

        inward_ring = result.loc[(result["ring_start"] == -2.0) & (result["ring_end"] == 0.0), "count"].item()
        first_outward = result.loc[(result["ring_start"] == 0.0) & (result["ring_end"] == 2.0), "count"].item()
        second_outward = result.loc[(result["ring_start"] == 2.0) & (result["ring_end"] == 4.0), "count"].item()

        assert inward_ring == 1
        assert first_outward == 1
        assert second_outward == 1

        contour = sdata_ring_density.shapes["contours"].geometry.iloc[0]
        expected_area = contour.area - contour.buffer(-2.0).area
        observed_area = result.loc[(result["ring_start"] == -2.0) & (result["ring_end"] == 0.0), "area"].item()
        assert observed_area == pytest.approx(expected_area)
        assert result["contour_id"].nunique() == 1
        assert result["contour_id"].iloc[0] == "contour_a"
        assert result["target_key"].nunique() == 1
        assert result["target_key"].iloc[0] == "cells"

    def test_ring_density_transcripts_feature_filter(self, sdata_ring_density: sd.SpatialData):
        result = ring_density(
            sdata_ring_density,
            contour_key="contours",
            target="transcripts",
            table_key="table",
            points_key="transcripts",
            ring_width=2.0,
            inward=4.0,
            outward=4.0,
            feature_values="A",
            copy=True,
        )

        inward_ring = result.loc[(result["ring_start"] == -2.0) & (result["ring_end"] == 0.0), "count"].item()
        first_outward = result.loc[(result["ring_start"] == 0.0) & (result["ring_end"] == 2.0), "count"].item()
        second_outward = result.loc[(result["ring_start"] == 2.0) & (result["ring_end"] == 4.0), "count"].item()

        assert inward_ring == 0
        assert first_outward == 1
        assert second_outward == 1
        assert result["feature_values"].dropna().iloc[0] == ("A",)
        assert result["target_key"].nunique() == 1
        assert result["target_key"].iloc[0] == "transcripts"

    def test_ring_density_save_to_uns(self, sdata_ring_density: sd.SpatialData):
        ring_density(
            sdata_ring_density,
            contour_key="contours",
            target="cells",
            table_key="table",
            ring_width=2.0,
            inward=2.0,
            outward=2.0,
            copy=False,
        )

        assert "ring_density" in sdata_ring_density.tables["table"].uns
        stored = sdata_ring_density.tables["table"].uns["ring_density"]
        assert isinstance(stored, pd.DataFrame)
        assert {"ring_start", "ring_end", "count", "area", "density"} <= set(stored.columns)

    def test_ring_density_accepts_single_region_sequence(self, sdata_ring_density: sd.SpatialData):
        sdata_ring_density.tables["table"].uns["spatialdata_attrs"]["region"] = ["cells"]

        result = ring_density(
            sdata_ring_density,
            contour_key="contours",
            target="cells",
            table_key="table",
            ring_width=2.0,
            inward=2.0,
            outward=2.0,
            copy=True,
        )

        assert result["target_key"].nunique() == 1
        assert result["target_key"].iloc[0] == "cells"
