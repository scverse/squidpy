from __future__ import annotations

import anndata as ad
import dask.dataframe as dd
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import spatialdata as sd
from shapely import Point, Polygon
from spatialdata.transformations import Identity

from squidpy.tl import smooth_density_by_distance
from squidpy.tl._smooth_density_by_distance import _shell_area


def _make_sdata(transcripts_df: pd.DataFrame, npartitions: int = 2) -> sd.SpatialData:
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
        {"radius": [1.0]},
        index=pd.Index(["cell_0"], name="cell_id"),
        geometry=[Point(1.0, 1.0)],
    )

    transcripts_ddf = dd.from_pandas(transcripts_df, npartitions=npartitions)

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
            "cells": sd.models.ShapesModel().parse(cells_raw, transformations={"global": Identity()}),
            "contours": sd.models.ShapesModel().parse(contour_df, transformations={"global": Identity()}),
            "transcripts": sd.models.PointsModel().parse(
                transcripts_ddf,
                feature_key="feature_name",
                instance_key="cell_id",
                transformations={"global": Identity()},
            ),
            "table": sd.models.TableModel().parse(adata),
        }
    )


@pytest.fixture()
def sdata_smooth_density() -> sd.SpatialData:
    transcripts_df = pd.DataFrame(
        {
            "x": [
                0.5,
                0.8,
                1.2,
                -0.5,
                -0.8,
                -1.2,
                10.5,
                10.8,
                3.5,
                4.0,
                13.5,
                14.0,
                0.5,
            ],
            "y": [5.0] * 13,
            "feature_name": ["A"] * 12 + ["B"],
            "cell_id": ["cell_0"] * 13,
        }
    )
    return _make_sdata(transcripts_df, npartitions=2)


@pytest.fixture()
def sdata_boundary_probe() -> sd.SpatialData:
    transcripts_df = pd.DataFrame(
        {
            "x": [3.9],
            "y": [5.0],
            "feature_name": ["A"],
            "cell_id": ["cell_0"],
        }
    )
    return _make_sdata(transcripts_df, npartitions=1)


class TestSmoothDensityByDistance:
    def test_smooth_density_output_structure(self, sdata_smooth_density: sd.SpatialData):
        result = smooth_density_by_distance(
            sdata_smooth_density,
            contour_key="contours",
            table_key="table",
            points_key="transcripts",
            feature_values="A",
            bandwidth=1.0,
            grid_step=0.5,
            inward=4.0,
            outward=4.0,
            copy=True,
        )

        expected_grid = np.arange(-4.0, 4.0 + 0.5, 0.5)
        assert np.allclose(result["signed_distance"].to_numpy(dtype=float), expected_grid)
        assert result["signed_distance"].is_monotonic_increasing
        assert len(result) == len(expected_grid)
        assert result["contour_id"].nunique() == 1
        assert result["contour_id"].iloc[0] == "contour_a"
        assert result["target"].nunique() == 1
        assert result["target"].iloc[0] == "transcripts"
        assert result["target_key"].nunique() == 1
        assert result["target_key"].iloc[0] == "transcripts"
        assert result["kernel"].nunique() == 1
        assert result["kernel"].iloc[0] == "gaussian"
        assert result["bandwidth"].nunique() == 1
        assert result["bandwidth"].iloc[0] == pytest.approx(1.0)
        assert result["grid_step"].nunique() == 1
        assert result["grid_step"].iloc[0] == pytest.approx(0.5)
        assert result["feature_values"].dropna().iloc[0] == ("A",)
        assert {
            "count_density",
            "geometry_measure",
            "density",
            "signed_distance",
            "bandwidth",
            "grid_step",
            "kernel",
        } <= set(result.columns)

    def test_smooth_density_save_to_uns(self, sdata_smooth_density: sd.SpatialData):
        smooth_density_by_distance(
            sdata_smooth_density,
            contour_key="contours",
            table_key="table",
            points_key="transcripts",
            feature_values="A",
            bandwidth=1.0,
            grid_step=0.5,
            inward=4.0,
            outward=4.0,
            copy=False,
        )

        assert "smooth_density_by_distance" in sdata_smooth_density.tables["table"].uns
        stored = sdata_smooth_density.tables["table"].uns["smooth_density_by_distance"]
        assert isinstance(stored, pd.DataFrame)
        assert {"signed_distance", "count_density", "geometry_measure", "density"} <= set(stored.columns)

    def test_smooth_density_is_boundary_enriched(self, sdata_smooth_density: sd.SpatialData):
        result = smooth_density_by_distance(
            sdata_smooth_density,
            contour_key="contours",
            table_key="table",
            points_key="transcripts",
            feature_values="A",
            bandwidth=1.0,
            grid_step=0.5,
            inward=4.0,
            outward=4.0,
            copy=True,
        )

        near_boundary = result.loc[result["signed_distance"] == 0.0, "count_density"].item()
        far_from_boundary = result.loc[result["signed_distance"] == 3.0, "count_density"].item()

        assert near_boundary > far_from_boundary
        assert result["density"].dropna().nunique() > 8

    def test_smooth_density_partition_invariance(self):
        transcripts_df = pd.DataFrame(
            {
                "x": [0.5, 0.8, 1.2, -0.5, -0.8, -1.2, 10.5, 10.8, 3.5, 4.0, 13.5, 14.0],
                "y": [5.0] * 12,
                "feature_name": ["A"] * 12,
                "cell_id": ["cell_0"] * 12,
            }
        )

        sdata_single = _make_sdata(transcripts_df, npartitions=1)
        sdata_multi = _make_sdata(transcripts_df, npartitions=3)

        result_single = smooth_density_by_distance(
            sdata_single,
            contour_key="contours",
            table_key="table",
            feature_values="A",
            bandwidth=1.0,
            grid_step=0.5,
            inward=4.0,
            outward=4.0,
            copy=True,
        )
        result_multi = smooth_density_by_distance(
            sdata_multi,
            contour_key="contours",
            table_key="table",
            feature_values="A",
            bandwidth=1.0,
            grid_step=0.5,
            inward=4.0,
            outward=4.0,
            copy=True,
        )

        result_single = result_single.sort_values("signed_distance", kind="stable").reset_index(drop=True)
        result_multi = result_multi.sort_values("signed_distance", kind="stable").reset_index(drop=True)

        assert np.allclose(
            result_single["count_density"].to_numpy(dtype=float),
            result_multi["count_density"].to_numpy(dtype=float),
        )
        assert np.allclose(
            result_single["geometry_measure"].to_numpy(dtype=float),
            result_multi["geometry_measure"].to_numpy(dtype=float),
        )
        assert np.allclose(
            result_single["density"].to_numpy(dtype=float),
            result_multi["density"].to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_smooth_density_geometry_measure_matches_local_shell_area(self, sdata_smooth_density: sd.SpatialData):
        result = smooth_density_by_distance(
            sdata_smooth_density,
            contour_key="contours",
            table_key="table",
            feature_values="A",
            bandwidth=1.0,
            grid_step=0.5,
            inward=6.0,
            outward=4.0,
            copy=True,
        )

        contour = sdata_smooth_density.shapes["contours"].geometry.iloc[0]
        row = result.loc[result["signed_distance"] == 1.0].iloc[0]
        expected_local_area = _shell_area(contour, 0.75, 1.25)
        assert row["geometry_measure"] * row["grid_step"] == pytest.approx(expected_local_area)

        deep_row = result.loc[result["signed_distance"] == -6.0].iloc[0]
        assert deep_row["geometry_measure"] == pytest.approx(0.0)
        assert np.isnan(deep_row["density"])

    def test_smooth_density_reflection_correction_at_range_edge(self, sdata_boundary_probe: sd.SpatialData):
        result = smooth_density_by_distance(
            sdata_boundary_probe,
            contour_key="contours",
            table_key="table",
            feature_values="A",
            bandwidth=1.0,
            grid_step=0.5,
            inward=4.0,
            outward=4.0,
            copy=True,
        )

        observed = result.loc[result["signed_distance"] == -4.0, "count_density"].item()
        expected = 2.0 * np.exp(-0.5 * (0.1**2)) / np.sqrt(2.0 * np.pi)
        assert observed == pytest.approx(expected, rel=1e-6)
