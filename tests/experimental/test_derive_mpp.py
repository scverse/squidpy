from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import Affine, Identity, Scale, Sequence, Translation, set_transformation

from squidpy.experimental.utils import derive_mpp_from_shapes


def _hex_lattice(pitch: float, n_rows: int = 6, n_cols: int = 6) -> np.ndarray:
    dy = pitch * math.sin(math.pi / 3.0)
    rows = []
    for r in range(n_rows):
        offset = (pitch / 2.0) if (r % 2) else 0.0
        for c in range(n_cols):
            rows.append((c * pitch + offset, r * dy))
    return np.asarray(rows, dtype=float)


def _square_lattice(pitch: float, n: int = 6) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(n) * pitch, np.arange(n) * pitch)
    return np.column_stack([xs.ravel(), ys.ravel()]).astype(float)


def _points_shapes(centers: np.ndarray, radius: float) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {"radius": np.full(len(centers), radius, dtype=float)},
        geometry=[Point(x, y) for x, y in centers],
    )
    return ShapesModel.parse(gdf)


def _square_polygons_shapes(centers: np.ndarray, edge: float) -> gpd.GeoDataFrame:
    half = edge / 2.0
    polys = [
        Polygon([(x - half, y - half), (x + half, y - half), (x + half, y + half), (x - half, y + half)])
        for x, y in centers
    ]
    gdf = gpd.GeoDataFrame(geometry=polys)
    return ShapesModel.parse(gdf)


def _make_sdata(gdf: gpd.GeoDataFrame, transforms: dict | None = None) -> SpatialData:
    sdata = SpatialData(shapes={"shapes": gdf})
    if transforms is not None:
        for cs_name, transform in transforms.items():
            set_transformation(sdata.shapes["shapes"], transform, to_coordinate_system=cs_name)
    return sdata


def _rotation_affine(angle_deg: float) -> Affine:
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    m = np.array(
        [
            [cos, -sin, 0.0],
            [sin, cos, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return Affine(m, input_axes=("x", "y"), output_axes=("x", "y"))


def test_hex_pitch_identity():
    centers = _hex_lattice(pitch=100.0)
    sdata = _make_sdata(_points_shapes(centers, radius=27.5))
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=100.0)
    assert mpp == pytest.approx(1.0, rel=1e-9)


def test_hex_pitch_scaled():
    centers = _hex_lattice(pitch=100.0)
    sdata = _make_sdata(_points_shapes(centers, radius=27.5), transforms={"global": Scale([2.0, 2.0], axes=("x", "y"))})
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=100.0)
    assert mpp == pytest.approx(0.5, rel=1e-9)


def test_square_pitch():
    centers = _square_lattice(pitch=8.0)
    sdata = _make_sdata(_points_shapes(centers, radius=1.0))
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=8.0)
    assert mpp == pytest.approx(1.0, rel=1e-9)


def test_diameter_points():
    centers = _hex_lattice(pitch=100.0)
    sdata = _make_sdata(_points_shapes(centers, radius=27.5), transforms={"global": Scale([4.0, 4.0], axes=("x", "y"))})
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=55.0)
    # native diam = 55, scale=4 -> target diam = 220 px, mpp = 55/220 = 0.25
    assert mpp == pytest.approx(0.25, rel=1e-9)


def test_diameter_polygons():
    centers = _square_lattice(pitch=8.0)
    sdata = _make_sdata(_square_polygons_shapes(centers, edge=8.0))
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=8.0)
    # native edge = 8 px under identity; sqrt(area)=8 -> mpp = 1
    assert mpp == pytest.approx(1.0, rel=1e-9)


def test_coordinate_system_selection():
    centers = _square_lattice(pitch=8.0)
    sdata = _make_sdata(
        _points_shapes(centers, radius=1.0),
        transforms={"native": Identity(), "downscaled": Scale([0.5, 0.5], axes=("x", "y"))},
    )
    mpp_native = derive_mpp_from_shapes(sdata, "shapes", "native", um_between_centers=8.0)
    mpp_down = derive_mpp_from_shapes(sdata, "shapes", "downscaled", um_between_centers=8.0)
    assert mpp_native == pytest.approx(1.0, rel=1e-9)
    assert mpp_down == pytest.approx(2.0, rel=1e-9)
    assert mpp_down / mpp_native == pytest.approx(2.0, rel=1e-9)


def test_anisotropy_rejected():
    centers = _square_lattice(pitch=8.0)
    sdata = _make_sdata(_points_shapes(centers, radius=1.0), transforms={"global": Scale([2.0, 4.0], axes=("x", "y"))})
    with pytest.raises(ValueError, match=r"anisotropic"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=8.0)


def test_rotation_preserved():
    centers = _hex_lattice(pitch=100.0)
    sdata = _make_sdata(
        _points_shapes(centers, radius=27.5),
        transforms={"global": Sequence([Scale([2.0, 2.0], axes=("x", "y")), _rotation_affine(30.0)])},
    )
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=100.0)
    assert mpp == pytest.approx(0.5, rel=1e-6)


def test_sequence_translation_ignored():
    centers = _hex_lattice(pitch=100.0)
    sdata = _make_sdata(
        _points_shapes(centers, radius=27.5),
        transforms={
            "global": Sequence([Scale([2.0, 2.0], axes=("x", "y")), Translation([10.0, -5.0], axes=("x", "y"))]),
        },
    )
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=100.0)
    assert mpp == pytest.approx(0.5, rel=1e-9)


def test_three_d_rejected():
    gdf = gpd.GeoDataFrame(
        {"radius": [1.0, 1.0]},
        geometry=[Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)],
    )
    gdf = ShapesModel.parse(gdf)
    sdata = SpatialData(shapes={"shapes": gdf})
    with pytest.raises(ValueError, match=r"3D"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=1.0)


def test_multipolygon_rejected():
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    gdf = gpd.GeoDataFrame(geometry=[MultiPolygon([p1, p2])])
    gdf = ShapesModel.parse(gdf)
    sdata = SpatialData(shapes={"shapes": gdf})
    with pytest.raises(ValueError, match=r"MultiPolygon"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=1.0)


def test_single_shape_pitch_rejected():
    gdf = _points_shapes(np.array([[0.0, 0.0]]), radius=1.0)
    sdata = _make_sdata(gdf)
    with pytest.raises(ValueError, match=r"single shape"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=100.0)


def test_single_shape_diameter_works():
    gdf = _points_shapes(np.array([[0.0, 0.0]]), radius=27.5)
    sdata = _make_sdata(gdf)
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=55.0)
    assert mpp == pytest.approx(1.0, rel=1e-9)


def test_neither_arg_rejected():
    sdata = _make_sdata(_points_shapes(_square_lattice(pitch=8.0), radius=1.0))
    with pytest.raises(ValueError, match=r"exactly one"):
        derive_mpp_from_shapes(sdata, "shapes", "global")


def test_both_args_rejected():
    sdata = _make_sdata(_points_shapes(_square_lattice(pitch=8.0), radius=1.0))
    with pytest.raises(ValueError, match=r"exactly one"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=8.0, um_diameter=1.0)


def test_unknown_coordinate_system():
    sdata = _make_sdata(
        _points_shapes(_square_lattice(pitch=8.0), radius=1.0),
        transforms={"native": Identity(), "downscaled": Scale([0.5, 0.5], axes=("x", "y"))},
    )
    with pytest.raises(ValueError, match=r"native.*downscaled|downscaled.*native"):
        derive_mpp_from_shapes(sdata, "shapes", "missing", um_between_centers=8.0)
