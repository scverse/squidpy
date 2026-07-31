from __future__ import annotations

import math

import geopandas as gpd
import numpy as np
import pytest
from shapely import MultiPolygon, Point, Polygon
from spatialdata import SpatialData
from spatialdata.models import ShapesModel
from spatialdata.transformations import Affine, Identity, Scale, Sequence, set_transformation

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


def _hex_polygons_shapes(centers: np.ndarray, edge: float) -> gpd.GeoDataFrame:
    angles = np.linspace(0.0, 2 * math.pi, 7)[:-1]
    offsets = np.column_stack([edge * np.cos(angles), edge * np.sin(angles)])
    polys = [Polygon([(x + ox, y + oy) for ox, oy in offsets]) for x, y in centers]
    gdf = gpd.GeoDataFrame(geometry=polys)
    return ShapesModel.parse(gdf)


def _make_sdata(gdf: gpd.GeoDataFrame, transforms: dict | None = None) -> SpatialData:
    sdata = SpatialData(shapes={"shapes": gdf})
    if transforms is not None:
        for cs_name, transform in transforms.items():
            set_transformation(sdata.shapes["shapes"], transform, to_coordinate_system=cs_name)
    return sdata


@pytest.mark.parametrize(
    ("lattice", "pitch", "transform", "expected_mpp"),
    [
        (_hex_lattice, 100.0, None, 1.0),
        (_hex_lattice, 100.0, Scale([2.0, 2.0], axes=("x", "y")), 0.5),
        (_square_lattice, 8.0, None, 1.0),
    ],
)
def test_pitch(lattice, pitch, transform, expected_mpp):
    centers = lattice(pitch=pitch)
    transforms = {"global": transform} if transform is not None else None
    sdata = _make_sdata(_points_shapes(centers, radius=pitch / 4.0), transforms=transforms)
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=pitch)
    assert mpp == pytest.approx(expected_mpp, rel=1e-9)


def test_pitch_large_grid_subsampling():
    # 120x120 = 14_400 points: well above _PITCH_MAX_SAMPLES (5_000). The buggy
    # tree-on-subsample implementation returns mpp ~0.35 here instead of 1.0.
    pitch = 8.0
    n = 120
    centers = _square_lattice(pitch=pitch, n=n)
    sdata = _make_sdata(_points_shapes(centers, radius=pitch / 4.0))
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=pitch)
    assert mpp == pytest.approx(1.0, rel=1e-9)


def test_diameter_points():
    centers = _hex_lattice(pitch=100.0)
    sdata = _make_sdata(_points_shapes(centers, radius=27.5), transforms={"global": Scale([4.0, 4.0], axes=("x", "y"))})
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=55.0)
    assert mpp == pytest.approx(0.25, rel=1e-9)


def test_square_edge_polygons():
    centers = _square_lattice(pitch=8.0)
    sdata = _make_sdata(_square_polygons_shapes(centers, edge=8.0))
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_square_edge=8.0)
    assert mpp == pytest.approx(1.0, rel=1e-9)


def test_um_diameter_on_polygons_rejected():
    sdata = _make_sdata(_square_polygons_shapes(_square_lattice(pitch=8.0), edge=8.0))
    with pytest.raises(ValueError, match=r"um_diameter.*Point geometries"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=8.0)


def test_um_square_edge_on_points_rejected():
    sdata = _make_sdata(_points_shapes(_square_lattice(pitch=8.0), radius=1.0))
    with pytest.raises(ValueError, match=r"um_square_edge.*Polygon geometries"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_square_edge=8.0)


def test_um_square_edge_on_non_square_polygons_rejected():
    centers = _square_lattice(pitch=10.0)
    sdata = _make_sdata(_hex_polygons_shapes(centers, edge=3.0))
    with pytest.raises(ValueError, match=r"square/rectangular polygons"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_square_edge=3.0)


def test_coordinate_system_selection():
    sdata = _make_sdata(
        _points_shapes(_square_lattice(pitch=8.0), radius=1.0),
        transforms={"native": Identity(), "downscaled": Scale([0.5, 0.5], axes=("x", "y"))},
    )
    mpp_native = derive_mpp_from_shapes(sdata, "shapes", "native", um_between_centers=8.0)
    mpp_down = derive_mpp_from_shapes(sdata, "shapes", "downscaled", um_between_centers=8.0)
    assert mpp_down / mpp_native == pytest.approx(2.0, rel=1e-9)


def test_anisotropy_rejected():
    sdata = _make_sdata(
        _points_shapes(_square_lattice(pitch=8.0), radius=1.0),
        transforms={"global": Scale([2.0, 4.0], axes=("x", "y"))},
    )
    with pytest.raises(ValueError, match=r"anisotropic"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=8.0)


def test_rotation_preserved():
    angle = math.radians(30.0)
    cos, sin = math.cos(angle), math.sin(angle)
    rotation = Affine(
        np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    sdata = _make_sdata(
        _points_shapes(_hex_lattice(pitch=100.0), radius=27.5),
        transforms={"global": Sequence([Scale([2.0, 2.0], axes=("x", "y")), rotation])},
    )
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=100.0)
    assert mpp == pytest.approx(0.5, rel=1e-6)


def test_three_d_rejected():
    gdf = ShapesModel.parse(
        gpd.GeoDataFrame({"radius": [1.0, 1.0]}, geometry=[Point(0.0, 0.0, 0.0), Point(1.0, 0.0, 0.0)])
    )
    sdata = SpatialData(shapes={"shapes": gdf})
    with pytest.raises(ValueError, match=r"3D"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=1.0)


def test_multipolygon_rejected():
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(2, 2), (3, 2), (3, 3), (2, 3)])
    gdf = ShapesModel.parse(gpd.GeoDataFrame(geometry=[MultiPolygon([p1, p2])]))
    sdata = SpatialData(shapes={"shapes": gdf})
    with pytest.raises(ValueError, match=r"MultiPolygon"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=1.0)


def test_single_shape_pitch_rejected():
    sdata = _make_sdata(_points_shapes(np.array([[0.0, 0.0]]), radius=1.0))
    with pytest.raises(ValueError, match=r"single shape"):
        derive_mpp_from_shapes(sdata, "shapes", "global", um_between_centers=100.0)


def test_single_shape_diameter_works():
    sdata = _make_sdata(_points_shapes(np.array([[0.0, 0.0]]), radius=27.5))
    mpp = derive_mpp_from_shapes(sdata, "shapes", "global", um_diameter=55.0)
    assert mpp == pytest.approx(1.0, rel=1e-9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"um_between_centers": 8.0, "um_diameter": 1.0},
        {"um_between_centers": 8.0, "um_square_edge": 8.0},
        {"um_diameter": 1.0, "um_square_edge": 8.0},
        {"um_between_centers": 8.0, "um_diameter": 1.0, "um_square_edge": 8.0},
    ],
)
def test_mutex_args_rejected(kwargs):
    sdata = _make_sdata(_points_shapes(_square_lattice(pitch=8.0), radius=1.0))
    with pytest.raises(ValueError, match=r"exactly one"):
        derive_mpp_from_shapes(sdata, "shapes", "global", **kwargs)


def test_unknown_coordinate_system():
    sdata = _make_sdata(
        _points_shapes(_square_lattice(pitch=8.0), radius=1.0),
        transforms={"native": Identity(), "downscaled": Scale([0.5, 0.5], axes=("x", "y"))},
    )
    with pytest.raises(ValueError, match=r"native.*downscaled|downscaled.*native"):
        derive_mpp_from_shapes(sdata, "shapes", "missing", um_between_centers=8.0)
