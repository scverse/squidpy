from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import spatialdata as sd
from scipy.spatial import cKDTree
from spatialdata.models import get_axes_names
from spatialdata.transformations import get_transformation

from squidpy._validators import assert_key_in_sdata

__all__ = ["derive_mpp_from_shapes"]

_ANISOTROPY_TOL = 1e-3
_PITCH_MAX_SAMPLES = 5000
_SQUARENESS_SAMPLE = 10
_SQUARENESS_TOL = 0.98


def derive_mpp_from_shapes(
    sdata: sd.SpatialData,
    shapes_key: str,
    coordinate_system: str,
    *,
    um_between_centers: float | None = None,
    um_diameter: float | None = None,
    um_square_edge: float | None = None,
) -> float:
    """
    Derive microns-per-pixel for a coordinate system from a shapes element with a known physical scale.

    Given a shapes element (e.g. Visium spots, Visium HD bins) whose physical spacing, spot diameter,
    or bin edge length is known, this function returns the microns-per-pixel of
    ``coordinate_system`` by measuring the corresponding geometric quantity in the target coordinate
    system and dividing the physical value by it.

    Exactly one of ``um_between_centers``, ``um_diameter``, or ``um_square_edge`` must be provided.
    Prefer ``um_between_centers`` when the technology's canonical pitch is known: it averages over
    thousands of centroid pairs in the realised grid and is robust to per-spot calibration noise in
    the stored radius/area. ``um_diameter`` and ``um_square_edge`` depend on a single stored scalar
    per shape and can disagree by a fraction of a percent on real Visium data where the radius and
    grid pitch are calibrated separately.

    Each physical input is geometry-specific:

    - ``um_diameter`` applies only to Point geometries (uses the ``radius`` column).
    - ``um_square_edge`` applies only to Polygon geometries and assumes square or rectangular
      polygons; a sample of polygons is checked for squareness and the call raises if any are not.
    - ``um_between_centers`` works for any 2D geometry.

    The function requires the transformation between the shapes' native frame and ``coordinate_system``
    to be a similarity (uniform scale plus optional rotation and translation). Non-uniform scales,
    shear, or other anisotropy raise ``ValueError``: a single scalar microns-per-pixel is not
    well-defined in that case.

    Parameters
    ----------
    sdata
        SpatialData object containing the shapes element.
    shapes_key
        Key of the shapes element in ``sdata.shapes``.
    coordinate_system
        Name of the target coordinate system (pixel grid) to derive microns-per-pixel for. Must be
        one of the coordinate systems the shapes element is registered against.
    um_between_centers
        Known physical center-to-center distance of neighbouring shapes, in microns. For Visium v1
        this is 100 (hex grid); for Visium HD it equals the bin size in microns. Works for any 2D
        geometry.
    um_diameter
        Known physical diameter of a circular spot, in microns. Point geometries only (uses the
        ``radius`` column). For Visium v1 this is 55.
    um_square_edge
        Known physical edge length of a square/rectangular bin, in microns. Polygon geometries only.
        For Visium HD this equals the bin size in microns. Non-rectangular polygons (e.g. hex bins,
        circular approximations) are rejected.

    Returns
    -------
    float
        Microns per pixel of ``coordinate_system``.

    Raises
    ------
    ValueError
        If not exactly one of the three physical inputs is given; if ``coordinate_system`` is not
        registered for the element; if shapes are 3D or contain ``MultiPolygon`` geometries; if
        ``um_diameter`` is paired with Polygons or ``um_square_edge`` with Points; if
        ``um_square_edge`` is paired with non-rectangular polygons; if ``um_between_centers`` is
        given with only one shape; or if the transformation to ``coordinate_system`` is not a
        similarity.
    """
    n_given = sum(x is not None for x in (um_between_centers, um_diameter, um_square_edge))
    if n_given != 1:
        raise ValueError("Provide exactly one of `um_between_centers`, `um_diameter`, or `um_square_edge`.")

    assert_key_in_sdata(sdata, shapes_key, attr="shapes")
    gdf = sdata.shapes[shapes_key]

    if len(gdf) == 0:
        raise ValueError(f"Shapes element '{shapes_key}' is empty; cannot derive mpp.")

    axes = get_axes_names(gdf)
    if "z" in axes:
        raise ValueError(f"Shapes element '{shapes_key}' is 3D (axes={axes}); only 2D shapes are supported.")

    all_transforms = get_transformation(gdf, get_all=True)
    if coordinate_system not in all_transforms:
        raise ValueError(
            f"Coordinate system '{coordinate_system}' is not registered for shapes element "
            f"'{shapes_key}'. Available: {sorted(all_transforms)}."
        )

    geom_types = set(gdf.geometry.geom_type.unique())
    if "MultiPolygon" in geom_types:
        raise ValueError(
            f"Shapes element '{shapes_key}' contains MultiPolygon geometries; only Point and Polygon are supported."
        )

    affine = np.asarray(all_transforms[coordinate_system].to_affine_matrix(("x", "y"), ("x", "y")))
    A = affine[:2, :2]
    t = affine[:2, 2]

    sv = np.linalg.svd(A, compute_uv=False)
    s1, s2 = float(sv[0]), float(sv[1])
    if abs(s1 - s2) / max(s1, s2) > _ANISOTROPY_TOL:
        physical = next(x for x in (um_between_centers, um_diameter, um_square_edge) if x is not None)
        raise ValueError(
            f"Transformation from shapes '{shapes_key}' to coordinate system "
            f"'{coordinate_system}' is anisotropic (singular values {s1:.6g}, {s2:.6g}). "
            f"A single scalar microns-per-pixel is not well-defined; per-axis values would be "
            f"{physical / s1:.6g} and {physical / s2:.6g}."
        )

    if um_between_centers is not None:
        return _mpp_from_pitch(gdf, A, t, um_between_centers)
    if um_diameter is not None:
        if geom_types != {"Point"}:
            raise ValueError(
                f"`um_diameter` requires Point geometries; got {sorted(geom_types)}. "
                "For square/rectangular polygons use `um_square_edge`."
            )
        return _mpp_from_diameter(gdf, A, um_diameter)
    assert um_square_edge is not None
    if geom_types != {"Polygon"}:
        raise ValueError(
            f"`um_square_edge` requires Polygon geometries; got {sorted(geom_types)}. "
            "For circular Point spots use `um_diameter`."
        )
    return _mpp_from_square_edge(gdf, A, um_square_edge)


def _mpp_from_pitch(gdf: gpd.GeoDataFrame, A: np.ndarray, t: np.ndarray, um_between_centers: float) -> float:
    n = len(gdf)
    if n < 2:
        raise ValueError("Pitch is undefined for a single shape; pass `um_diameter` or `um_square_edge` instead.")
    centroids = gdf.geometry.centroid
    xy_native = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    xy_target = xy_native @ A.T + t
    query_xy = xy_target
    if n > _PITCH_MAX_SAMPLES:
        rng = np.random.default_rng(0)
        query_xy = xy_target[rng.choice(n, size=_PITCH_MAX_SAMPLES, replace=False)]
    nn_dist = cKDTree(xy_target).query(query_xy, k=2)[0][:, 1]
    return um_between_centers / float(np.median(nn_dist))


def _mpp_from_diameter(gdf: gpd.GeoDataFrame, A: np.ndarray, um_diameter: float) -> float:
    if "radius" not in gdf.columns:
        raise ValueError("Point shapes element is missing the 'radius' column required for diameter-based mpp.")
    scale = float(np.sqrt(abs(np.linalg.det(A))))
    diam_target = float(np.median(2.0 * gdf["radius"].to_numpy())) * scale
    return um_diameter / diam_target


def _mpp_from_square_edge(gdf: gpd.GeoDataFrame, A: np.ndarray, um_square_edge: float) -> float:
    _assert_polygons_are_square(gdf)
    det = float(abs(np.linalg.det(A)))
    edge_target = float(np.sqrt(np.median(gdf.geometry.area.to_numpy()) * det))
    return um_square_edge / edge_target


def _assert_polygons_are_square(gdf: gpd.GeoDataFrame) -> None:
    n = len(gdf)
    sample_size = min(_SQUARENESS_SAMPLE, n)
    rng = np.random.default_rng(0)
    sample = gdf.geometry.iloc[rng.choice(n, size=sample_size, replace=False)]
    # shapely's oriented_envelope emits benign divide-by-zero RuntimeWarnings on axis-aligned
    # rectangles; result is correct, so we silence them here only.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"shapely\..*")
        mrr_area = sample.minimum_rotated_rectangle().area.to_numpy()
    ratios = sample.area.to_numpy() / mrr_area
    if np.any(ratios < _SQUARENESS_TOL):
        raise ValueError(
            f"`um_square_edge` requires square/rectangular polygons; sampled {sample_size} polygons "
            f"and found area/minimum-rotated-rectangle ratio below {_SQUARENESS_TOL} "
            f"(min={float(ratios.min()):.4f}). For non-rectangular geometries use `um_between_centers`."
        )
