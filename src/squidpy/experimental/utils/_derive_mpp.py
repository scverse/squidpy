from __future__ import annotations

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


def derive_mpp_from_shapes(
    sdata: sd.SpatialData,
    shapes_element: str,
    coordinate_system: str,
    *,
    um_between_centers: float | None = None,
    um_diameter: float | None = None,
) -> float:
    """
    Derive microns-per-pixel for a coordinate system from a shapes element with a known physical scale.

    Given a shapes element (e.g. Visium spots, Visium HD bins) whose physical spacing or diameter is
    known, this function returns the microns-per-pixel of ``coordinate_system`` by measuring the
    corresponding geometric quantity in the target coordinate system and dividing the physical value
    by it.

    Exactly one of ``um_between_centers`` or ``um_diameter`` must be provided. Prefer
    ``um_between_centers`` when the technology's canonical pitch is known: it averages over the
    realised grid and is robust to per-spot calibration noise in the stored radius. ``um_diameter``
    depends on a single stored scalar and can disagree by a fraction of a percent on real Visium
    data where the radius and grid pitch are calibrated separately.

    The function requires the transformation between the shapes' native frame and ``coordinate_system``
    to be a similarity (uniform scale plus optional rotation and translation). Non-uniform scales,
    shear, or other anisotropy raise ``ValueError``: a single scalar microns-per-pixel is not
    well-defined in that case.

    Parameters
    ----------
    sdata
        SpatialData object containing the shapes element.
    shapes_element
        Key of the shapes element in ``sdata.shapes``.
    coordinate_system
        Name of the target coordinate system (pixel grid) to derive microns-per-pixel for. Must be
        one of the coordinate systems the shapes element is registered against.
    um_between_centers
        Known physical center-to-center distance of neighbouring shapes, in microns. For Visium v1
        this is 100 (hex grid); for Visium HD it equals the bin size in microns.
    um_diameter
        Known physical diameter of a shape, in microns. For Visium v1 this is 55. For square-bin
        Visium HD shapes this is interpreted as the edge length and equated with
        ``sqrt(median(area))`` of the transformed polygons.

    Returns
    -------
    float
        Microns per pixel of ``coordinate_system``.

    Raises
    ------
    ValueError
        If neither or both of ``um_between_centers`` / ``um_diameter`` are given; if
        ``coordinate_system`` is not registered for the element; if shapes are 3D or contain
        ``MultiPolygon`` geometries; if ``um_between_centers`` is given with only one shape; or
        if the transformation to ``coordinate_system`` is not a similarity.
    """
    if (um_between_centers is None) == (um_diameter is None):
        raise ValueError("Provide exactly one of `um_between_centers` or `um_diameter`.")

    assert_key_in_sdata(sdata, shapes_element, attr="shapes")
    gdf = sdata.shapes[shapes_element]

    axes = get_axes_names(gdf)
    if "z" in axes:
        raise ValueError(f"Shapes element '{shapes_element}' is 3D (axes={axes}); only 2D shapes are supported.")

    all_transforms = get_transformation(gdf, get_all=True)
    if coordinate_system not in all_transforms:
        raise ValueError(
            f"Coordinate system '{coordinate_system}' is not registered for shapes element "
            f"'{shapes_element}'. Available: {sorted(all_transforms)}."
        )

    geom_types = set(gdf.geometry.geom_type.unique())
    if "MultiPolygon" in geom_types:
        raise ValueError(
            f"Shapes element '{shapes_element}' contains MultiPolygon geometries; only Point and Polygon are supported."
        )

    affine = np.asarray(all_transforms[coordinate_system].to_affine_matrix(("x", "y"), ("x", "y")))
    A = affine[:2, :2]
    t = affine[:2, 2]

    sv = np.linalg.svd(A, compute_uv=False)
    s1, s2 = float(sv[0]), float(sv[1])
    if abs(s1 - s2) / max(s1, s2) > _ANISOTROPY_TOL:
        physical = um_between_centers if um_between_centers is not None else um_diameter
        raise ValueError(
            f"Transformation from shapes '{shapes_element}' to coordinate system "
            f"'{coordinate_system}' is anisotropic (singular values {s1:.6g}, {s2:.6g}). "
            f"A single scalar microns-per-pixel is not well-defined; per-axis values would be "
            f"{physical / s1:.6g} and {physical / s2:.6g}."
        )

    if um_between_centers is not None:
        return _mpp_from_pitch(gdf, A, t, um_between_centers)
    assert um_diameter is not None  # guaranteed by the XOR check above
    return _mpp_from_diameter(gdf, A, um_diameter)


def _mpp_from_pitch(gdf: gpd.GeoDataFrame, A: np.ndarray, t: np.ndarray, um_between_centers: float) -> float:
    n = len(gdf)
    if n < 2:
        raise ValueError("Pitch is undefined for a single shape; pass `um_diameter` instead.")
    centroids = gdf.geometry.centroid
    xy_native = np.column_stack([centroids.x.to_numpy(), centroids.y.to_numpy()])
    if n > _PITCH_MAX_SAMPLES:
        rng = np.random.default_rng(0)
        xy_native = xy_native[rng.choice(n, size=_PITCH_MAX_SAMPLES, replace=False)]
    xy_target = xy_native @ A.T + t
    nn_dist = cKDTree(xy_target).query(xy_target, k=2)[0][:, 1]
    return um_between_centers / float(np.median(nn_dist))


def _mpp_from_diameter(gdf: gpd.GeoDataFrame, A: np.ndarray, um_diameter: float) -> float:
    geom_types = set(gdf.geometry.geom_type.unique())
    if geom_types == {"Point"}:
        if "radius" not in gdf.columns:
            raise ValueError("Point shapes element is missing the 'radius' column required for diameter-based mpp.")
        scale = float(np.sqrt(abs(np.linalg.det(A))))
        diam_target = float(np.median(2.0 * gdf["radius"].to_numpy())) * scale
    elif geom_types <= {"Polygon"}:
        # area transforms by |det A| under any affine, so we avoid per-geometry shapely calls
        det = float(abs(np.linalg.det(A)))
        diam_target = float(np.sqrt(np.median(gdf.geometry.area.to_numpy()) * det))
    else:
        raise ValueError(f"Unsupported geometry types {sorted(geom_types)}; expected Point or Polygon.")
    return um_diameter / diam_target
