from __future__ import annotations

import numpy as np
import shapely.affinity
import spatialdata as sd
from scipy.spatial import cKDTree
from spatialdata.models import ShapesModel
from spatialdata.models._utils import get_axes_names
from spatialdata.transformations import get_transformation

from squidpy._validators import assert_key_in_sdata

__all__ = ["derive_mpp_from_shapes"]

_ANISOTROPY_TOL = 1e-3


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

    Exactly one of ``um_between_centers`` or ``um_diameter`` must be provided.

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
        Visium HD shapes this is interpreted as the edge length and compared against
        ``sqrt(area)`` of the transformed polygons.

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
    ShapesModel().validate(gdf)

    axes = get_axes_names(gdf)
    if "z" in axes:
        raise ValueError(f"Shapes element '{shapes_element}' is 3D (axes={axes}); only 2D shapes are supported.")

    geom_types = set(gdf.geometry.geom_type.unique())
    if "MultiPolygon" in geom_types:
        raise ValueError(
            f"Shapes element '{shapes_element}' contains MultiPolygon geometries; only Point and Polygon are supported."
        )

    all_transforms = get_transformation(gdf, get_all=True)
    if coordinate_system not in all_transforms:
        raise ValueError(
            f"Coordinate system '{coordinate_system}' is not registered for shapes element "
            f"'{shapes_element}'. Available: {sorted(all_transforms)}."
        )
    transform = all_transforms[coordinate_system]

    affine = np.asarray(transform.to_affine_matrix(("x", "y"), ("x", "y")))
    A = affine[:2, :2]
    t = affine[:2, 2]

    sv = np.linalg.svd(A, compute_uv=False)
    s1, s2 = float(sv[0]), float(sv[1])
    if abs(s1 - s2) / max(s1, s2) > _ANISOTROPY_TOL:
        if um_between_centers is not None:
            mpp1, mpp2 = um_between_centers / s1, um_between_centers / s2
        else:
            mpp1, mpp2 = um_diameter / s1, um_diameter / s2  # type: ignore[operator]
        raise ValueError(
            f"Transformation from shapes '{shapes_element}' to coordinate system "
            f"'{coordinate_system}' is anisotropic (singular values {s1:.6g}, {s2:.6g}). "
            f"A single scalar microns-per-pixel is not well-defined; per-axis values would be "
            f"{mpp1:.6g} and {mpp2:.6g}."
        )
    scale = float(np.sqrt(abs(np.linalg.det(A))))

    if um_between_centers is not None:
        return _mpp_from_pitch(gdf, A, t, um_between_centers)
    return _mpp_from_diameter(gdf, geom_types, scale, A, t, um_diameter)  # type: ignore[arg-type]


def _mpp_from_pitch(gdf, A, t, um_between_centers: float) -> float:
    if len(gdf) < 2:
        raise ValueError("Pitch is undefined for a single shape; pass `um_diameter` instead.")
    xy_native = np.column_stack([gdf.geometry.centroid.x.to_numpy(), gdf.geometry.centroid.y.to_numpy()])
    xy_target = xy_native @ A.T + t
    tree = cKDTree(xy_target)
    nn_dist = tree.query(xy_target, k=2)[0][:, 1]
    pitch_px = float(np.median(nn_dist))
    return um_between_centers / pitch_px


def _mpp_from_diameter(
    gdf,
    geom_types: set[str],
    scale: float,
    A: np.ndarray,
    t: np.ndarray,
    um_diameter: float,
) -> float:
    if geom_types == {"Point"}:
        if "radius" not in gdf.columns:
            raise ValueError("Point shapes element is missing the 'radius' column required for diameter-based mpp.")
        diam_native = float(np.median(2.0 * gdf["radius"].to_numpy()))
        diam_target = diam_native * scale
    elif geom_types <= {"Polygon"}:
        shapely_affine = [A[0, 0], A[0, 1], A[1, 0], A[1, 1], t[0], t[1]]
        transformed = gdf.geometry.apply(lambda g: shapely.affinity.affine_transform(g, shapely_affine))
        diam_target = float(np.sqrt(np.median(transformed.area.to_numpy())))
    else:
        raise ValueError(f"Unsupported geometry types {sorted(geom_types)}; expected Point or Polygon.")
    return um_diameter / diam_target
