"""Skeleton-level tests for ``squidpy.experimental.tl.align_*``.

These tests verify *wiring* — argument resolution, dispatch, validation, and
lazy-import hygiene.  Real solver tests come with the next PR that drops the
actual implementations into the prepared ``NotImplementedError`` slots.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, TableModel

__all__: list[str] = []  # silence the import-only test module check


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_adata(n: int = 8, seed: int = 0) -> AnnData:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3)).astype(np.float32)
    adata = AnnData(X=X)
    adata.obs["region"] = "r"
    adata.obs["instance_id"] = np.arange(n)
    adata.obsm["spatial"] = rng.uniform(0, 100, (n, 2))
    return adata


def _make_table(name: str, n: int = 8, seed: int = 0) -> AnnData:
    adata = _make_adata(n=n, seed=seed)
    adata.obs["region"] = pd.Categorical([name] * n)
    adata.uns["spatialdata_attrs"] = {
        "region": name,
        "region_key": "region",
        "instance_key": "instance_id",
    }
    return TableModel.parse(adata)


def _make_sdata(image_keys=("img_ref", "img_query"), table_keys=("tbl_ref", "tbl_query")) -> SpatialData:
    images = {}
    for k in image_keys:
        arr = np.zeros((3, 32, 32), dtype=np.uint8)
        xa = xr.DataArray(arr, dims=["c", "y", "x"], coords={"c": ["R", "G", "B"]})
        images[k] = Image2DModel.parse(xa)
    tables = {k: _make_table(k) for k in table_keys}
    return SpatialData(images=images, tables=tables)


def _make_sdata_two_cs() -> SpatialData:
    """Single SpatialData with two distinct coordinate systems (``cs_a``/``cs_b``).

    Used by the landmark tests where we want to align one cs to another
    inside the same container.
    """
    from spatialdata.transformations import Identity, set_transformation

    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    xa = xr.DataArray(arr, dims=["c", "y", "x"], coords={"c": ["R", "G", "B"]})
    img_a = Image2DModel.parse(xa)
    img_b = Image2DModel.parse(xa.copy())
    set_transformation(img_a, Identity(), to_coordinate_system="cs_a")
    set_transformation(img_b, Identity(), to_coordinate_system="cs_b")
    return SpatialData(images={"img_a": img_a, "img_b": img_b})


@pytest.fixture
def adata_pair() -> tuple[AnnData, AnnData]:
    return _make_adata(seed=1), _make_adata(seed=2)


@pytest.fixture
def sdata_pair() -> tuple[SpatialData, SpatialData]:
    return _make_sdata(), _make_sdata()


@pytest.fixture
def sdata_single() -> SpatialData:
    return _make_sdata()


@pytest.fixture
def sdata_two_cs() -> SpatialData:
    return _make_sdata_two_cs()


# ---------------------------------------------------------------------------
# 1. Public path
# ---------------------------------------------------------------------------


def test_public_callables_exist() -> None:
    import squidpy as sq

    assert callable(sq.experimental.tl.align_obs)
    assert callable(sq.experimental.tl.align_images)
    assert callable(sq.experimental.tl.align_by_landmarks)


# ---------------------------------------------------------------------------
# 2. Lazy-import hygiene
# ---------------------------------------------------------------------------


def test_optional_deps_not_imported_at_import_time() -> None:
    """Subprocess-isolated import to defeat module caching from other tests.

    A pop+reimport in-process is unreliable here because other tests in the
    same session may have already pulled stalign/moscot/jax in transitively.
    Spawn a clean Python and inspect its sys.modules.
    """
    import subprocess
    import sys as _sys

    out = subprocess.check_output(
        [
            _sys.executable,
            "-c",
            (
                "import sys, squidpy; "
                "leaked = [m for m in ('jax', 'stalign', 'moscot') if m in sys.modules]; "
                "print(','.join(leaked))"
            ),
        ],
        text=True,
        timeout=30,
    ).strip()
    assert out == "", f"Optional deps imported by `import squidpy`: {out}"


# ---------------------------------------------------------------------------
# 3. Resolver matrix for align_obs
# ---------------------------------------------------------------------------


def test_resolve_obs_pair_two_anndata(adata_pair) -> None:
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    a, b = adata_pair
    pair = resolve_obs_pair(a, b, None, None)
    assert pair.ref is a
    assert pair.query is b
    assert pair.ref_container is None
    assert pair.query_container is None


def test_resolve_obs_pair_two_anndata_rejects_unneeded_name(adata_pair) -> None:
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    a, b = adata_pair
    with pytest.raises(ValueError, match="adata_ref_name"):
        resolve_obs_pair(a, b, "tbl_ref", None)
    with pytest.raises(ValueError, match="adata_query_name"):
        resolve_obs_pair(a, b, None, "tbl_query")


def test_resolve_obs_pair_two_sdata(sdata_pair) -> None:
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    sa, sb = sdata_pair
    pair = resolve_obs_pair(sa, sb, "tbl_ref", "tbl_query")
    assert pair.ref_container is sa
    assert pair.query_container is sb
    assert pair.ref_element_key == "tbl_ref"
    assert pair.query_element_key == "tbl_query"
    assert isinstance(pair.ref, AnnData)
    assert isinstance(pair.query, AnnData)


def test_resolve_obs_pair_two_sdata_requires_names(sdata_pair) -> None:
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    sa, sb = sdata_pair
    with pytest.raises(ValueError, match="adata_ref_name"):
        resolve_obs_pair(sa, sb, None, "tbl_query")
    with pytest.raises(ValueError, match="adata_query_name"):
        resolve_obs_pair(sa, sb, "tbl_ref", None)


def test_resolve_obs_pair_single_sdata(sdata_single) -> None:
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    pair = resolve_obs_pair(sdata_single, None, "tbl_ref", "tbl_query")
    assert pair.ref_container is sdata_single
    assert pair.query_container is sdata_single
    assert pair.ref_element_key == "tbl_ref"
    assert pair.query_element_key == "tbl_query"


def test_resolve_obs_pair_single_sdata_same_name_passes_through(sdata_single) -> None:
    """Same-name within one sdata is a valid no-op-ish call (identity fit).

    The resolver is not in the business of semantic-uniqueness validation;
    backends are free to treat identical inputs however they like.
    """
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    pair = resolve_obs_pair(sdata_single, None, "tbl_ref", "tbl_ref")
    assert pair.ref_element_key == "tbl_ref"
    assert pair.query_element_key == "tbl_ref"


def test_resolve_obs_pair_mixed_inputs_rejected(adata_pair, sdata_single) -> None:
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    a, _ = adata_pair
    with pytest.raises(TypeError, match="Mixed AnnData/SpatialData"):
        resolve_obs_pair(a, sdata_single, None, "tbl_ref")
    with pytest.raises(TypeError, match="Mixed AnnData/SpatialData"):
        resolve_obs_pair(sdata_single, a, "tbl_ref", None)


def test_resolve_obs_pair_anndata_without_query(adata_pair) -> None:
    from squidpy.experimental.tl._align._io import resolve_obs_pair

    a, _ = adata_pair
    with pytest.raises(ValueError, match="`data_query` is required"):
        resolve_obs_pair(a, None, None, None)


# ---------------------------------------------------------------------------
# 4. Multiscale image resolution
# ---------------------------------------------------------------------------


def test_resolve_image_pair_single_and_multiscale() -> None:
    from squidpy.experimental.tl._align._io import resolve_image_pair

    # Single-scale image
    arr = np.zeros((3, 32, 32), dtype=np.uint8)
    xa = xr.DataArray(arr, dims=["c", "y", "x"], coords={"c": ["R", "G", "B"]})
    single = Image2DModel.parse(xa)

    # Multiscale image
    multi = Image2DModel.parse(xa, scale_factors=[2])

    sdata = SpatialData(images={"single": single, "multi": multi})

    pair = resolve_image_pair(sdata, None, "single", "multi")
    assert isinstance(pair.ref, xr.DataArray)
    assert isinstance(pair.query, xr.DataArray)
    assert pair.ref_element_key == "single"
    assert pair.query_element_key == "multi"


def test_resolve_image_pair_same_name_passes_through() -> None:
    """Same image name in a single sdata is a valid no-op call."""
    from squidpy.experimental.tl._align._io import resolve_image_pair

    arr = np.zeros((3, 16, 16), dtype=np.uint8)
    xa = xr.DataArray(arr, dims=["c", "y", "x"], coords={"c": ["R", "G", "B"]})
    sdata = SpatialData(images={"img": Image2DModel.parse(xa)})

    pair = resolve_image_pair(sdata, None, "img", "img")
    assert pair.ref_element_key == "img"
    assert pair.query_element_key == "img"


# ---------------------------------------------------------------------------
# 5. Landmark validation
# ---------------------------------------------------------------------------


def test_validate_landmarks_unequal_length() -> None:
    from squidpy.experimental.tl._align._validation import validate_landmarks

    with pytest.raises(ValueError, match="same length"):
        validate_landmarks(((0, 0), (1, 1)), ((0, 0),), model="similarity")


def test_validate_landmarks_requires_three_points() -> None:
    """spatialdata's get_transformation_between_landmarks requires n>=3."""
    from squidpy.experimental.tl._align._validation import validate_landmarks

    with pytest.raises(ValueError, match="at least 3"):
        validate_landmarks(((0, 0), (1, 1)), ((0, 0), (1, 1)), model="similarity")


def test_validate_landmarks_outside_extent() -> None:
    from squidpy.experimental.tl._align._validation import validate_landmarks

    with pytest.raises(ValueError, match="outside the coordinate-system extent"):
        validate_landmarks(
            ((0, 0), (50, 50), (3, 3)),
            ((0, 0), (5, 5), (1, 1)),
            model="similarity",
            cs_ref_extent=(0, 0, 10, 10),
        )


def test_validate_landmarks_happy_path() -> None:
    from squidpy.experimental.tl._align._validation import validate_landmarks

    ref, query = validate_landmarks(
        ((0, 0), (10, 0), (0, 10)),
        ((1, 1), (11, 1), (1, 11)),
        model="affine",
        cs_ref_extent=(0, 0, 100, 100),
        cs_query_extent=(0, 0, 100, 100),
    )
    assert ref.shape == (3, 2)
    assert query.shape == (3, 2)


# ---------------------------------------------------------------------------
# 6. Output-mode guards
# ---------------------------------------------------------------------------


def test_align_images_rejects_output_mode_obs(sdata_single) -> None:
    import squidpy as sq

    with pytest.raises(ValueError, match="output_mode"):
        sq.experimental.tl.align_images(
            sdata_single,
            None,
            img_ref_name="img_ref",
            img_query_name="img_query",
            output_mode="obs",  # type: ignore[arg-type]
        )


def test_key_added_only_with_obs_mode(sdata_single) -> None:
    import squidpy as sq

    with pytest.raises(ValueError, match="key_added"):
        sq.experimental.tl.align_obs(
            sdata_single,
            None,
            adata_ref_name="tbl_ref",
            adata_query_name="tbl_query",
            output_mode="affine",
            key_added="aligned",
        )


# ---------------------------------------------------------------------------
# 7. Dispatch
# ---------------------------------------------------------------------------


def test_align_obs_stalign_image_path_raises(sdata_single) -> None:
    """``align_images(flavour='stalign')`` is still NotImplementedError; the
    PR-#1150 lift only ships point alignment.  This pins the contract that
    the dispatch reaches the backend cleanly (no ImportError/AttributeError)."""
    import squidpy as sq

    pytest.importorskip("jax")
    with pytest.raises(NotImplementedError, match="stalign image alignment"):
        sq.experimental.tl.align_images(
            sdata_single,
            None,
            img_ref_name="img_ref",
            img_query_name="img_query",
            flavour="stalign",
        )


def test_align_obs_moscot_dispatch_reaches_backend(sdata_single) -> None:
    import squidpy as sq

    pytest.importorskip("jax")
    with pytest.raises(NotImplementedError, match="moscot backend"):
        sq.experimental.tl.align_obs(
            sdata_single,
            None,
            adata_ref_name="tbl_ref",
            adata_query_name="tbl_query",
            flavour="moscot",
        )


def test_align_obs_unknown_flavour(sdata_single) -> None:
    import squidpy as sq

    with pytest.raises(ValueError, match="flavour"):
        sq.experimental.tl.align_obs(
            sdata_single,
            None,
            adata_ref_name="tbl_ref",
            adata_query_name="tbl_query",
            flavour="bogus",  # type: ignore[arg-type]
        )


def test_align_images_rejects_moscot(sdata_single) -> None:
    import squidpy as sq

    with pytest.raises(ValueError, match="flavour"):
        sq.experimental.tl.align_images(
            sdata_single,
            None,
            img_ref_name="img_ref",
            img_query_name="img_query",
            flavour="moscot",  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# 8. align_by_landmarks is JAX-free
# ---------------------------------------------------------------------------


def test_align_by_landmarks_two_cs_in_same_sdata(monkeypatch, sdata_two_cs) -> None:
    """Align two distinct coordinate systems inside a single SpatialData via
    the closed-form spatialdata fit.  Also pins that the landmark path never
    touches JAX: with ``jax`` blocked the call must still succeed, because
    the landmark backend is pure NumPy/spatialdata.
    """
    from spatialdata.transformations import Affine, get_transformation

    import squidpy as sq

    monkeypatch.setitem(sys.modules, "jax", None)

    # Three corresponding landmarks: identity translation by (+1, +2).
    landmarks_ref = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0))
    landmarks_query = ((1.0, 2.0), (11.0, 2.0), (1.0, 12.0))

    sq.experimental.tl.align_by_landmarks(
        sdata_two_cs,
        None,
        cs_name_ref="cs_a",
        cs_name_query="cs_b",
        landmarks_ref=landmarks_ref,
        landmarks_query=landmarks_query,
        model="similarity",
    )

    # The fit should now have attached an affine on `img_b` mapping cs_b -> cs_a.
    img_b = sdata_two_cs.images["img_b"]
    transforms = get_transformation(img_b, get_all=True)
    assert "cs_a" in transforms, f"alignment didn't register a cs_a transform on img_b; have {list(transforms)}"
    assert isinstance(transforms["cs_a"], Affine)


def test_align_by_landmarks_affine_model(sdata_two_cs) -> None:
    """The 6-DOF affine model fits via skimage and registers a transform."""
    from spatialdata.transformations import Affine, get_transformation

    import squidpy as sq

    # 4 landmarks (>3, since affine has 6 DOF and skimage wants over-determined input).
    landmarks_ref = ((0.0, 0.0), (10.0, 0.0), (0.0, 10.0), (10.0, 10.0))
    landmarks_query = ((1.0, 2.0), (11.0, 2.0), (1.0, 12.0), (11.0, 12.0))

    sq.experimental.tl.align_by_landmarks(
        sdata_two_cs,
        None,
        cs_name_ref="cs_a",
        cs_name_query="cs_b",
        landmarks_ref=landmarks_ref,
        landmarks_query=landmarks_query,
        model="affine",
    )

    img_b = sdata_two_cs.images["img_b"]
    transforms = get_transformation(img_b, get_all=True)
    assert "cs_a" in transforms
    assert isinstance(transforms["cs_a"], Affine)


# ---------------------------------------------------------------------------
# 9. JAX-required flavours fail cleanly without JAX
# ---------------------------------------------------------------------------


def test_stalign_without_jax_raises_importerror(monkeypatch, sdata_single) -> None:
    import squidpy as sq

    # Block the import: the lazy `import jax` inside _jax.require_jax will hit None.
    monkeypatch.setitem(sys.modules, "jax", None)

    with pytest.raises(ImportError, match="JAX is required"):
        sq.experimental.tl.align_obs(
            sdata_single,
            None,
            adata_ref_name="tbl_ref",
            adata_query_name="tbl_query",
            flavour="stalign",
        )
