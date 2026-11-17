from __future__ import annotations

import copy

import numpy as np
import pytest
import spatialdata_plot as sdp

import squidpy as sq
from tests.conftest import PlotTester, PlotTesterMeta

_ = sdp

# Large tile size to keep tests fast (fewer tiles = less computation).
_FAST_TILE = (1000, 1000)


@pytest.fixture(scope="session")
def _sdata_hne_with_tissue():
    """Load the Visium H&E dataset once and pre-compute the tissue mask.

    The tissue mask is the most expensive setup step (~16 s).  Pre-computing
    it here and deepcopying per test avoids paying that cost repeatedly.
    """
    sdata = sq.datasets.visium_hne_sdata()
    sq.experimental.im.detect_tissue(sdata, image_key="hne", scale="scale0", inplace=True, new_labels_key="hne_tissue")
    return sdata


@pytest.fixture()
def sdata_hne(_sdata_hne_with_tissue):
    """Per-test deep copy so each test gets a fresh SpatialData with tissue mask."""
    return copy.deepcopy(_sdata_hne_with_tissue)


def test_qc_image_accepts_a_bare_metric_string(sdata_hne) -> None:
    """A single metric name, not wrapped in a list.

    `QCMetric` was a `StrEnum`, but the guard was `isinstance(m, QCMetric)`, which is False
    for the plain string the member compares equal to. So this raised `TypeError` while
    the docs advertised the string form.
    """
    sq.experimental.im.qc_image(sdata_hne, image_key="hne", tile_size=_FAST_TILE, metrics="tenengrad", progress=False)
    assert any("tenengrad" in name for name in sdata_hne.tables["qc_img_hne"].var_names)


@pytest.mark.parametrize(
    ("metrics", "match"),
    [(["not_a_metric"], r"Unknown metrics \['not_a_metric'\]"), (5, r"Unknown metrics \[5\]\. Available")],
    ids=["unknown_name", "not_iterable"],
)
def test_qc_image_names_the_unknown_metric(sdata_hne, metrics, match) -> None:
    with pytest.raises(ValueError, match=match):
        sq.experimental.im.qc_image(sdata_hne, image_key="hne", tile_size=_FAST_TILE, metrics=metrics, progress=False)


class TestQCImage(PlotTester, metaclass=PlotTesterMeta):
    def test_plot_calc_qc_image_hne(self, sdata_hne):
        """Test QC image overlay with a single sharpness metric."""
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            metrics=["tenengrad"],
            progress=False,
        )

        (
            sdata_hne.pl.render_images()
            .pl.render_shapes("qc_img_hne_grid", color="qc_outlier", groups="True", palette="red", fill_alpha=1.0)
            .pl.show()
        )

    def test_plot_calc_qc_image_not_hne(self, sdata_hne):
        """Test QC image overlay with is_hne=False default metrics."""
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            is_hne=False,
            metrics=None,
            progress=False,
        )

        (
            sdata_hne.pl.render_images()
            .pl.render_shapes("qc_img_hne_grid", color="qc_outlier", groups="True", palette="red", fill_alpha=1.0)
            .pl.show()
        )

    def test_plot_plot_qc_image(self, sdata_hne):
        """Test QC image plotting function."""
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            metrics=["tenengrad"],
            progress=False,
        )

        sq.experimental.pl.qc_image(
            sdata_hne,
            image_key="hne",
        )


def test_qc_image_hne_metric_without_hne_flag(sdata_hne):
    """Test that H&E metrics raise ValueError when is_hne=False."""
    with pytest.raises(ValueError, match="H&E-specific metrics"):
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            is_hne=False,
            metrics=["hematoxylin_mean"],
        )


def test_qc_image_default_metrics_hne(sdata_hne):
    """Test that default H&E metrics produce expected var_names."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=None,
        is_hne=True,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    expected = {
        "qc_edge_sharpness",
        "qc_tenengrad",
        "qc_var_of_laplacian",
        "qc_entropy",
        "qc_brightness_mean",
        "qc_hematoxylin_mean",
        "qc_eosin_mean",
    }
    assert set(adata.var_names) == expected


def test_qc_image_default_metrics_generic(sdata_hne):
    """Test that default generic metrics produce expected var_names."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=None,
        is_hne=False,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    expected = {
        "qc_edge_sharpness",
        "qc_tenengrad",
        "qc_var_of_laplacian",
        "qc_entropy",
        "qc_brightness_mean",
    }
    assert set(adata.var_names) == expected


def test_qc_image_rgb_metric(sdata_hne):
    """Test running a single RGB metric."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        is_hne=True,
        metrics=["hematoxylin_mean"],
        detect_tissue=False,
        detect_outliers=False,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    assert "qc_hematoxylin_mean" in adata.var_names


def test_qc_image_outlier_detection_with_tissue(sdata_hne):
    """Test that outlier detection with tissue classification populates expected columns."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=["tenengrad"],
        detect_outliers=True,
        detect_tissue=True,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    assert "qc_outlier" in adata.obs.columns
    assert "is_tissue" in adata.obs.columns
    assert "is_background" in adata.obs.columns
    assert "unfocus_score" in adata.obs.columns
    assert set(adata.obs["qc_outlier"].cat.categories) == {"False", "True"}
    # At least some tiles should be classified as tissue
    assert (adata.obs["is_tissue"] == "True").any()


def test_qc_image_outlier_detection_without_tissue(sdata_hne):
    """Test outlier detection without tissue classification scores all tiles."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=["tenengrad"],
        detect_outliers=True,
        detect_tissue=False,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    assert "qc_outlier" in adata.obs.columns
    assert "unfocus_score" in adata.obs.columns
    # Without tissue detection, these columns should not exist
    assert "is_tissue" not in adata.obs.columns
    assert "is_background" not in adata.obs.columns
    # All tiles should have a valid unfocus score (no NaNs)
    assert not np.any(np.isnan(adata.obs["unfocus_score"].values))


def test_qc_image_compute_only(sdata_hne):
    """Test compute-only mode without outlier detection."""
    sq.experimental.im.qc_image(
        sdata_hne,
        image_key="hne",
        tile_size=_FAST_TILE,
        metrics=["tenengrad", "brightness_mean"],
        detect_outliers=False,
        detect_tissue=False,
        progress=False,
    )
    adata = sdata_hne.tables["qc_img_hne"]
    assert set(adata.var_names) == {"qc_tenengrad", "qc_brightness_mean"}
    # No outlier-related columns
    assert "qc_outlier" not in adata.obs.columns
    assert "unfocus_score" not in adata.obs.columns
    # Spatial coordinates should still be present
    assert "centroid_y" in adata.obs.columns
    assert "spatial" in adata.obsm


def _synthetic_tile(seed: int = 0) -> np.ndarray:
    """Random dark blobs on a bright background: sharp intensity steps, like nuclei in H&E."""
    from scipy import ndimage as ndi

    rng = np.random.default_rng(seed)
    img = np.full((256, 256), 220.0, dtype=np.float32)
    for _ in range(40):
        y, x = rng.integers(10, 246, 2)
        r = rng.integers(3, 8)
        yy, xx = np.ogrid[:256, :256]
        img[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 80.0
    return ndi.gaussian_filter(img, 0.5)  # a scanner's own slight softness


def test_edge_sharpness_falls_with_blur_and_ignores_contrast() -> None:
    from scipy import ndimage as ndi

    from squidpy.experimental.im._sharpness_metrics import edge_sharpness

    tile = _synthetic_tile()
    values = [float(edge_sharpness(ndi.gaussian_filter(tile, s))[0, 0]) for s in (0, 1, 2, 4)]
    assert values == sorted(values, reverse=True)
    # halving the contrast of every step leaves the edge width, and the score, nearly unchanged
    low_contrast = 150 + (tile - 150) * 0.5
    assert abs(float(edge_sharpness(low_contrast)[0, 0]) - values[0]) < 0.05 * values[0]


def test_edge_sharpness_is_nan_without_edges() -> None:
    from squidpy.experimental.im._sharpness_metrics import edge_sharpness

    assert np.isnan(edge_sharpness(np.full((128, 128), 200.0, dtype=np.float32))[0, 0])


def test_robust_unfocus_z_flags_only_the_blurred_tile() -> None:
    from scipy import ndimage as ndi

    from squidpy.experimental.im._qc_image import _robust_unfocus_z
    from squidpy.experimental.im._sharpness_metrics import edge_sharpness

    tiles = [_synthetic_tile(seed) for seed in range(20)]
    tiles[7] = ndi.gaussian_filter(tiles[7], 3)
    z = _robust_unfocus_z(np.array([edge_sharpness(t)[0, 0] for t in tiles]))
    assert np.flatnonzero(z > 3).tolist() == [7]


def test_qc_image_robust_z_needs_edge_sharpness(sdata_hne) -> None:
    with pytest.raises(ValueError, match="edge_sharpness"):
        sq.experimental.im.qc_image(
            sdata_hne,
            image_key="hne",
            tile_size=_FAST_TILE,
            metrics=["tenengrad"],
            outlier_method="robust_z",
            progress=False,
            preview=False,
        )


def test_qc_image_default_outlier_method_follows_metrics(sdata_hne) -> None:
    sq.experimental.im.qc_image(sdata_hne, image_key="hne", tile_size=_FAST_TILE, progress=False, preview=False)
    assert sdata_hne.tables["qc_img_hne"].uns["qc_image"]["outlier_method"] == "robust_z"
    sq.experimental.im.qc_image(
        sdata_hne, image_key="hne", tile_size=_FAST_TILE, metrics=["tenengrad"], progress=False, preview=False
    )
    assert sdata_hne.tables["qc_img_hne"].uns["qc_image"]["outlier_method"] == "percentile"
