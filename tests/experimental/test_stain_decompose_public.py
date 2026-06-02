from __future__ import annotations

import numpy as np
import pytest
import spatialdata as sd
import xarray as xr
from spatialdata.models import Image2DModel, Labels2DModel
from spatialdata.transformations import get_transformation

import squidpy as sq
from squidpy.experimental.im import (
    StainReference,
    decompose_stains,
    fit_stain_reference,
    normalize_stains,
)
from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._conversion import sda_to_rgb
from squidpy.experimental.im._stain._validation import (
    StainFittingError,
    complement_third_column,
    reorder_to_canonical,
)

_WHITE = np.array([255.0, 255.0, 255.0])


def _synthetic_rgb(seed: int = 0, n_side: int = 48, white: np.ndarray = _WHITE) -> np.ndarray:
    w = complement_third_column(
        reorder_to_canonical(np.stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"]], axis=1))
    )
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    conc = rng.uniform(0.0, 70.0, size=(n, 2))
    third = n // 3
    conc[:third, 1] = 0.0
    conc[third : 2 * third, 0] = 0.0
    od = (conc @ w[:, :2].T).T.reshape(3, n_side, n_side)
    rgb = sda_to_rgb(xr.DataArray(od, dims=("c", "y", "x")), white)
    return np.asarray(rgb.data).astype(np.uint8)


def _make_sdata(values: np.ndarray, *, with_tissue: bool = True) -> sd.SpatialData:
    sdata = sd.SpatialData(images={"img": Image2DModel.parse(values, dims=("c", "y", "x"))})
    if with_tissue:
        h, w = values.shape[-2], values.shape[-1]
        sdata.labels["img_tissue"] = Labels2DModel.parse(np.ones((h, w), dtype=np.uint32), dims=("y", "x"))
    return sdata


@pytest.mark.parametrize("method", ["macenko", "vahadane"])
class TestDecompositionThroughDispatchers:
    def test_fit_and_apply_end_to_end(self, method: str) -> None:
        sdata = _make_sdata(_synthetic_rgb(seed=1))
        ref = fit_stain_reference(sdata, "img", method=method, white_point=_WHITE)
        assert ref.method == method
        assert ref.stain_matrix.shape == (3, 3)
        assert ref.max_concentrations.shape == (2,)

        out = normalize_stains(sdata, "img", ref, inplace=False)
        assert isinstance(out, xr.DataArray)
        assert out.sizes["c"] == 3

    def test_apply_writes_back(self, method: str) -> None:
        sdata = _make_sdata(_synthetic_rgb(seed=2))
        ref = fit_stain_reference(sdata, "img", method=method, white_point=_WHITE)
        result = normalize_stains(sdata, "img", ref, image_key_added="norm")
        assert result is None
        assert get_transformation(sdata.images["norm"], get_all=True).keys() == (
            get_transformation(sdata.images["img"], get_all=True).keys()
        )


class TestDecomposeStains:
    def test_returns_named_concentration_maps(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        conc = decompose_stains(sdata, "img", "macenko", white_point=_WHITE, inplace=False)
        assert set(conc) == {"hematoxylin", "eosin", "residual"}
        assert all(set(c.dims) == {"y", "x"} for c in conc.values())  # one (y, x) map per stain
        assert all(c.dtype == np.float16 for c in conc.values())  # default output_dtype

    def test_drop_residual(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        conc = decompose_stains(sdata, "img", "macenko", white_point=_WHITE, include_residual=False, inplace=False)
        assert set(conc) == {"hematoxylin", "eosin"}

    def test_output_dtype_override(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        conc = decompose_stains(sdata, "img", "macenko", white_point=_WHITE, output_dtype=np.float32, inplace=False)
        assert all(c.dtype == np.float32 for c in conc.values())

    def test_inplace_default_writes_derived_keys(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        ref = fit_stain_reference(sdata, "img", method="macenko", white_point=_WHITE)
        out = decompose_stains(sdata, "img", ref)  # inplace=True, prefix defaults to image_key
        assert out is None
        for stain in ("hematoxylin", "eosin", "residual"):
            assert f"img_{stain}" in sdata.images

    def test_with_reference_writes_separate_images(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        ref = fit_stain_reference(sdata, "img", method="macenko", white_point=_WHITE)
        out = decompose_stains(sdata, "img", ref, image_key_added="conc")
        assert out is None
        for stain in ("hematoxylin", "eosin", "residual"):
            assert f"conc_{stain}" in sdata.images
            assert list(sdata.images[f"conc_{stain}"].coords["c"].values) == [stain]

    def test_atomic_write_aborts_on_any_existing_key(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        ref = fit_stain_reference(sdata, "img", method="macenko", white_point=_WHITE)
        # pre-occupy only the *eosin* target; the whole write must abort, leaving
        # no half-written hematoxylin/residual behind.
        sdata.images["conc_eosin"] = sdata.images["img"]
        with pytest.raises(ValueError, match="would overwrite"):
            decompose_stains(sdata, "img", ref, image_key_added="conc")
        assert "conc_hematoxylin" not in sdata.images
        assert "conc_residual" not in sdata.images

    def test_reinhard_reference_rejected(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        reinhard_ref = fit_stain_reference(sdata, "img", method="reinhard")
        with pytest.raises(ValueError, match="macenko/vahadane reference"):
            decompose_stains(sdata, "img", reinhard_ref)

    def test_bad_method_rejected(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        with pytest.raises(ValueError, match="method must be"):
            decompose_stains(sdata, "img", "reinhard")


class TestBackgroundDefault:
    def test_fit_defaults_to_white_when_absent(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        ref = fit_stain_reference(sdata, "img", method="macenko")
        # default I_0 is a fixed full-white point, not an image-derived estimate
        np.testing.assert_array_equal(ref.white_point, [255.0, 255.0, 255.0])

    def test_explicit_background_is_used(self) -> None:
        I0 = np.array([240.0, 245.0, 250.0])
        # build the synthetic image against this white point so the fit is consistent
        sdata = _make_sdata(_synthetic_rgb(white=I0))
        ref = fit_stain_reference(sdata, "img", method="vahadane", white_point=I0)
        np.testing.assert_array_equal(ref.white_point, I0)


class TestUnknownMethod:
    def test_fit_unknown_method_raises(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        with pytest.raises(ValueError, match="Unknown method"):
            fit_stain_reference(sdata, "img", method="bogus")


class TestDefaultMethodAndGate:
    def test_default_method_is_macenko(self) -> None:
        sdata = _make_sdata(_synthetic_rgb())
        ref = fit_stain_reference(sdata, "img")  # no method -> default
        assert ref.method == "macenko"

    def test_max_angle_deg_gate_too_strict_raises(self) -> None:
        # an impossibly tight tolerance trips the H/E sanity gate
        sdata = _make_sdata(_synthetic_rgb())
        with pytest.raises(StainFittingError, match="deviates"):
            fit_stain_reference(sdata, "img", method="macenko", white_point=_WHITE, max_angle_deg=0.01)

    def test_canonical_reference_passthrough(self) -> None:
        # passing the Ruifrok canonical explicitly reproduces the default fit
        sdata = _make_sdata(_synthetic_rgb())
        default = fit_stain_reference(sdata, "img", method="macenko", white_point=_WHITE)
        custom = fit_stain_reference(sdata, "img", method="macenko", white_point=_WHITE, canonical_reference=RUIFROK_HE)
        np.testing.assert_allclose(default.stain_matrix, custom.stain_matrix)


class TestDecompositionOnHnE:
    # Correctness is gated by the synthetic-recovery tests above (per the arc
    # decision); these are real-data smoke checks that the pipeline fits a
    # valid matrix, applies lazily, and decomposes - fast because the source
    # matrix is fit on the coarse level, not the full-resolution image.
    # Both methods are exercised: once the fit consumes a real detect_tissue
    # mask (dropping the fiducial ring + dim background), Macenko fits this
    # low-contrast Visium H&E cleanly and agrees with Vahadane.
    @pytest.mark.parametrize("method", ["macenko", "vahadane"])
    def test_fit_apply_decompose_smoke(self, sdata_hne, method: str) -> None:
        image_key = next(iter(sdata_hne.images))
        sq.experimental.im.detect_tissue(sdata_hne, image_key)
        ref = sq.experimental.im.fit_stain_reference(sdata_hne, image_key, method=method)
        assert isinstance(ref, StainReference)
        assert ref.stain_matrix.shape == (3, 3)
        normalized = sq.experimental.im.normalize_stains(sdata_hne, image_key, ref, inplace=False)
        assert normalized.sizes["c"] == 3
        conc = sq.experimental.im.decompose_stains(sdata_hne, image_key, ref, inplace=False)
        assert set(conc) == {"hematoxylin", "eosin", "residual"}
