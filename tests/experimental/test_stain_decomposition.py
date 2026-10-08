from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from squidpy._params import resolve_params
from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._conversion import sda_to_rgb
from squidpy.experimental.im._stain._decomposition import (
    apply_decomposition,
    fit_decomposition,
)
from squidpy.experimental.im._stain._validation import (
    StainFittingError,
    angle_between_deg,
    complement_third_column,
    reorder_to_canonical,
)
from squidpy.types import MacenkoParams, VahadaneParams

_WHITE = np.array([255.0, 255.0, 255.0])


def _canonical(h: np.ndarray, e: np.ndarray) -> np.ndarray:
    return complement_third_column(reorder_to_canonical(np.stack([h, e], axis=1)))


def _synthetic_he(stain_matrix: np.ndarray, *, n_side: int = 48, seed: int = 0, chunked: bool = False) -> xr.DataArray:
    """Build an RGB image from known H/E concentrations and a stain matrix."""
    rng = np.random.default_rng(seed)
    n = n_side * n_side
    conc = rng.uniform(0.0, 70.0, size=(n, 2))
    # dense pure-H / pure-E populations so the angular extremes are well sampled
    # (real H&E slides have many near-pure pixels; a uniform mix under-samples them)
    third = n // 3
    conc[:third, 1] = 0.0  # pure-H pixels (one angular extreme)
    conc[third : 2 * third, 0] = 0.0  # pure-E pixels (the other extreme)
    od = (conc @ stain_matrix[:, :2].T).T.reshape(3, n_side, n_side)
    data = da.from_array(od, chunks=(3, 16, 16)) if chunked else od
    return sda_to_rgb(xr.DataArray(data, dims=("c", "y", "x")), _WHITE)


class TestMacenko:
    @pytest.mark.parametrize("chunked", [False, True])
    def test_recovers_planted_matrix(self, chunked: bool) -> None:
        truth = _canonical(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
        img = _synthetic_he(truth, chunked=chunked)
        ref = fit_decomposition(img, "macenko", resolve_params(None, MacenkoParams), _WHITE)
        assert angle_between_deg(ref.stain_matrix[:, 0], truth[:, 0]) < 12.0
        assert angle_between_deg(ref.stain_matrix[:, 1], truth[:, 1]) < 12.0
        assert ref.max_concentrations.shape == (2,)
        assert np.all(ref.max_concentrations > 0)


class TestVahadane:
    def test_recovers_planted_matrix(self) -> None:
        truth = _canonical(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
        img = _synthetic_he(truth)
        ref = fit_decomposition(img, "vahadane", resolve_params(None, VahadaneParams), _WHITE)
        assert angle_between_deg(ref.stain_matrix[:, 0], truth[:, 0]) < 20.0
        assert angle_between_deg(ref.stain_matrix[:, 1], truth[:, 1]) < 20.0


class TestApplyDecomposition:
    def test_transfer_matches_reference_matrix(self) -> None:
        truth_a = _canonical(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
        # a slightly rotated source staining
        e_shift = RUIFROK_HE["eosin"] + 0.15 * RUIFROK_HE["hematoxylin"]
        truth_b = _canonical(RUIFROK_HE["hematoxylin"], e_shift / np.linalg.norm(e_shift))

        img_a = _synthetic_he(truth_a, seed=1)
        img_b = _synthetic_he(truth_b, seed=2)
        ref_a = fit_decomposition(img_a, "macenko", resolve_params(None, MacenkoParams), _WHITE)

        normalized = apply_decomposition(img_b, ref_a, resolve_params(None, MacenkoParams))
        refit = fit_decomposition(normalized, "macenko", resolve_params(None, MacenkoParams), _WHITE)
        assert angle_between_deg(refit.stain_matrix[:, 0], ref_a.stain_matrix[:, 0]) < 12.0
        assert angle_between_deg(refit.stain_matrix[:, 1], ref_a.stain_matrix[:, 1]) < 12.0

    def test_lazy_in_lazy_out(self) -> None:
        truth = _canonical(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
        ref = fit_decomposition(_synthetic_he(truth), "macenko", resolve_params(None, MacenkoParams), _WHITE)
        out = apply_decomposition(_synthetic_he(truth, chunked=True), ref, resolve_params(None, MacenkoParams))
        assert isinstance(out.data, da.Array)

    def test_max_concentrations_ignored_on_apply(self) -> None:
        # Colour-basis transfer carries no maxC_ref/maxC_src term: two references
        # with the same stain matrix + white point but different max_concentrations
        # produce identical output. A canonical-Macenko intensity rescale would not.
        from squidpy.experimental.im._stain._reference import StainReference

        truth = _canonical(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
        img = _synthetic_he(truth, seed=2)
        ref1 = fit_decomposition(_synthetic_he(truth, seed=1), "macenko", resolve_params(None, MacenkoParams), _WHITE)
        ref2 = StainReference(
            method="macenko",
            stain_matrix=ref1.stain_matrix,
            white_point=ref1.white_point,
            max_concentrations=ref1.max_concentrations * 5.0,
        )
        out1 = apply_decomposition(img, ref1, resolve_params(None, MacenkoParams)).values
        out2 = apply_decomposition(img, ref2, resolve_params(None, MacenkoParams)).values
        np.testing.assert_array_equal(out1, out2)


class TestDegenerate:
    def test_empty_tissue_raises(self) -> None:
        white = xr.DataArray(np.full((3, 16, 16), 255.0), dims=("c", "y", "x"))
        with pytest.raises(StainFittingError, match="mask is empty"):
            fit_decomposition(white, "macenko", resolve_params(None, MacenkoParams), _WHITE)


@pytest.mark.parametrize(("method", "spec"), [("macenko", MacenkoParams), ("vahadane", VahadaneParams)])
def test_private_fns_trust_resolved_params(method: str, spec: type) -> None:
    # the public dispatchers resolve + validate once; the private fns neither merge defaults nor re-validate
    img = _synthetic_he(_canonical(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"]))
    with pytest.raises(KeyError, match="beta"):
        fit_decomposition(img, method, {}, _WHITE)
    fit_decomposition(img, method, {**resolve_params(None, spec), "bogus": 1}, _WHITE)
