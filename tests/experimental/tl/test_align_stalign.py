"""Smoke tests for the STalign align functions.

Deliberately tiny: `TINY_SOLVER` runs one or two iterations, so these assert that the
whole path executes and writes where it says it does, not that the fit converges.
"""

from __future__ import annotations

import numpy as np
import pytest
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, Image3DModel

from tests.experimental.conftest import ALIGN_PTS, TINY_SOLVER, make_adata

pytest.importorskip("jax")

from squidpy.experimental.tl import align_stalign_image, align_stalign_obs, align_stalign_slice  # noqa: E402

IMAGE_SOLVER = {"a": 4.0, "nt": 1, "niter": 2, "epV": 1.0}


def _sdata_image(array: np.ndarray, key: str, **kwargs: object) -> SpatialData:
    model = Image3DModel if array.ndim == 4 else Image2DModel
    dims = ("c", "z", "y", "x") if array.ndim == 4 else ("c", "y", "x")
    return SpatialData(images={key: model.parse(array, dims=dims, **kwargs)})


def test_obs_fit_returns_a_result_and_writes_a_copy() -> None:
    ref, query = make_adata(ALIGN_PTS), make_adata(ALIGN_PTS + 0.4)

    result = align_stalign_obs(ref, query, **TINY_SOLVER)
    assert result.aligned_points.shape == ALIGN_PTS.shape

    out = align_stalign_obs(ref, query, key_added="aligned", **TINY_SOLVER)
    assert out.obsm["aligned"].shape == ALIGN_PTS.shape
    assert "aligned" not in query.obsm, "the input must be left untouched"


@pytest.mark.parametrize("scale_factors", [None, [2]], ids=["single_scale", "multiscale"])
def test_image_fit_reads_both_element_layouts(scale_factors: list[int] | None) -> None:
    """A multiscale element is a DataTree, which carries a `.dims` attribute of its own.

    Duck-typing on that attribute silently skipped the scale resolution and crashed on
    every multiscale image -- the layout spatialdata's readers actually produce.
    """
    image = np.random.default_rng(0).random((1, 32, 32))
    kwargs = {} if scale_factors is None else {"scale_factors": scale_factors}
    sdata_ref = _sdata_image(image, "img", **kwargs)
    sdata_query = _sdata_image(np.roll(image, 1, axis=1), "img", **kwargs)

    result = align_stalign_image(sdata_ref, sdata_query, image_key="img", **IMAGE_SOLVER)
    assert result.ref_axes[0].shape == (32,)

    out = align_stalign_image(sdata_ref, sdata_query, image_key="img", key_added="warped", **IMAGE_SOLVER)
    assert out.images["warped"].shape == image.shape
    assert "warped" not in sdata_query.images, "the input must be left untouched"


def test_slice_fit_places_a_section_in_a_volume() -> None:
    volume = np.random.default_rng(0).random((1, 6, 12, 12))
    sdata_ref = _sdata_image(volume, "volume")
    sdata_query = _sdata_image(volume[:, 3], "section")

    result = align_stalign_slice(sdata_ref, sdata_query, image_key=("volume", "section"), a=3.0, nt=1, niter=2, epV=1.0)

    assert result.affine_xyz.shape == (4, 4)
    assert result.transform(ALIGN_PTS).shape == (len(ALIGN_PTS), 3)


def test_a_2d_reference_is_rejected_with_a_pointer_to_the_2d_path() -> None:
    section = np.random.default_rng(0).random((1, 12, 12))
    sdata = _sdata_image(section, "section")

    with pytest.raises(ValueError, match=r"align_stalign_image"):
        align_stalign_slice(sdata, sdata, image_key=("section", "section"), **IMAGE_SOLVER)
