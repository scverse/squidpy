from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._reference import StainReference

# Tests construct stain matrices and background estimates by hand; there is
# no library-wide pure-white default to lean on.
_TEST_BACKGROUND = np.array([245.0, 250.0, 240.0])


def _ruifrok_matrix() -> np.ndarray:
    third = np.cross(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
    third /= np.linalg.norm(third)
    return np.column_stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"], third])


def test_macenko_basic() -> None:
    ref = StainReference(
        method="macenko",
        stain_matrix=_ruifrok_matrix(),
        background_intensity=_TEST_BACKGROUND,
    )
    assert ref.method == "macenko"
    assert ref.stain_matrix.shape == (3, 3)
    assert ref.mu is None and ref.sigma is None
    np.testing.assert_array_equal(ref.background_intensity, _TEST_BACKGROUND)


def test_reinhard_basic() -> None:
    ref = StainReference(method="reinhard", mu=np.array([1.0, 0.5, -0.2]), sigma=np.array([0.1, 0.1, 0.1]))
    assert ref.method == "reinhard"
    assert ref.stain_matrix is None
    assert ref.background_intensity is None


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        StainReference(method="not-a-method")  # type: ignore[arg-type]


def test_decomposition_requires_stain_matrix() -> None:
    with pytest.raises(ValueError, match="requires stain_matrix"):
        StainReference(method="macenko", background_intensity=_TEST_BACKGROUND)


def test_decomposition_requires_background_intensity() -> None:
    with pytest.raises(ValueError, match="requires background_intensity"):
        StainReference(method="macenko", stain_matrix=_ruifrok_matrix())


def test_decomposition_forbids_mu_sigma() -> None:
    with pytest.raises(ValueError, match="forbids mu/sigma"):
        StainReference(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            background_intensity=_TEST_BACKGROUND,
            mu=np.zeros(3),
            sigma=np.ones(3),
        )


def test_reinhard_requires_mu_and_sigma() -> None:
    with pytest.raises(ValueError, match="requires both mu and sigma"):
        StainReference(method="reinhard", mu=np.zeros(3))


def test_reinhard_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        StainReference(method="reinhard", mu=np.zeros(3), sigma=np.array([1.0, 0.0, 1.0]))


def test_reinhard_forbids_stain_matrix() -> None:
    with pytest.raises(ValueError, match="forbids stain_matrix"):
        StainReference(
            method="reinhard",
            mu=np.zeros(3),
            sigma=np.ones(3),
            stain_matrix=_ruifrok_matrix(),
        )


def test_reinhard_forbids_background_intensity() -> None:
    with pytest.raises(ValueError, match="forbids background_intensity"):
        StainReference(
            method="reinhard",
            mu=np.zeros(3),
            sigma=np.ones(3),
            background_intensity=_TEST_BACKGROUND,
        )


def test_bad_background_intensity() -> None:
    with pytest.raises(ValueError, match="background_intensity"):
        StainReference(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            background_intensity=np.array([255.0, -1.0, 255.0]),
        )


def test_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"stain_matrix must have shape"):
        StainReference(
            method="macenko",
            stain_matrix=np.zeros((2, 3)),
            background_intensity=_TEST_BACKGROUND,
        )


def test_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match=r"mu contains non-finite values"):
        StainReference(
            method="reinhard",
            mu=np.array([np.nan, 0.0, 0.0]),
            sigma=np.ones(3),
        )
