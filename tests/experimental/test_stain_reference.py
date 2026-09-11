from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._reference import StainFit

# Tests construct stain matrices and background estimates by hand; there is
# no library-wide pure-white default to lean on.
_TEST_BACKGROUND = np.array([245.0, 250.0, 240.0])


def _ruifrok_matrix() -> np.ndarray:
    third = np.cross(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
    third /= np.linalg.norm(third)
    return np.column_stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"], third])


def test_macenko_basic() -> None:
    ref = StainFit(
        method="macenko",
        stain_matrix=_ruifrok_matrix(),
        white_point=_TEST_BACKGROUND,
    )
    assert ref.method == "macenko"
    assert ref.stain_matrix.shape == (3, 3)
    assert ref.mu is None and ref.sigma is None
    np.testing.assert_array_equal(ref.white_point, _TEST_BACKGROUND)


def test_reinhard_basic() -> None:
    ref = StainFit(method="reinhard", mu=np.array([1.0, 0.5, -0.2]), sigma=np.array([0.1, 0.1, 0.1]))
    assert ref.method == "reinhard"
    assert ref.stain_matrix is None
    assert ref.white_point is None


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        StainFit(method="not-a-method")  # type: ignore[arg-type]


def test_decomposition_requires_stain_matrix() -> None:
    with pytest.raises(ValueError, match="requires stain_matrix"):
        StainFit(method="macenko", white_point=_TEST_BACKGROUND)


def test_decomposition_requires_white_point() -> None:
    with pytest.raises(ValueError, match="requires white_point"):
        StainFit(method="macenko", stain_matrix=_ruifrok_matrix())


def test_decomposition_forbids_mu_sigma() -> None:
    with pytest.raises(ValueError, match="forbids mu/sigma"):
        StainFit(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            white_point=_TEST_BACKGROUND,
            mu=np.zeros(3),
            sigma=np.ones(3),
        )


def test_reinhard_requires_mu_and_sigma() -> None:
    with pytest.raises(ValueError, match="requires both mu and sigma"):
        StainFit(method="reinhard", mu=np.zeros(3))


def test_reinhard_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        StainFit(method="reinhard", mu=np.zeros(3), sigma=np.array([1.0, 0.0, 1.0]))


def test_reinhard_forbids_stain_matrix() -> None:
    with pytest.raises(ValueError, match="forbids stain_matrix"):
        StainFit(
            method="reinhard",
            mu=np.zeros(3),
            sigma=np.ones(3),
            stain_matrix=_ruifrok_matrix(),
        )


def test_reinhard_forbids_white_point() -> None:
    with pytest.raises(ValueError, match="forbids white_point"):
        StainFit(
            method="reinhard",
            mu=np.zeros(3),
            sigma=np.ones(3),
            white_point=_TEST_BACKGROUND,
        )


def test_bad_white_point() -> None:
    with pytest.raises(ValueError, match="white_point"):
        StainFit(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            white_point=np.array([255.0, -1.0, 255.0]),
        )


def test_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"stain_matrix must have shape"):
        StainFit(
            method="macenko",
            stain_matrix=np.zeros((2, 3)),
            white_point=_TEST_BACKGROUND,
        )


def test_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match=r"mu contains non-finite values"):
        StainFit(
            method="reinhard",
            mu=np.array([np.nan, 0.0, 0.0]),
            sigma=np.ones(3),
        )


def test_equality_is_array_aware_and_hashable() -> None:
    """Two fits holding equal arrays compare equal, and a fit can key a dict.

    The dataclass-generated `__eq__` cannot do this -- comparing array fields raises
    "truth value of an array is ambiguous" -- so `eq=False` plus an explicit `__eq__` is
    what makes `fit in fits` and `{fit: slide}` work.
    """
    a = StainFit(method="reinhard", mu=np.array([1.0, 2.0, 3.0]), sigma=np.ones(3))
    b = StainFit(method="reinhard", mu=np.array([1.0, 2.0, 3.0]), sigma=np.ones(3))
    c = StainFit(method="reinhard", mu=np.array([9.0, 9.0, 9.0]), sigma=np.ones(3))
    assert a == b
    assert a != c
    assert a in [c, b]
    assert {a: "slide-1"}[a] == "slide-1"
    assert len({a, b}) == 2, "hashing stays identity-based; array fields are unhashable"
