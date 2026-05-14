from __future__ import annotations

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import RUIFROK_HE
from squidpy.experimental.im._stain._validation import (
    StainFittingError,
    complement_third_column,
    reorder_to_canonical,
    validate_stain_matrix,
)


def _canonical_he() -> np.ndarray:
    return np.column_stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"]])


def _canonical_full() -> np.ndarray:
    return complement_third_column(_canonical_he())


class TestReorderToCanonical:
    def test_already_canonical_unchanged(self) -> None:
        W = _canonical_he()
        out = reorder_to_canonical(W)
        np.testing.assert_allclose(out, W, atol=1e-12)

    def test_swapped_columns_restored(self) -> None:
        W = _canonical_he()
        swapped = W[:, ::-1]
        out = reorder_to_canonical(swapped)
        np.testing.assert_allclose(out, W, atol=1e-12)

    def test_sign_flipped_columns_restored(self) -> None:
        W = _canonical_he()
        flipped = W.copy()
        flipped[:, 0] *= -1.0
        flipped[:, 1] *= -1.0
        out = reorder_to_canonical(flipped)
        np.testing.assert_allclose(out, W, atol=1e-12)

    def test_three_column_input_preserves_third(self) -> None:
        W = _canonical_full()
        swapped = W.copy()
        swapped[:, [0, 1]] = swapped[:, [1, 0]]
        out = reorder_to_canonical(swapped)
        np.testing.assert_allclose(out[:, :2], W[:, :2], atol=1e-12)
        np.testing.assert_allclose(out[:, 2], swapped[:, 2], atol=1e-12)


class TestComplementThirdColumn:
    def test_third_column_is_unit_norm_and_orthogonal(self) -> None:
        W = complement_third_column(_canonical_he())
        np.testing.assert_allclose(np.linalg.norm(W[:, 2]), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(W[:, 2], W[:, 0]), 0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(W[:, 2], W[:, 1]), 0.0, atol=1e-12)
        assert W.shape == (3, 3)

    def test_collinear_inputs_raise(self) -> None:
        W = np.column_stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["hematoxylin"]])
        with pytest.raises(StainFittingError, match="collinear"):
            complement_third_column(W)


class TestValidateStainMatrix:
    def test_canonical_passes(self) -> None:
        validate_stain_matrix(_canonical_full())

    def test_rotated_h_column_raises(self) -> None:
        W = _canonical_full().copy()
        # Far-from-H direction (cosine ~0.29 with hematoxylin, ~73 degrees).
        W[:, 0] = np.array([0.0, 0.0, 1.0])
        with pytest.raises(StainFittingError, match="hematoxylin"):
            validate_stain_matrix(W, image_key="bad_slide")

    def test_image_key_attached(self) -> None:
        W = _canonical_full().copy()
        # Far-from-H direction (cosine ~0.29 with hematoxylin, ~73 degrees).
        W[:, 0] = np.array([0.0, 0.0, 1.0])
        with pytest.raises(StainFittingError) as excinfo:
            validate_stain_matrix(W, image_key="bad_slide")
        assert excinfo.value.image_key == "bad_slide"
        assert "[bad_slide]" in str(excinfo.value)

    def test_rank_deficient_raises(self) -> None:
        # Replace E with H so the matrix has only one independent stain
        # direction in the H/E plane, then put a tiny third column.
        W = np.column_stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["hematoxylin"], np.array([1e-12, 0.0, 0.0])])
        with pytest.raises(StainFittingError, match="near-zero column|rank-deficient"):
            validate_stain_matrix(W)

    def test_zero_column_raises(self) -> None:
        W = _canonical_full().copy()
        W[:, 2] = 0.0
        with pytest.raises(StainFittingError, match="near-zero column"):
            validate_stain_matrix(W)

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
            validate_stain_matrix(np.zeros((3, 2)))
