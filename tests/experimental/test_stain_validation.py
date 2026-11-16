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


def _he_matrix() -> np.ndarray:
    return np.stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"]], axis=1)


class TestReorderToCanonical:
    def test_swapped_columns_restored(self) -> None:
        he = _he_matrix()
        swapped = he[:, ::-1]
        out = reorder_to_canonical(swapped)
        np.testing.assert_allclose(out, he, atol=1e-8)

    def test_sign_flips_corrected(self) -> None:
        he = _he_matrix()
        flipped = he * np.array([-1.0, 1.0])
        out = reorder_to_canonical(flipped)
        np.testing.assert_allclose(out, he, atol=1e-8)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            reorder_to_canonical(np.eye(3))


class TestComplementThirdColumn:
    def test_unit_orthogonal(self) -> None:
        w = complement_third_column(_he_matrix())
        assert w.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.norm(w[:, 2]), 1.0, atol=1e-8)
        assert abs(w[:, 2] @ w[:, 0]) < 1e-8
        assert abs(w[:, 2] @ w[:, 1]) < 1e-8

    def test_colinear_raises(self) -> None:
        v = RUIFROK_HE["hematoxylin"]
        with pytest.raises(StainFittingError, match="colinear"):
            complement_third_column(np.stack([v, v], axis=1))


class TestValidateStainMatrix:
    def test_canonical_passes(self) -> None:
        validate_stain_matrix(complement_third_column(_he_matrix()))

    def test_rank_deficient_raises_with_image_key(self) -> None:
        w = complement_third_column(_he_matrix())
        w[:, 1] = w[:, 0]  # collapse E onto H
        with pytest.raises(StainFittingError) as exc:
            validate_stain_matrix(w, image_key="slideA")
        assert exc.value.image_key == "slideA"

    def test_rotated_he_raises(self) -> None:
        # rotate H far toward an unrelated direction
        w = complement_third_column(_he_matrix())
        w[:, 0] = np.array([1.0, 0.0, 0.0])
        with pytest.raises(StainFittingError, match="deviates"):
            validate_stain_matrix(w)

    def test_zero_column_raises(self) -> None:
        w = complement_third_column(_he_matrix())
        w[:, 0] = 0.0
        with pytest.raises(StainFittingError, match="zero-norm"):
            validate_stain_matrix(w)

    def test_bad_shape_raises(self) -> None:
        with pytest.raises(StainFittingError, match="shape"):
            validate_stain_matrix(np.eye(2))
