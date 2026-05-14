from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from squidpy.experimental.im._stain._constants import (
    RUIFROK_HE,
    STAIN_REFERENCE_SCHEMA_VERSION,
)
from squidpy.experimental.im._stain._reference import StainReference


def _ruifrok_matrix() -> np.ndarray:
    third = np.cross(RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"])
    third = third / np.linalg.norm(third)
    return np.column_stack([RUIFROK_HE["hematoxylin"], RUIFROK_HE["eosin"], third])


class TestConstruction:
    def test_macenko_basic(self) -> None:
        ref = StainReference(method="macenko", stain_matrix=_ruifrok_matrix())
        assert ref.method == "macenko"
        assert ref.version == STAIN_REFERENCE_SCHEMA_VERSION
        assert ref.mu is None and ref.sigma is None
        assert ref.stain_matrix.shape == (3, 3)

    def test_reinhard_basic(self) -> None:
        ref = StainReference(method="reinhard", mu=np.array([1.0, 0.5, -0.2]), sigma=np.array([0.1, 0.1, 0.1]))
        assert ref.method == "reinhard"
        assert ref.stain_matrix is None

    def test_unknown_method_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            StainReference(method="not-a-method")  # type: ignore[arg-type]

    def test_decomposition_requires_stain_matrix(self) -> None:
        with pytest.raises(ValueError, match="requires stain_matrix"):
            StainReference(method="macenko")

    def test_decomposition_forbids_mu_sigma(self) -> None:
        with pytest.raises(ValueError, match="forbids mu/sigma"):
            StainReference(
                method="macenko",
                stain_matrix=_ruifrok_matrix(),
                mu=np.zeros(3),
                sigma=np.ones(3),
            )

    def test_reinhard_requires_mu_and_sigma(self) -> None:
        with pytest.raises(ValueError, match="requires both mu and sigma"):
            StainReference(method="reinhard", mu=np.zeros(3))

    def test_reinhard_rejects_non_positive_sigma(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            StainReference(method="reinhard", mu=np.zeros(3), sigma=np.array([1.0, 0.0, 1.0]))

    def test_reinhard_forbids_stain_matrix(self) -> None:
        with pytest.raises(ValueError, match="forbids stain_matrix"):
            StainReference(
                method="reinhard",
                mu=np.zeros(3),
                sigma=np.ones(3),
                stain_matrix=_ruifrok_matrix(),
            )

    def test_bad_background_intensity(self) -> None:
        with pytest.raises(ValueError, match="background_intensity"):
            StainReference(
                method="macenko",
                stain_matrix=_ruifrok_matrix(),
                background_intensity=np.array([255.0, -1.0, 255.0]),
            )

    def test_bad_max_concentrations_shape(self) -> None:
        with pytest.raises(ValueError, match="max_concentrations"):
            StainReference(
                method="macenko",
                stain_matrix=_ruifrok_matrix(),
                max_concentrations=np.array([1.0, 2.0, 3.0]),
            )


class TestPersistence:
    def test_round_trip_macenko(self, tmp_path: Path) -> None:
        ref = StainReference(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            max_concentrations=np.array([2.5, 2.0]),
            fit_metadata={"seed": 0, "pyramid_level": "scale1"},
        )
        path = tmp_path / "ref.json"
        ref.save(path)
        loaded = StainReference.load(path)
        np.testing.assert_array_equal(loaded.stain_matrix, ref.stain_matrix)
        np.testing.assert_array_equal(loaded.max_concentrations, ref.max_concentrations)
        np.testing.assert_array_equal(loaded.background_intensity, ref.background_intensity)
        assert loaded.method == "macenko"
        assert loaded.fit_metadata == ref.fit_metadata
        assert loaded.version == STAIN_REFERENCE_SCHEMA_VERSION

    def test_round_trip_vahadane(self, tmp_path: Path) -> None:
        ref = StainReference(method="vahadane", stain_matrix=_ruifrok_matrix())
        path = tmp_path / "ref.json"
        ref.save(path)
        loaded = StainReference.load(path)
        assert loaded.method == "vahadane"
        np.testing.assert_array_equal(loaded.stain_matrix, ref.stain_matrix)

    def test_round_trip_reinhard(self, tmp_path: Path) -> None:
        ref = StainReference(
            method="reinhard",
            mu=np.array([1.2, -0.3, 0.05]),
            sigma=np.array([0.4, 0.3, 0.2]),
        )
        path = tmp_path / "ref.json"
        ref.save(path)
        loaded = StainReference.load(path)
        np.testing.assert_array_equal(loaded.mu, ref.mu)
        np.testing.assert_array_equal(loaded.sigma, ref.sigma)
        assert loaded.stain_matrix is None

    def test_round_trip_cohort_fields(self, tmp_path: Path) -> None:
        wiggled = _ruifrok_matrix() + 0.01
        wiggled /= np.linalg.norm(wiggled, axis=0)
        per_image = {
            "img_a": {"stain_matrix": _ruifrok_matrix()},
            "img_b": {"stain_matrix": wiggled},
        }
        ref = StainReference(
            method="macenko",
            stain_matrix=_ruifrok_matrix(),
            cohort_members=("img_a", "img_b"),
            per_image_stats=per_image,
        )
        path = tmp_path / "ref.json"
        ref.save(path)
        loaded = StainReference.load(path)
        assert loaded.cohort_members == ("img_a", "img_b")
        assert isinstance(loaded.cohort_members, tuple)
        np.testing.assert_array_equal(
            loaded.per_image_stats["img_b"]["stain_matrix"],
            per_image["img_b"]["stain_matrix"],
        )

    def test_corrupted_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json at all", encoding="utf-8")
        with pytest.raises(ValueError, match="Could not parse"):
            StainReference.load(path)

    def test_missing_schema_marker_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"method": "macenko"}', encoding="utf-8")
        with pytest.raises(ValueError, match="schema marker"):
            StainReference.load(path)

    def test_newer_schema_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "future.json"
        path.write_text(
            '{"schema": "squidpy.stain_reference", "version": 999, "method": "macenko"}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="newer than this squidpy"):
            StainReference.load(path)
