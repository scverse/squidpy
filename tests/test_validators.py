"""Tests for squidpy._validators."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from squidpy._validators import (
    _assert_in_range,
    _assert_isinstance,
    _assert_key_in_adata,
    _assert_key_in_sdata,
    _assert_non_empty_sequence,
    _assert_non_negative,
    _assert_one_of,
    _assert_positive,
    _check_tuple_needles,
    _get_valid_values,
)


# ---------------------------------------------------------------------------
# _assert_positive
# ---------------------------------------------------------------------------
class TestAssertPositive:
    def test_positive_value(self):
        _assert_positive(1.0, name="x")
        _assert_positive(0.001, name="x")

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _assert_positive(0, name="x")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            _assert_positive(-1, name="x")


# ---------------------------------------------------------------------------
# _assert_non_negative
# ---------------------------------------------------------------------------
class TestAssertNonNegative:
    def test_non_negative_value(self):
        _assert_non_negative(0, name="x")
        _assert_non_negative(1, name="x")

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            _assert_non_negative(-0.1, name="x")


# ---------------------------------------------------------------------------
# _assert_in_range
# ---------------------------------------------------------------------------
class TestAssertInRange:
    def test_in_range(self):
        _assert_in_range(0.5, 0, 1, name="x")
        _assert_in_range(0, 0, 1, name="x")
        _assert_in_range(1, 0, 1, name="x")

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="interval"):
            _assert_in_range(1.1, 0, 1, name="x")
        with pytest.raises(ValueError, match="interval"):
            _assert_in_range(-0.1, 0, 1, name="x")


# ---------------------------------------------------------------------------
# _assert_non_empty_sequence
# ---------------------------------------------------------------------------
class TestAssertNonEmptySequence:
    def test_list(self):
        assert _assert_non_empty_sequence(["a", "b"], name="items") == ["a", "b"]

    def test_scalar_conversion(self):
        assert _assert_non_empty_sequence("a", name="items") == ["a"]

    def test_no_scalar_conversion(self):
        with pytest.raises(TypeError, match="sequence"):
            _assert_non_empty_sequence(42, name="items", convert_scalar=False)

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="No items"):
            _assert_non_empty_sequence([], name="items")


# ---------------------------------------------------------------------------
# _get_valid_values
# ---------------------------------------------------------------------------
class TestGetValidValues:
    def test_valid(self):
        assert _get_valid_values(["a", "b"], ["a", "b", "c"]) == ["a", "b"]

    def test_partial(self):
        assert _get_valid_values(["a", "z"], ["a", "b"]) == ["a"]

    def test_none_valid(self):
        with pytest.raises(ValueError, match="No valid values"):
            _get_valid_values(["z"], ["a", "b"])


# ---------------------------------------------------------------------------
# _check_tuple_needles
# ---------------------------------------------------------------------------
class TestCheckTupleNeedles:
    def test_valid_needles(self):
        result = _check_tuple_needles([("a", "b")], ["a", "b", "c"], "Value `{}` not found.")
        assert result == [("a", "b")]

    def test_invalid_needle_reraise(self):
        with pytest.raises(ValueError, match="z"):
            _check_tuple_needles([("z", "b")], ["a", "b"], "Value `{}` not found.")

    def test_invalid_needle_no_reraise(self):
        result = _check_tuple_needles([("z", "b")], ["a", "b"], "Value `{}` not found.", reraise=False)
        assert result == []

    def test_wrong_length(self):
        with pytest.raises(ValueError, match="length"):
            _check_tuple_needles([("a",)], ["a"], "msg {}")

    def test_not_sequence(self):
        with pytest.raises(TypeError, match="Sequence"):
            _check_tuple_needles([42], ["a"], "msg {}")


# ---------------------------------------------------------------------------
# _assert_isinstance
# ---------------------------------------------------------------------------
class TestAssertIsinstance:
    def test_correct_type(self):
        _assert_isinstance("hello", str, name="x")
        _assert_isinstance(42, int, name="x")

    def test_tuple_of_types(self):
        _assert_isinstance("hello", (str, int), name="x")
        _assert_isinstance(42, (str, int), name="x")

    def test_wrong_type(self):
        with pytest.raises(TypeError, match="str"):
            _assert_isinstance(42, str, name="x")

    def test_wrong_type_tuple(self):
        with pytest.raises(TypeError, match="str or int"):
            _assert_isinstance(3.14, (str, int), name="x")


# ---------------------------------------------------------------------------
# _assert_one_of
# ---------------------------------------------------------------------------
class TestAssertOneOf:
    def test_valid(self):
        _assert_one_of("a", ["a", "b", "c"], name="x")

    def test_invalid(self):
        with pytest.raises(ValueError, match="one of"):
            _assert_one_of("z", ["a", "b"], name="x")


# ---------------------------------------------------------------------------
# _assert_key_in_adata
# ---------------------------------------------------------------------------
class TestAssertKeyInAdata:
    def test_key_present(self):
        adata = MagicMock()
        adata.obs = {"cell_type": [1, 2, 3]}
        _assert_key_in_adata(adata, "cell_type", attr="obs")

    def test_key_missing(self):
        adata = MagicMock()
        adata.obs = {"cell_type": [1, 2, 3]}
        with pytest.raises(KeyError, match="missing_key"):
            _assert_key_in_adata(adata, "missing_key", attr="obs")

    def test_extra_msg(self):
        adata = MagicMock()
        adata.obs = {}
        with pytest.raises(KeyError, match="Run this first"):
            _assert_key_in_adata(adata, "key", attr="obs", extra_msg="Run this first.")

    def test_lists_available_keys(self):
        adata = MagicMock()
        adata.obs = {"a": 1, "b": 2}
        with pytest.raises(KeyError, match="Available keys"):
            _assert_key_in_adata(adata, "missing", attr="obs")

    def test_container_without_keys_method(self):
        """Fallback to list(container) when .keys() is not available."""
        adata = MagicMock()
        adata.obsm = ["X_pca", "X_umap"]  # list has no .keys()
        with pytest.raises(KeyError, match="X_spatial"):
            _assert_key_in_adata(adata, "X_spatial", attr="obsm")


# ---------------------------------------------------------------------------
# _assert_key_in_sdata
# ---------------------------------------------------------------------------
class TestAssertKeyInSdata:
    def test_key_present(self):
        sdata = MagicMock()
        sdata.images = {"image1": "data"}
        _assert_key_in_sdata(sdata, "image1", attr="images")

    def test_key_missing(self):
        sdata = MagicMock()
        sdata.images = {"image1": "data"}
        with pytest.raises(KeyError, match="missing"):
            _assert_key_in_sdata(sdata, "missing", attr="images")

    def test_extra_msg(self):
        sdata = MagicMock()
        sdata.labels = {}
        with pytest.raises(KeyError, match="Please provide"):
            _assert_key_in_sdata(sdata, "mask", attr="labels", extra_msg="Please provide a mask.")

    def test_lists_available_keys(self):
        sdata = MagicMock()
        sdata.images = {"img1": "data", "img2": "data"}
        with pytest.raises(KeyError, match="Available keys"):
            _assert_key_in_sdata(sdata, "missing", attr="images")


# ---------------------------------------------------------------------------
# _assert_isinstance edge cases
# ---------------------------------------------------------------------------
class TestAssertIsinstanceEdgeCases:
    def test_bool_is_subclass_of_int(self):
        """bool is a subclass of int — _assert_isinstance(True, int) passes."""
        _assert_isinstance(True, int, name="x")

    def test_none_type(self):
        with pytest.raises(TypeError, match="str"):
            _assert_isinstance(None, str, name="x")


# ---------------------------------------------------------------------------
# Re-export smoke test
# ---------------------------------------------------------------------------
class TestReExports:
    def test_gr_utils_reexports(self):
        from squidpy.gr._utils import (
            _assert_in_range,
            _assert_non_empty_sequence,
            _assert_non_negative,
            _assert_positive,
            _check_tuple_needles,
            _get_valid_values,
        )

        # Just verify they are the same objects
        from squidpy._validators import _assert_positive as _ap

        assert _assert_positive is _ap
