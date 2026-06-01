"""Argument validation for the public ``align`` / ``align_by_landmarks`` functions."""

from __future__ import annotations

from squidpy._validators import assert_one_of

OUTPUT_MODES = ("object", "copy", "inplace")
ON_VALUES = ("obs", "image")


def validate_output_mode(value: str) -> None:
    assert_one_of(value, OUTPUT_MODES, name="output_mode")


def validate_on(value: str) -> None:
    assert_one_of(value, ON_VALUES, name="on")
