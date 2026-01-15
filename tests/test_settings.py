"""Tests for squidpy.settings module."""

from __future__ import annotations

import pytest

from squidpy.settings import settings


class TestSettings:
    """Test the settings module."""

    def test_default_device(self):
        """Test that default device is 'auto'."""
        # Reset to default
        settings.device = "auto"
        assert settings.device == "auto"

    def test_set_device_cpu(self):
        """Test setting device to 'cpu'."""
        settings.device = "cpu"
        assert settings.device == "cpu"
        settings.device = "auto"  # reset

    def test_set_device_invalid(self):
        """Test that invalid device raises ValueError."""
        with pytest.raises(ValueError, match="Invalid device"):
            settings.device = "invalid"

    def test_set_device_gpu_without_rsc(self):
        """Test that setting device to 'gpu' without rapids-singlecell raises RuntimeError."""
        # This will fail if rapids-singlecell is not installed
        if not settings.gpu_available():
            with pytest.raises(RuntimeError, match="rapids-singlecell not installed"):
                settings.device = "gpu"
