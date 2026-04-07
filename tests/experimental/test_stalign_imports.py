from __future__ import annotations

import pytest

import squidpy.experimental.tl as tl


def test_stalign_import_reports_missing_jax(monkeypatch: pytest.MonkeyPatch):
    def _raise_missing_jax(module_name: str):
        raise ModuleNotFoundError("No module named 'jax'", name="jax")

    monkeypatch.setattr(tl, "import_module", _raise_missing_jax)

    with pytest.raises(ImportError, match=r"Install it with `pip install \"squidpy\[jax\]\"`"):
        tl.__getattr__("stalign")
