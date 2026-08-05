"""Tests for the shared ``n_jobs`` resolution semantics."""

from __future__ import annotations

import numba
import pytest  # type: ignore[import]

from squidpy import _utils
from squidpy._utils import get_n_processes, get_n_threads, thread_map

MAX_THREADS = numba.config.NUMBA_NUM_THREADS


@pytest.mark.parametrize("n_jobs", [-1, -2, -100])
def test_negative_uses_the_numba_default(n_jobs: int):
    assert get_n_processes(n_jobs) == MAX_THREADS
    assert get_n_threads(n_jobs) == MAX_THREADS


def test_none_is_serial_for_processes_and_the_numba_default_for_threads():
    assert get_n_processes(None) == 1
    assert get_n_threads(None) == MAX_THREADS


@pytest.mark.parametrize("resolve", [get_n_processes, get_n_threads])
def test_zero_raises(resolve):
    with pytest.raises(ValueError, match=r"cannot be `0`"):
        resolve(0)


@pytest.mark.parametrize("resolve", [get_n_processes, get_n_threads])
def test_too_many_warns_and_falls_back(resolve, monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(_utils.logg, "warning", messages.append)

    assert resolve(MAX_THREADS + 1) == MAX_THREADS
    assert len(messages) == 1
    assert f"n_jobs={MAX_THREADS + 1}" in messages[0]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_thread_map_expects_a_resolved_count(n_jobs: int):
    assert thread_map(lambda x: x * 2, [1, 2, 3], n_jobs=n_jobs) == [2, 4, 6]
