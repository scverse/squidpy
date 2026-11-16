"""Tests for the shared ``n_jobs`` resolution semantics."""

from __future__ import annotations

import numba
import pytest  # type: ignore[import]

from squidpy import _utils
from squidpy._utils import _cpu_count, get_n_numba_threads, get_n_processes, thread_map

MAX_THREADS = numba.config.NUMBA_NUM_THREADS
MAX_CORES = _cpu_count()


def test_minus_one_uses_the_default():
    assert get_n_processes(-1) == MAX_CORES
    assert get_n_numba_threads(-1) == MAX_THREADS


def test_none_is_serial_for_processes_and_the_numba_default_for_threads():
    assert get_n_processes(None) == 1
    assert get_n_numba_threads(None) == MAX_THREADS


# scanpy only supports `n_jobs >= -1`, so the countdown convention (`-2` == all but one) is
# not silently reinterpreted -- it is rejected.
@pytest.mark.parametrize("n_jobs", [0, -2, -3, -100])
@pytest.mark.parametrize("resolve", [get_n_processes, get_n_numba_threads])
def test_zero_and_below_minus_one_raise(resolve, n_jobs: int):
    with pytest.raises(ValueError, match=r"must be `-1` or a positive integer"):
        resolve(n_jobs)


@pytest.mark.parametrize(
    ("resolve", "maximum"),
    [(get_n_processes, MAX_CORES), (get_n_numba_threads, MAX_THREADS)],
)
def test_too_many_warns_and_falls_back(resolve, maximum: int, monkeypatch):
    messages: list[str] = []
    monkeypatch.setattr(_utils.logg, "warning", messages.append)

    assert resolve(maximum + 1) == maximum
    assert len(messages) == 1
    assert f"n_jobs={maximum + 1}" in messages[0]


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_thread_map_expects_a_resolved_count(n_jobs: int):
    assert thread_map(lambda x: x * 2, [1, 2, 3], n_jobs=n_jobs) == [2, 4, 6]
