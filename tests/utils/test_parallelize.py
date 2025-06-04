"""Tests for verifying process/thread usage in parallelized functions."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import dask.array as da
import numba
import numpy as np
import pytest  # type: ignore[import]

from squidpy._utils import Signal, parallelize

# Functions to be parallelized


@numba.njit(parallel=True)
def numba_parallel_func(x, y) -> np.ndarray:
    return x * 2 + y


@numba.njit(parallel=False)
def numba_serial_func(x, y) -> np.ndarray:
    return x * 2 + y


def dask_func(x, y) -> np.ndarray:
    return (da.from_array(x) * 2 + y).compute()


def vanilla_func(x, y) -> np.ndarray:
    return x * 2 + y


# Mock runner function


def mock_runner(x, y, queue, func):
    for i in range(len(x)):
        print(len(x[i]), len(y))
        x[i] = func(x[i], y)
        if queue is not None:
            queue.put(Signal.UPDATE)
    if queue is not None:
        queue.put(Signal.FINISH)
    return x


@pytest.fixture(params=["numba_parallel", "numba_serial", "dask", "vanilla"])
def func(request) -> Callable:
    return {
        "numba_parallel": numba_parallel_func,
        "numba_serial": numba_serial_func,
        "dask": dask_func,
        "vanilla": vanilla_func,
    }[request.param]


@pytest.mark.parametrize("n_jobs", [1, 2, 8])
def test_parallelize_loky(func, n_jobs):
    n = 8
    arr1 = [np.arange(n) for _ in range(n)]
    arr2 = np.arange(n)
    runner = partial(mock_runner, func=func)
    expected = [func(arr1[i], arr2) for i in range(len(arr1))]
    p_func = parallelize(runner, arr1, n_jobs=n_jobs, backend="loky", use_ixs=False)
    result = p_func(arr2)[0]
    assert len(result) == len(expected), f"Expected: {expected} but got {result}. Length mismatch"
    for i in range(len(arr1)):
        assert np.all(result[i] == expected[i]), f"Expected {expected[i]} but got {result[i]}"
