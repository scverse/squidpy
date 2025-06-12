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


def wrap_numba_check(x, y, inner_function, check_threads=True):
    if check_threads:
        assert numba.get_num_threads() == 1
    return inner_function(x, y)


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


def mock_runner(x, y, queue, function):
    for i, xi in enumerate(x):
        x[i] = function(xi, y, check_threads=True)
        if queue is not None:
            queue.put(Signal.UPDATE)
    if queue is not None:
        queue.put(Signal.FINISH)
    return x


@pytest.fixture(params=["numba_parallel", "numba_serial", "dask", "vanilla"])
def func(request) -> Callable:
    return {
        "numba_parallel": partial(wrap_numba_check, inner_function=numba_parallel_func),
        "numba_serial": partial(wrap_numba_check, inner_function=numba_serial_func),
        "dask": partial(wrap_numba_check, inner_function=dask_func),
        "vanilla": partial(wrap_numba_check, inner_function=vanilla_func),
    }[request.param]


@pytest.mark.timeout(60)
@pytest.mark.parametrize("n_jobs", [1, 2, 8])
def test_parallelize_loky(func, n_jobs):
    seed = 42
    rng = np.random.RandomState(seed)
    n = 8
    arr1 = [rng.randint(0, 100, n) for _ in range(n)]
    arr2 = np.arange(n)
    runner = partial(mock_runner, function=func)
    # this is the expected result of the function
    expected = np.vstack([func(a1, arr2, check_threads=False) for a1 in arr1])
    # this will be set to something other than 1,2,8
    # we want to check if setting the threads works
    # then after the function is run if the numba cores are set back to 1
    old_num_threads = 3
    numba.set_num_threads(old_num_threads)

    p_func = parallelize(runner, arr1, n_jobs=n_jobs, backend="loky", use_ixs=False, extractor=np.vstack)
    result = p_func(arr2)

    final_numba_threads = numba.get_num_threads()

    assert final_numba_threads == old_num_threads, "Numba threads should not change"
    assert len(result) == len(expected), f"Expected: {expected} but got {result}. Length mismatch"
    for i in range(len(arr1)):
        assert np.all(result[i] == expected[i]), f"Expected {expected[i]} but got {result[i]}"
