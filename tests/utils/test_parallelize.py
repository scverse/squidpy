"""Tests for verifying process/thread usage in parallelized functions."""

from __future__ import annotations

import os
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


# Timeouts are also useful because some processes don't return in
# in case of failure.


@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(
            "threading",
            marks=pytest.mark.skipif(
                os.environ.get("CI") == "true", reason="Only testing 'loky' backend in CI environment"
            ),
        ),
        pytest.param(
            "multiprocessing",
            marks=pytest.mark.skipif(
                os.environ.get("CI") == "true", reason="Only testing 'loky' backend in CI environment"
            ),
        ),
        "loky",
    ],
)
def test_parallelize(func, backend):
    seed = 42
    n = 2
    n_jobs = 2
    rng = np.random.RandomState(seed)
    arr1 = [rng.randint(0, 100, n) for _ in range(n)]
    arr2 = np.arange(n)
    runner = partial(mock_runner, function=func)

    init_threads = numba.get_num_threads()
    expected = np.vstack([func(a1, arr2, check_threads=False) for a1 in arr1])

    p_func = parallelize(
        runner, arr1, n_jobs=n_jobs, backend=backend, use_ixs=False, extractor=np.vstack, show_progress=False
    )
    result = p_func(arr2)

    assert numba.get_num_threads() == init_threads, "Number of threads should stay the same after parallelization"
    assert np.allclose(result, expected), f"Expected: {expected} but got {result}"
