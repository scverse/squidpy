"""Tests for verifying process/thread usage in parallelized functions."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import partial

import dask.array as da
import numba
import numpy as np
import psutil
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


@pytest.mark.timeout(60)
@pytest.mark.parametrize("n_jobs", [1, 2, 8])
def test_parallelize_loky(func, n_jobs):
    start_time = time.time()
    seed = 42
    rng = np.random.RandomState(seed)
    n = 8
    arr1 = [rng.randint(0, 100, n) for _ in range(n)]
    arr2 = np.arange(n)
    runner = partial(mock_runner, func=func)
    # this is the expected result of the function
    expected = [func(arr1[i], arr2) for i in range(len(arr1))]
    # this will be set to something other than 1,2,8
    # we want to check if setting the threads works
    # then after the function is run if the numba cores are set back to 1
    old_num_threads = 3
    numba.set_num_threads(old_num_threads)
    # Get initial state
    initial_process = psutil.Process()
    initial_children = {p.pid for p in initial_process.children(recursive=True)}
    initial_children = {psutil.Process(pid) for pid in initial_children}
    init_numba_threads = numba.get_num_threads()

    p_func = parallelize(runner, arr1, n_jobs=n_jobs, backend="loky", use_ixs=False, n_split=1)
    result = p_func(arr2)[0]

    final_children = {p.pid for p in initial_process.children(recursive=True)}
    final_numba_threads = numba.get_num_threads()

    assert init_numba_threads == old_num_threads, "Numba threads should not change"
    assert final_numba_threads == 1, "Numba threads should be 1"
    assert len(result) == len(expected), f"Expected: {expected} but got {result}. Length mismatch"
    for i in range(len(arr1)):
        assert np.all(result[i] == expected[i]), f"Expected {expected[i]} but got {result[i]}"

    processes = final_children - initial_children

    processes = {psutil.Process(pid) for pid in processes}
    processes = {p for p in processes if not any("resource_tracker" in cl for cl in p.cmdline())}
    if n_jobs > 1:  # expect exactly n_jobs
        assert len(processes) == n_jobs, f"Unexpected processes created or not created: {processes}"
    else:  # some functions use the main process others use a new process
        processes = {p for p in processes if p.create_time() > start_time}
        assert len(processes) <= 1, f"Unexpected processes created or not created: {processes}"
