"""Benchmarks for squidpy.gr.co_occurrence function.

Run benchmarks with:
    pytest benchmarks/ --benchmark-only -v

Compare against baseline:
    pytest benchmarks/ --benchmark-only --benchmark-compare

Save benchmark results:
    pytest benchmarks/ --benchmark-only --benchmark-autosave

Multithreading Behavior
-----------------------
pytest-benchmark runs each benchmark function sequentially (one at a time).
Within each benchmark iteration, the function under test (e.g., co_occurrence)
can use multiple threads/processes as configured by its parameters (n_jobs).

This means:
- Benchmarks do NOT run in parallel with each other
- Each benchmark has full access to system resources
- Functions using numba @njit(parallel=True) will use multiple threads
- Functions using joblib/loky parallelization will spawn worker processes

To benchmark single-threaded vs multi-threaded performance, create separate
test cases with different n_jobs values, or use pytest parametrization.

Note: pytest-xdist (-n auto) parallelizes test COLLECTION, not benchmark
execution. For accurate benchmarks, avoid -n with benchmark tests.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from squidpy.gr import co_occurrence

if TYPE_CHECKING:
    from collections.abc import Callable

    from anndata import AnnData


class TestCoOccurrenceBenchmarks:
    """Benchmark suite for co_occurrence function scaling with dataset size."""

    @pytest.mark.benchmark(group="co_occurrence_scaling")
    def test_co_occurrence_scaling(
        self,
        benchmark: "pytest.benchmark.fixture.BenchmarkFixture",
        adata_scaling: "AnnData",
    ) -> None:
        """
        Benchmark co_occurrence across different dataset sizes.

        The adata_scaling fixture is parameterized with sizes defined in
        conftest.BENCHMARK_SIZES. Modify that dict to change scale points.
        """
        benchmark.extra_info["n_obs"] = adata_scaling.n_obs
        benchmark(co_occurrence, adata_scaling, cluster_key="cluster", copy=True)


class TestCoOccurrenceIntervalBenchmarks:
    """Benchmark suite for co_occurrence with different interval parameters."""

    @pytest.mark.benchmark(group="co_occurrence_intervals")
    @pytest.mark.parametrize("n_intervals", [10, 25, 50, 100])
    def test_co_occurrence_intervals(
        self,
        benchmark: "pytest.benchmark.fixture.BenchmarkFixture",
        adata_default: "AnnData",
        n_intervals: int,
    ) -> None:
        """Benchmark co_occurrence with different interval counts."""
        benchmark.extra_info["n_intervals"] = n_intervals
        benchmark.extra_info["n_obs"] = adata_default.n_obs
        benchmark(
            co_occurrence,
            adata_default,
            cluster_key="cluster",
            interval=n_intervals,
            copy=True,
        )


class TestCoOccurrenceNumbaCompilation:
    """Benchmark numba compilation overhead."""

    @pytest.mark.benchmark(group="co_occurrence_warmup")
    def test_co_occurrence_first_run(
        self,
        benchmark: "pytest.benchmark.fixture.BenchmarkFixture",
        make_adata: "Callable[[int, int], AnnData]",
    ) -> None:
        """
        Benchmark first run to capture numba compilation overhead.

        Note: This test measures cold-start performance including JIT compilation.
        Run with --benchmark-warmup=off to capture compilation time.
        """
        # Create fresh adata each time to avoid caching effects
        def run_co_occurrence() -> None:
            adata = make_adata(100)
            co_occurrence(adata, cluster_key="cluster", copy=True)

        benchmark.pedantic(run_co_occurrence, warmup_rounds=0, rounds=3)


# Parametrized benchmark for comprehensive scaling analysis
@pytest.mark.benchmark(group="co_occurrence_comprehensive")
@pytest.mark.parametrize(
    "n_obs,n_clusters,n_intervals",
    [
        (1_000, 5, 25),
        (1_000, 10, 50),
        (5_000, 10, 50),
        (10_000, 10, 50),
        (10_000, 20, 50),
        (50_000, 10, 50),
    ],
)
def test_co_occurrence_comprehensive(
    benchmark: "pytest.benchmark.fixture.BenchmarkFixture",
    make_adata: "Callable[[int, int], AnnData]",
    n_obs: int,
    n_clusters: int,
    n_intervals: int,
) -> None:
    """
    Comprehensive parametrized benchmark for co_occurrence.

    Tests various combinations of dataset size, cluster count, and intervals
    to understand scaling behavior across multiple dimensions.
    """
    adata = make_adata(n_obs, n_clusters=n_clusters)
    benchmark.extra_info.update(
        {
            "n_obs": n_obs,
            "n_clusters": n_clusters,
            "n_intervals": n_intervals,
        }
    )
    benchmark(
        co_occurrence,
        adata,
        cluster_key="cluster",
        interval=n_intervals,
        copy=True,
    )
