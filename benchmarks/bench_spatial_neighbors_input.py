"""Measure issue #1199 without changing Squidpy's implementation.

Run with a Python environment satisfying this checkout's runtime dependencies.
The default cases use 10k/100k polygons and fixed density, radius and complexity.
Each case runs in a fresh process. Imports, warmup and fixture preparation are
excluded from timings; the first sample is a first graph call, not cold startup.
Both inputs already contain identical, SpatialData-derived coordinates.
Memory samples run separately from timing/validation, with no reference graph.
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import gc
import importlib.metadata
import json
import os
import platform
import pstats
import statistics
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch


def fixture(n_cells: int, n_regions: int, seed: int, vertices: int):
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    import shapely
    from anndata import AnnData
    from spatialdata import SpatialData
    from spatialdata.models import ShapesModel, TableModel
    from spatialdata.transformations import Identity

    rng = np.random.default_rng(seed)
    regions = [f"region_{i:03d}" for i in range(n_regions)]
    shapes, observations = {}, []
    angles = np.linspace(0, 2 * np.pi, vertices, endpoint=False)
    offsets = 0.25 * np.column_stack((np.cos(angles), np.sin(angles)))
    for i, region in enumerate(regions):
        n = n_cells // n_regions + (i < n_cells % n_regions)
        # Area grows with n, keeping density at one cell per square unit.
        centers = rng.uniform(0, np.sqrt(n), size=(n, 2))
        ids = np.arange(1, n + 1)
        polygons = shapely.polygons(centers[:, None, :] + offsets[None, :, :])
        shapes[region] = ShapesModel.parse(
            gpd.GeoDataFrame(geometry=polygons, index=ids),
            transformations={"global": Identity()},
        )
        observations.append(
            pd.DataFrame(
                {"region": region, "instance_id": ids},
                index=[f"{region}:{j}" for j in ids],
            )
        )
    obs = pd.concat(observations)
    obs["region"] = pd.Categorical(obs["region"], categories=regions)
    table = TableModel.parse(AnnData(obs=obs), region=regions, region_key="region", instance_key="instance_id")
    return SpatialData(shapes=shapes, tables={"table": table}), dict.fromkeys(regions, "global")


def inputs(args):
    from anndata import AnnData

    from squidpy.gr._build import _resolve_data

    sdata, mapping = fixture(args.cells, args.regions, args.seed, args.vertices)
    start = time.perf_counter()
    resolved, library_key = _resolve_data(sdata, table_key="table", elements_to_coordinate_systems=mapping, copy=True)
    coordinate_preparation_s = time.perf_counter() - start
    assert library_key == "region"
    coords = resolved.obsm["spatial"]
    # Existing coordinates deliberately do not stop today's SpatialData resolver.
    sdata.tables["table"].obsm["spatial"] = coords.copy()
    adata = AnnData(obs=resolved.obs.copy(), obsm={"spatial": coords.copy()})
    return sdata, adata, mapping, coordinate_preparation_s


def graph(data, mapping, radius):
    from spatialdata import SpatialData

    from squidpy.gr import spatial_neighbors_radius

    kwargs = {"radius": radius, "copy": True, "n_jobs": 1, "library_key": "region"}
    if isinstance(data, SpatialData):
        kwargs.update(table_key="table", elements_to_coordinate_systems=mapping)
    return spatial_neighbors_radius(data, **kwargs)


def warmup(args):
    from anndata import AnnData

    from squidpy.gr._build import _resolve_data

    small, mapping = fixture(200, 1, args.seed, args.vertices)
    resolved, _ = _resolve_data(small, table_key="table", elements_to_coordinate_systems=mapping, copy=True)
    adata = AnnData(obs=resolved.obs.copy(), obsm={"spatial": resolved.obsm["spatial"]})
    graph(small, mapping, args.radius)
    graph(adata, mapping, args.radius)
    gc.collect()


@contextmanager
def instrumentation():
    from squidpy.gr import _build

    values = {"resolve_s": 0.0, "graph_s": 0.0, "centroid_calls": 0}

    def timed(name, function):
        def wrapped(*args, **kwargs):
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                values[name] += time.perf_counter() - start

        return wrapped

    original_centroids = _build.get_centroids

    def centroids(*args, **kwargs):
        values["centroid_calls"] += 1
        return original_centroids(*args, **kwargs)

    with (
        patch.object(_build, "_resolve_data", timed("resolve_s", _build._resolve_data)),
        patch.object(_build, "_run_spatial_neighbors", timed("graph_s", _build._run_spatial_neighbors)),
        patch.object(_build, "get_centroids", centroids),
    ):
        yield values


def assert_graph_equal(actual, reference):
    import numpy as np

    for name in ("connectivities", "distances"):
        a, b = getattr(actual, name).copy(), getattr(reference, name).copy()
        a.sort_indices()
        b.sort_indices()
        np.testing.assert_array_equal(a.indptr, b.indptr)
        np.testing.assert_array_equal(a.indices, b.indices)
        np.testing.assert_allclose(a.data, b.data, rtol=1e-12, atol=1e-12)


def timing(args, sdata, adata, mapping):
    import numpy as np
    import pandas as pd

    # Keep one reference only; these runs are not used for peak-memory reporting.
    reference = graph(adata, mapping, args.radius)
    before_obs = sdata.tables["table"].obs.copy()
    before_coords = sdata.tables["table"].obsm["spatial"].copy()
    samples = []
    order = [("spatialdata_first_graph", sdata, 0)]
    for trial in range(1, args.repeats + 1):
        pair = [("spatialdata_repeat", sdata, trial), ("anndata_coordinates", adata, trial)]
        order.extend(pair if trial % 2 else pair[::-1])
    for kind, data, trial in order:
        gc.collect()
        with instrumentation() as metrics:
            start = time.perf_counter()
            result = graph(data, mapping, args.radius)
            elapsed = time.perf_counter() - start
        assert metrics["centroid_calls"] == (args.regions if data is sdata else 0)
        assert_graph_equal(result, reference)
        samples.append(
            {
                "kind": kind,
                "trial": trial,
                "total_s": elapsed,
                **metrics,
                "matrix_nnz": int(result.connectivities.nnz),
                "nonzero_connections": int(result.connectivities.count_nonzero()),
            }
        )
        del result
    table = sdata.tables["table"]
    pd.testing.assert_frame_equal(table.obs, before_obs)
    np.testing.assert_array_equal(table.obsm["spatial"], before_coords)
    assert table.X is None and not table.obsp
    pd.testing.assert_frame_equal(adata.obs, before_obs)
    np.testing.assert_array_equal(adata.obsm["spatial"], before_coords)
    assert adata.X is None and not adata.obsp
    del reference
    gc.collect()

    # Profile separately, so profiler overhead never enters timing samples.
    from squidpy.gr._build import _resolve_data

    profile = cProfile.Profile()
    profile.runcall(_resolve_data, sdata, table_key="table", elements_to_coordinate_systems=mapping, copy=True)
    profile_path = Path(args.output) / f"profile-{args.cells}-{args.regions}.txt"
    with profile_path.open("w") as stream:
        pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
    profile_path.write_text(profile_path.read_text().rstrip() + "\n")
    return {"samples": samples, "equivalence": "passed", "input_preservation": "passed"}


def memory(args, sdata, adata, mapping):
    import psutil

    gc.collect()
    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.002):
            peak[0] = max(peak[0], process.memory_info().rss)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    try:
        result = graph(sdata if args.worker == "memory_spatialdata" else adata, mapping, args.radius)
        peak[0] = max(peak[0], process.memory_info().rss)
        nnz = int(result.connectivities.nnz)
    finally:
        stop.set()
        sampler.join()
    return {
        "kind": args.worker,
        "baseline_rss_mib": baseline / 2**20,
        "sampled_peak_rss_mib": peak[0] / 2**20,
        "sampled_increment_mib": (peak[0] - baseline) / 2**20,
        "sampling_interval_ms": 2,
        "matrix_nnz": nnz,
    }


def worker(args):
    import scanpy as sc

    import squidpy

    expected = Path(__file__).resolve().parents[1] / "src/squidpy/__init__.py"
    assert Path(squidpy.__file__).resolve() == expected
    sc.settings.verbosity = 0
    warmup(args)
    sdata, adata, mapping, prep_s = inputs(args)
    measured = timing(args, sdata, adata, mapping) if args.worker == "timing" else memory(args, sdata, adata, mapping)
    packages = ["spatialdata", "anndata", "numpy", "pandas", "scipy", "scikit-learn", "shapely", "dask"]
    return {
        "cells": args.cells,
        "regions": args.regions,
        "coordinate_preparation_s": prep_s,
        "python": platform.python_version(),
        "source_path": "src/squidpy/__init__.py",
        "packages": {name: importlib.metadata.version(name) for name in packages},
        **measured,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", default=["10000:1", "100000:1", "100000:20"])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--radius", type=float, default=1.6)
    parser.add_argument("--vertices", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1199)
    parser.add_argument("--output", default="benchmarks/results/1199-first-pass")
    parser.add_argument("--worker", choices=["timing", "memory_spatialdata", "memory_anndata"])
    parser.add_argument("--cells", type=int)
    parser.add_argument("--regions", type=int)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.worker:
        print("BENCHMARK_JSON=" + json.dumps(worker(args)))
        return

    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        environment[name] = "1"
    environment["TQDM_DISABLE"] = "1"
    case_results, rows = [], []
    for case in args.cases:
        cells, regions = map(int, case.split(":"))
        if not (cells >= regions >= 1 and args.repeats >= 1 and args.vertices >= 3 and args.radius > 0):
            parser.error("Require cells >= regions >= 1, repeats >= 1, vertices >= 3 and radius > 0.")
        collected = {}
        for mode in ("timing", "memory_spatialdata", "memory_anndata"):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                mode,
                "--cells",
                str(cells),
                "--regions",
                str(regions),
                "--repeats",
                str(args.repeats),
                "--radius",
                str(args.radius),
                "--vertices",
                str(args.vertices),
                "--seed",
                str(args.seed),
                "--output",
                str(output),
            ]
            completed = subprocess.run(command, env=environment, capture_output=True, text=True, check=False)
            if completed.returncode:
                raise RuntimeError(f"{case} / {mode} failed:\n{completed.stdout}\n{completed.stderr}")
            line = next(line for line in completed.stdout.splitlines() if line.startswith("BENCHMARK_JSON="))
            collected[mode] = json.loads(line.split("=", 1)[1])
            print(f"Completed {case} {mode}", flush=True)
        samples = collected["timing"]["samples"]
        medians = {}
        for kind in ("spatialdata_first_graph", "spatialdata_repeat", "anndata_coordinates"):
            group = [sample for sample in samples if sample["kind"] == kind]
            medians[kind] = {
                name: statistics.median(sample[name] for sample in group)
                for name in ("total_s", "resolve_s", "graph_s", "centroid_calls")
            }
        collected["medians"] = medians
        case_results.append(collected)
        rows.extend({"cells": cells, "regions": regions, **sample} for sample in samples)
        print(json.dumps({"case": case, "medians": medians}), flush=True)

    repository = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    summary = {
        "git_sha": sha,
        "interpreter": Path(sys.executable).name,
        "hardware": {"platform": platform.platform(), "machine": platform.machine(), "logical_cpus": os.cpu_count()},
        "thread_environment": {
            name: environment[name]
            for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")
        },
        "parameters": vars(args),
        "method": {
            "density": 1.0,
            "polygon_radius": 0.25,
            "coordinate_system": "global (Identity)",
            "copy": True,
            "n_jobs": 1,
            "expression_matrix": "absent",
            "timing": "warm process; existing coordinates; alternating repeated-input order",
            "memory": "separate process per input type; both prepared inputs resident; sampled RSS",
        },
        "cases": case_results,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (output / "samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
