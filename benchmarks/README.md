# Spatial-neighbor input benchmark for issue #1199

This first pass measures the existing implementation at Git commit `f03069089ce7fdc45a513eef83598796073665ac`.
No Squidpy runtime code was changed. The AnnData path is a reference using
already prepared coordinates, not a measured implementation of a new cache.

## Method

- Synthetic 16-vertex polygons, radius 0.25; density one cell per square unit.
  Region area increases with cell count. Seed 1199, graph radius 1.6.
- Identical SpatialData-derived coordinates, categorical region groups and row
  order for both inputs. Coordinates already exist in the SpatialData table.
- No expression matrix; `copy=True`, `n_jobs=1`, identity coordinate transforms.
- Imports, small-fixture warmup, input construction and coordinate preparation
  are outside graph timings. The first graph call is not a cold-start measurement.
- Five repeats per input, alternating execution order. Separate cumulative
  timings for `_resolve_data` and `_run_spatial_neighbors`; centroid calls counted.
- Every measured graph agrees with the reference in CSR structure and values
  (`rtol=atol=1e-12`). Input observations and coordinates remain unchanged.
- Profiling runs separately. Memory runs in fresh subprocesses without reference
  graphs, samples RSS every 2 ms, and keeps both prepared inputs resident.

## Timing results

Times are seconds. Repeated-call values are medians of five measurements;
resolver share is the ratio of the two medians. Ratio is SpatialData repeat /
AnnData reference.

| Cells | Regions | SpatialData first graph | SpatialData repeat | Resolver | Resolver share | AnnData reference | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 0.061400 | 0.059459 | 0.018401 | 30.9% | 0.041753 | 1.42 |
| 100,000 | 1 | 0.519355 | 0.517828 | 0.066715 | 12.9% | 0.453132 | 1.14 |
| 100,000 | 20 | 0.669610 | 0.669665 | 0.275670 | 41.2% | 0.392684 | 1.71 |

Each SpatialData graph call recomputed centroids once per region; AnnData calls
made zero centroid calls. Existing table coordinates did not remove that work.
At fixed total cell count, the 20-region case incurred more input-preparation
cost than the single-region case. Its resolver profile includes repeated
SpatialData transformation and Dask conversion/materialization work; resolver
time must not be labelled solely polygon-centroid computation.

## Sampled RSS

Values are MiB. These are separate single-run snapshots, not timing-run memory
measurements or allocation totals. The incremental column subtracts RSS just
before graph construction from sampled peak RSS.

| Cells | Regions | SData baseline | SData peak | SData increment | AnnData baseline | AnnData peak | AnnData increment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 440.72 | 451.64 | 10.92 | 440.98 | 452.84 | 11.86 |
| 100,000 | 1 | 608.47 | 729.05 | 120.58 | 594.31 | 700.02 | 105.70 |
| 100,000 | 20 | 518.58 | 582.50 | 63.92 | 518.91 | 578.55 | 59.64 |

## Reproduce

Use an environment satisfying this checkout's `pyproject.toml`, then run from
its root:

```bash
PYTHONPATH="$PWD/src" python benchmarks/bench_spatial_neighbors_input.py
```

A smaller smoke run:

```bash
PYTHONPATH="$PWD/src" python benchmarks/bench_spatial_neighbors_input.py \
  --cases 200:2 --repeats 2 --output /tmp/squidpy-1199-pilot
```

The script asserts that Squidpy is imported from this checkout. OMP, OpenBLAS,
MKL and Numba thread limits are set to one in worker processes. `summary.json`
records the interpreter, hardware, dependency versions, parameters and SHA.
The environment's editable package version label refers to an older installation;
the verified source path and Git SHA identify the code measured here.

Python 3.13.3; macOS-26.7-arm64-arm-64bit-Mach-O. Package versions: spatialdata 0.8.0, anndata 0.13.3.post0, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, shapely 2.1.2, dask 2026.8.0.

Raw measurements: [samples.csv](results/1199-first-pass/samples.csv).
Metadata and summaries: [summary.json](results/1199-first-pass/summary.json).
Independent resolver profiles: [10k / 1 region](results/1199-first-pass/profile-10000-1.txt),
[100k / 1 region](results/1199-first-pass/profile-100000-1.txt),
[100k / 20 regions](results/1199-first-pass/profile-100000-20.txt).
