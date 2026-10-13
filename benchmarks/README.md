# Spatial-neighbor input benchmark for issue #1199

The first-pass baseline below measures the implementation at Git commit `f03069089ce7fdc45a513eef83598796073665ac`.
That baseline uses unchanged Squidpy runtime code. The AnnData reference measures
graph construction with already prepared coordinates; no new cache was measured.

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

Values are MiB, measured in separate single runs. The incremental column subtracts
RSS just before graph construction from sampled peak RSS. These snapshots measure
resident memory separately from the timing runs; they do not measure allocation totals.

| Cells | Regions | SData baseline | SData peak | SData increment | AnnData baseline | AnnData peak | AnnData increment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 440.72 | 451.64 | 10.92 | 440.98 | 452.84 | 11.86 |
| 100,000 | 1 | 608.47 | 729.05 | 120.58 | 594.31 | 700.02 | 105.70 |
| 100,000 | 20 | 518.58 | 582.50 | 63.92 | 518.91 | 578.55 | 59.64 |

## Reproduce

Use an environment satisfying this checkout's `pyproject.toml`, then run from
its root. The updated script compares all three paths and saves new results to
`results/1199-coordinate-reuse`; the first-pass results above remain unchanged:

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
records the interpreter, hardware, dependency versions, parameters, SHA, dirty
working-tree status and SHA-256 hashes of the measured source and benchmark script.
The source hashes identify uncommitted implementations independently of the
editable package version label.

First-pass environment: Python 3.13.3; macOS-26.7-arm64-arm-64bit-Mach-O. Package versions: spatialdata 0.8.0, anndata 0.13.3.post0, numpy 2.5.3, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.1, shapely 2.1.2, dask 2026.8.0.

Raw measurements: [samples.csv](results/1199-first-pass/samples.csv).
Metadata and summaries: [summary.json](results/1199-first-pass/summary.json).
Independent resolver profiles: [10k / 1 region](results/1199-first-pass/profile-10000-1.txt),
[100k / 1 region](results/1199-first-pass/profile-100000-1.txt),
[100k / 20 regions](results/1199-first-pass/profile-100000-20.txt).

## Coordinate reuse implementation

All six public spatial-neighbor interfaces accept `coordinate_source="table"`
to reuse existing coordinates. The default `"elements"` path computes and
transforms centroids. Table mode reads `obsm[spatial_key]` as stored, so the caller
supplies aligned, finite, real numeric NumPy coordinates in the desired coordinate
system. Region and instance association checks apply to both paths. With
`copy=True`, coordinates are copied into a lightweight temporary AnnData;
the expression matrix is not copied.

This run compares SpatialData element mode, SpatialData table mode and the
AnnData reference on the same prepared inputs. Five repeats rotate the first
path in each trial. The first graph call for each SpatialData mode is recorded
separately; imports, fixture preparation and warmup are excluded. Every timed
graph matches the reference CSR structure and values (`rtol=atol=1e-12`).
Both inputs remain unchanged. Table mode makes zero centroid calls in every
sample, compared with one call per region for element mode.

These measurements use Linux x86_64 and Python 3.12.14. The paths in this run
can be compared directly. The historical baseline above used macOS, so comparing
the two runs would also include a hardware difference. The measured code is an uncommitted
working-tree implementation based on `704348fcd6418af5deb051fe09050321b88b20c1`;
`summary.json` records dirty status and SHA-256 hashes of the source and script.

### Repeated graph timings

Times are seconds; each entry is the median of five runs. The ratio is
element-mode total / table-mode total; values above one indicate faster table mode.

| Cells | Regions | Elements total | Elements resolver | Table total | Table resolver | AnnData total | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 0.621056 | 0.353286 | 0.272928 | 0.036006 | 0.287705 | 2.28 |
| 100,000 | 1 | 2.418332 | 0.304803 | 2.541684 | 0.089694 | 2.322077 | 0.95 |
| 100,000 | 20 | 4.528471 | 2.584326 | 2.080445 | 0.229310 | 1.515549 | 2.18 |

Input preparation is about 3.4–11.3 times faster across the three cases. Total
repeated-graph time improves by about 2.3 times for 10k / 1 region and 2.2 times
for 100k / 20 regions. For 100k / 1 region, table mode is about 5% slower overall
despite a faster resolver. Its graph-stage medians are 2.424483 s (table) versus 2.155961 s
(elements). Graph-stage timings vary even though the builder is unchanged;
these five-repeat measurements do not establish a universal end-to-end speedup.
The resolver includes association checks, coordinate validation/copying and
element processing; its savings must not be labelled solely centroid computation.

### Sampled RSS for the implementation

Each path runs once in its own fresh process, with both prepared inputs
resident and no reference graph. RSS is sampled every 2 ms. Values are MiB;
increments subtract the pre-graph baseline from the sampled peak. These single-run
observations report sampled resident memory; they do not measure allocation totals
or establish exact peak-memory bounds.

| Cells | Regions | Elements baseline | Elements peak | Elements increment | Table baseline | Table peak | Table increment | AnnData baseline | AnnData peak | AnnData increment |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10,000 | 1 | 366.39 | 375.57 | 9.18 | 366.40 | 374.36 | 7.96 | 366.37 | 376.25 | 9.88 |
| 100,000 | 1 | 488.84 | 540.08 | 51.24 | 489.99 | 545.27 | 55.29 | 487.59 | 541.91 | 54.32 |
| 100,000 | 20 | 438.45 | 503.82 | 65.38 | 438.34 | 498.33 | 59.99 | 438.29 | 494.04 | 55.75 |

The RSS measurements do not show a consistent memory reduction across all
cases. Table mode still allocates graph outputs, performs association checks
and copies coordinates for `copy=True`. Labels may still require reading
instance IDs even when centroid computation is skipped.

Packages: spatialdata 0.8.0, anndata 0.13.4, numpy 2.5.4, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, shapely 2.2.0, dask 2026.8.0.

Implementation measurements: [samples.csv](results/1199-coordinate-reuse/samples.csv).
Metadata and summaries: [summary.json](results/1199-coordinate-reuse/summary.json).
Separate resolver profiles:
- 10k / 1 region: [elements](results/1199-coordinate-reuse/profile-10000-1-elements.txt), [table](results/1199-coordinate-reuse/profile-10000-1-table.txt).
- 100k / 1 region: [elements](results/1199-coordinate-reuse/profile-100000-1-elements.txt), [table](results/1199-coordinate-reuse/profile-100000-1-table.txt).
- 100k / 20 regions: [elements](results/1199-coordinate-reuse/profile-100000-20-elements.txt), [table](results/1199-coordinate-reuse/profile-100000-20-table.txt).
