# Per-cell image feature extraction — Squidpy 1.8.4

Use `sq.experimental.im.calculate_image_features` for SpatialData labels or shapes. This reference follows the installed 1.8.4 release source, rather than the stale documentation used in the first draft. The legacy `sq.im.calculate_image_features` uses a different input model and contract.

## Choose the measurements explicitly

`features=None` computes all feature groups and requires an image. Select the measurements needed for the analysis to avoid unnecessary computation and accidental backend mixing:

| Measurement                             | Feature flag                         | Image required                     |
| --------------------------------------- | ------------------------------------ | ---------------------------------- |
| Region area                             | `"skimage:morphology:area"`          | No, for labels input               |
| Region morphology                       | `"skimage:morphology"`               | No, for labels input               |
| Mean intensity                          | `"skimage:intensity:intensity_mean"` | Yes                                |
| Region intensity properties             | `"skimage:intensity"`                | Yes                                |
| Per-cell summary statistics             | `"squidpy:summary"`                  | Yes                                |
| GLCM texture                            | `"squidpy:texture"`                  | Yes                                |
| Intensity histogram                     | `"squidpy:histogram"`                | Yes                                |
| CellProfiler-derived shape features     | `"cp_measure:sizeshape"`             | Yes, required by the 1.8.4 wrapper |
| CellProfiler-derived intensity features | `"cp_measure:intensity"`             | Yes                                |

The prefix is **`cp_measure:`**, not `cpmeasure:`. `squidpy:color_hist` is not a valid 1.8.4 flag. cp_measure uses its native feature names, while skimage area is named `area`. When combining cp_measure sizeshape with skimage morphology, redundant skimage properties are removed; do not assume both copies will remain.

## Inputs, output, and alignment

- Select the actual `labels_key` or `shapes_key` and, for intensity/texture, `image_key`. Shapes input also requires an image for rasterization. Channels are selected by **names**, not integer indices.
- Set the output table with **`key_added`**. There is no `adata_key_added` argument. By default the table key is `f"morphology_{labels_key or shapes_key}_{image_key}"`, or `f"morphology_{labels_key or shapes_key}"` without an image. With `inplace=False`, the function returns an AnnData instead of writing it.
- `drop_constant_features=True` by default removes zero-variance columns. Use `False` when requested measurements must remain present or when combining samples with a shared feature schema; filter constants after concatenation when appropriate. `invalid_as_zero=False` preserves undefined values rather than conflating them with genuine zeros.
- `align_mode="strict"` accepts identity or integer-pixel translation between image and labels. Equal array shapes alone do not establish physical registration. `align_mode="rasterize"` can resample labels onto the image grid, but materializes the label grid; it is not supported for multiscale labels under a non-integer transform. Prefer explicit pre-alignment for large images.
- `tile_size` controls computational chunks. Cell-preserving overlap is inferred internally; **there is no public `overlap_margin` argument**. Check IDs and measurements for cells spanning tile boundaries.
- The output table retains the region/instance relationship. Match rows through its declared instance key, not row position, and inspect cells dropped outside the image/labels overlap.
- Inspect `.var_names` for backend-specific names. Mean intensity from skimage is `intensity_mean__<channel>` and squidpy texture uses `texture_<stat>__<channel>` (a double underscore before the channel). cp_measure retains its native names; do not invent a shared naming scheme across backends.
- `n_jobs=1` is the default. Larger values use worker processes unless a Dask client already exists. Standalone multiprocessing scripts on macOS/Windows need a main guard.

## Example call shape

Replace all element and channel names after inspecting the input:

```python
table = sq.experimental.im.calculate_image_features(
    sdata,
    image_key="microscopy",
    labels_key="segmentation",
    channels=["DAPI"],
    features=["skimage:morphology:area", "skimage:intensity:intensity_mean", "squidpy:texture"],
    drop_constant_features=False,
    inplace=False,
)
```

Every feature flag listed above has executed on a small runtime fixture, as has strict-alignment rejection. Shapes rasterization and other alignment modes still need representative workflow tests before making scientific claims about their output.

Release source: https://github.com/scverse/squidpy/blob/v1.8.4/src/squidpy/experimental/im/\_calculate_image_features.py
