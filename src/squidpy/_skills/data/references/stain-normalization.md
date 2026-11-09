# H&E stain normalization and decomposition — Squidpy 1.8.4

Use `sq.experimental.im.fit_stain_reference`, the returned `StainFit`, and optionally `estimate_white_point`. These operations target RGB histology in SpatialData, **not arbitrary fluorescence channels**.

## Correct order and output behavior

1. Inspect the target RGB image key and create/identify its tissue mask in `sdata.labels` (often `f"{image_key}_tissue"`). **Fitting requires this mask and will raise if absent**; use `sq.experimental.im.detect_tissue` beforehand if needed.
2. Fit once: `fit = sq.experimental.im.fit_stain_reference(sdata, image_key="histology", tissue_mask_key="histology_tissue", method="macenko")`. This returns a `StainFit` and does not write to SpatialData.
3. Apply to one or more slide images: `fit.transform(sdata, image_key="histology", tissue_mask_key="histology_tissue")`. By default it writes `sdata.images["histology_normalized"]`; this key must be free. With `inplace=False`, it returns a lazy DataArray.
4. For stain separation with Macenko/Vahadane only: `fit.decompose(sdata, image_key="histology")`. By default this writes separate H, E, and optional residual images (or returns a dict if `inplace=False`). Reinhard does **not** support physical stain decomposition.

## Method selection

| Method       | Meaning                                             | Limitations                                                   |
| ------------ | --------------------------------------------------- | ------------------------------------------------------------- |
| `"macenko"`  | Stain-matrix estimation / H&E transfer; the default | Fit can be compromised by artifacts; tissue mask is essential |
| `"vahadane"` | Stain-matrix decomposition                          | More computation; needs tissue mask                           |
| `"reinhard"` | Faster statistical color transfer                   | No H/E stain separation                                       |

A cohort reference can be fit with `image_key=[...]` and corresponding `tissue_mask_key=[...]`. The pool is by **tissue pixel count** at selected scales, not by equal slide weighting; image dtypes must match.

`fit_stain_reference(scale="auto")` works at the coarsest level. `fit.transform(scale="auto")` and `fit.decompose(scale="auto")` use the finest level. `transform` preserves SpatialData transformations and rebuilds multiscale pyramids when writing a new image. Background is preserved by default with `preserve_background=True`.

White point `I_0`: fit uses a full-white default for decomposition methods. Use `estimate_white_point` only when the slide background is known to be non-white; it samples non-tissue pixels at a coarse scale.

**Caution:** decomposition `"residual"` is a diagnostic absorbance component, not a biological third stain; do not interpret it as one. Quantitative downstream uses may warrant `output_dtype="float32"` rather than float16.

Sources:
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.fit_stain_reference.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.StainFit.transform.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.StainFit.decompose.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.estimate_white_point.html
