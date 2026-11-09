# Tissue detection, image QC, and tiling — Squidpy 1.8.4

All procedures here use `spatialdata.SpatialData`, with named image, labels, and shapes elements. Most calls write to the container. Inspect existing keys and choose output names deliberately.

## Tissue segmentation and tissue masks

`sq.experimental.im.detect_tissue(sdata, image_key=..., ...)` supports `method="otsu"` (default), `"felzenszwalb"`, or `"weka"`. `scale="auto"` uses the smallest/coarsest scale. `method_params` must correspond to the chosen method (and is not accepted for Otsu). Corner-background priors matter for tissue/background polarity. It can store labeled tissue regions; default name is based on the image key (inspect output and/or pass `new_labels_key`).

Use a representative low-resolution scale for whole slides; do not materialize the full-resolution pyramid for a mask meant only to guide later operations.

## Image QC

`sq.experimental.im.qc_image(sdata, image_key=..., ...)` computes tile metrics: sharpness, intensity, entropy, H&E-specific stain metrics and folds, and tissue coverage. Set `is_hne=False` for non-H&E data (H&E-only metrics are invalid there). With defaults, it can create a missing tissue mask and writes:

- `sdata.tables[f"qc_img_{image_key}"]`: QC AnnData, including metadata in `.uns["qc_image"]`.
- `sdata.shapes[f"qc_img_{image_key}_grid"]`: QC tile geometries.

Use `preview=False` for a headless agent or automated evaluation; inspect a representative region separately when necessary.

## Regular tiling

`sq.experimental.im.make_tiles(sdata, image_key=..., tile_size=(224, 224), ...)` creates shapes in `sdata.shapes[f"{image_key}_tiles"]` by default. Classes are `"background"`, `"partial_tissue"`, and `"tissue"`, based on `min_tissue_fraction`. It uses the full-resolution image as the tile grid coordinate basis; `scale` selects the mask scale for centering/classification. If a required mask is absent, it can invoke `detect_tissue()` and **mutate `sdata.labels`**. `image_mask_key` determines grid centering; `tissue_mask_key` determines tissue classification.

## Spot-centered tiling

`sq.experimental.im.make_tiles_from_spots(sdata, spots_key=..., ...)` derives one tile per point-like spot in `sdata.shapes[spots_key]`; it estimates a square tile width from vertical spot spacing. Ensure spot and image/mask elements refer to the same coordinate system. If `image_key` is omitted, an explicit tissue mask still supports classification; without both, classification is unavailable.

## Checks

- Distinguish tile geometries from actual image crops: shapes do not imply pixel patches were materialized.
- Verify shape count, tile extents, mask scale, coordinate transformations, and classification proportions.
- Preview controls can cause plotting; disable them in pipelines that must not open figures.
- Avoid using a classification threshold of `1.0` (the default) without understanding that partly covered tissue tiles will be excluded from the `"tissue"` class.

Sources:
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.detect_tissue.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.qc_image.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.make_tiles.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.make_tiles_from_spots.html
