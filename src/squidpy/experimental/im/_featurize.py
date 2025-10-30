from __future__ import annotations

from collections.abc import Callable
from multiprocessing import Manager
from threading import Thread

import dask.array as da
import numpy as np
import pandas as pd
import spatialdata as sd
from anndata import AnnData
from spatialdata._logging import logger
from spatialdata.models import TableModel

from squidpy._utils import Signal, SigQueue

from ._utils import _get_element_data

try:
    import ipywidgets  # noqa: F401
    from tqdm.auto import tqdm
except ImportError:
    try:
        from tqdm.std import tqdm
    except ImportError:
        tqdm = None

try:
    import joblib as jl
except ImportError:
    jl = None

__all__ = ["featurize_tiles"]


def featurize_tiles(
    sdata: sd.SpatialData,
    image_key: str,
    featurizer: Callable[[np.ndarray], np.ndarray],
    *,
    shapes_key: str | None = None,
    table_key: str | None = None,
    filter_tissue_only: bool = True,
    batch_size: int | None = None,
    n_jobs: int | None = None,
    device: str | None = None,
    new_features_table_key: str | None = None,
    scale: str = "scale0",
    show_progress: bool = True,
) -> None:
    """
    Extract features from tiles using a featurizer function.

    This function extracts tiles from the image, filters to valid "tissue" tiles,
    applies a featurizer function (e.g., a vision model), and saves the features
    as an AnnData table linked to the tile grid.

    Parameters
    ----------
    sdata
        SpatialData object containing the image and tile grid.
    image_key
        Key of the image in ``sdata.images``.
    featurizer
        Callable function. The input shape depends on processing mode:
        - If ``batch_size`` is ``None``: receives single tiles of shape (H, W, C)
        - If ``batch_size`` > 1: receives batches of shape (batch_size, H, W, C)
        Returns features of shape (feature_dim,) for single tiles or
        (batch_size, feature_dim) for batches. The function should handle
        device placement (CPU/GPU) internally if needed.

        **Important**: If you use ``batch_size``, your featurizer MUST handle
        batched input (4D arrays). For single-tile featurizers, pass ``batch_size=None``.
    shapes_key
        Key of the tile grid in ``sdata.shapes``. If ``None``, defaults to
        ``f"{image_key}_tile_grid"``.
    table_key
        Key of the inference tiles table in ``sdata.tables``. If ``None``,
        defaults to ``f"{image_key}_inference_tiles"``.
    filter_tissue_only
        If ``True``, only process tiles classified as "tissue".
        Default is ``True``.
    batch_size
        Batch size for GPU processing. If set, tiles are grouped into batches
        of this size and passed as a numpy array of shape (batch_size, H, W, C)
        to the featurizer. This is more efficient for GPU models. For CPU processing,
        use ``n_jobs`` for parallelization instead. If ``None``, processes tiles
        one at a time. Default is ``None``.
    n_jobs
        Number of parallel jobs for CPU processing. If > 1, uses multiprocessing
        to process tiles in parallel (only works for CPU-based featurizers).
        For GPU processing, use ``batch_size`` instead. If ``None`` or 1,
        processes tiles sequentially. Default is ``None``.
    device
        Optional device string (e.g., "cuda", "cpu") for documentation.
        The featurizer is responsible for device management.
        Default is ``None``.
    show_progress
        Whether to show a progress bar. Default is ``True``.
    new_features_table_key
        Key to save the features table in ``sdata.tables``. If ``None``,
        defaults to ``f"{image_key}_tile_features"``.
    scale
        Scale level to use for image extraction. Default is ``"scale0"``
        (largest/finest scale).

    Returns
    -------
    None
        Features are saved to ``sdata.tables[new_features_table_key]``.

    Raises
    ------
    KeyError
        If required keys are not found in ``sdata``.
    ValueError
        If no valid tiles are found after filtering.

    Examples
    --------
    >>> import numpy as np
    >>> import squidpy as sq
    >>> from torchvision import models
    >>> import torch
    >>>
    >>> # Define a simple featurizer
    >>> model = models.resnet18(pretrained=True)
    >>> model.eval()
    >>> model = model.to("cuda")  # Move to GPU if available
    >>>
    >>> # Option 1: Single-tile processing (GPU or CPU)
    >>> def extract_features(tile):
    ...     # tile is shape (H, W, C) - single tile
    ...     tile = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
    ...     tile = tile.unsqueeze(0).to("cuda")
    ...     with torch.no_grad():
    ...         features = model(tile)
    ...     return features.cpu().numpy().squeeze()
    >>>
    >>> sq.experimental.im.featurize_tiles(sdata, "image", extract_features)
    >>>
    >>> # Option 2: Batched processing for GPU efficiency
    >>> # IMPORTANT: When using batch_size, featurizer receives 4D arrays (batch_size, H, W, C)
    >>> def extract_features_batch(tile_batch):
    ...     # tile_batch is shape (batch_size, H, W, C) - note the 4 dimensions
    ...     batch = torch.from_numpy(tile_batch).permute(0, 3, 1, 2).float() / 255.0
    ...     batch = batch.to("cuda")
    ...     with torch.no_grad():
    ...         features = model(batch)
    ...     return features.cpu().numpy()
    >>>
    >>> sq.experimental.im.featurize_tiles(sdata, "image", extract_features_batch, batch_size=32)
    >>>
    >>> # Note: If you have a single-tile featurizer and want to use batching,
    >>> # you need to modify it to handle batched input, OR don't use batch_size
    >>> # (the default sequential processing will work fine)
    >>>
    >>> # Option 3: Parallel CPU processing
    >>> sq.experimental.im.featurize_tiles(sdata, "image", extract_features, n_jobs=4)
    """
    # Resolve keys
    shapes_key = shapes_key or f"{image_key}_tile_grid"
    table_key = table_key or f"{image_key}_inference_tiles"
    features_table_key = new_features_table_key or f"{image_key}_tile_features"

    # Validate keys exist
    if image_key not in sdata.images:
        raise KeyError(f"Image key '{image_key}' not found in sdata.images")
    if shapes_key not in sdata.shapes:
        raise KeyError(f"Shapes key '{shapes_key}' not found in sdata.shapes")
    if table_key not in sdata.tables:
        raise KeyError(f"Table key '{table_key}' not found in sdata.tables")

    # Get tile information from table
    adata_table = sdata.tables[table_key]

    # Filter to tissue tiles if requested
    if filter_tissue_only:
        if "tile_classification" not in adata_table.obs.columns:
            raise KeyError(f"'tile_classification' column not found in table '{table_key}'")
        tissue_mask = adata_table.obs["tile_classification"] == "tissue"
        adata_tissue = adata_table[tissue_mask]
        if adata_tissue.n_obs == 0:
            raise ValueError("No 'tissue' tiles found after filtering")
    else:
        adata_tissue = adata_table

    # Get tile bounds
    required_cols = ["pixel_y0", "pixel_x0", "pixel_y1", "pixel_x1"]
    missing_cols = [col for col in required_cols if col not in adata_tissue.obs.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in table '{table_key}': {missing_cols}")

    tile_bounds = np.column_stack(
        [
            adata_tissue.obs["pixel_y0"].values,
            adata_tissue.obs["pixel_x0"].values,
            adata_tissue.obs["pixel_y1"].values,
            adata_tissue.obs["pixel_x1"].values,
        ]
    )
    tile_ids = adata_tissue.obs.index

    # Get image data
    img_node = sdata.images[image_key]
    img_da = _get_element_data(img_node, scale, "image", image_key)

    # Convert to numpy array, ensuring (y, x, c) format
    if isinstance(img_da.data, da.Array):
        # Handle dask arrays
        img_array = np.asarray(img_da.compute())
    else:
        img_array = np.asarray(img_da.values)

    # Reorder dimensions to (y, x, c) if needed
    if "y" in img_da.dims and "x" in img_da.dims:
        dim_order = ["y", "x"]
        if "c" in img_da.dims:
            dim_order.append("c")
        img_transposed = img_da.transpose(*dim_order)
        if isinstance(img_transposed.data, da.Array):
            img_array = np.asarray(img_transposed.compute())
        else:
            img_array = np.asarray(img_transposed.values)

    H, W = img_array.shape[0], img_array.shape[1]
    if img_array.ndim == 2:
        img_array = img_array[:, :, np.newaxis]  # Add channel dimension

    logger.info(f"Extracting features from {len(tile_bounds)} tiles (image shape: {H}x{W})...")

    # Extract all valid tiles first
    valid_tiles = []
    valid_tile_ids = []
    skipped_tiles = 0

    for i, (y0, x0, y1, x1) in enumerate(tile_bounds):
        # Clamp and validate coordinates
        y0_int = max(0, int(np.round(y0)))
        y1_int = min(H, int(np.round(y1)))
        x0_int = max(0, int(np.round(x0)))
        x1_int = min(W, int(np.round(x1)))

        if y1_int <= y0_int or x1_int <= x0_int:
            logger.warning(f"Tile {i} (id: {tile_ids[i]}) has invalid bounds, skipping")
            skipped_tiles += 1
            continue

        # Extract tile region
        tile = img_array[y0_int:y1_int, x0_int:x1_int, :]
        valid_tiles.append(tile)
        valid_tile_ids.append(tile_ids[i])

    if not valid_tiles:
        raise ValueError("No valid tiles found after filtering")

    # Determine processing mode
    use_batching = batch_size is not None and batch_size > 1
    use_parallel = n_jobs is not None and n_jobs > 1 and jl is not None and not use_batching

    if use_parallel and use_batching:
        logger.warning("Both batch_size and n_jobs are set. Using batch_size for GPU processing.")
        use_parallel = False

    # Process tiles
    if use_batching:
        # GPU batching mode: process tiles in batches
        all_features = []
        all_tile_ids = []
        pbar = tqdm(total=len(valid_tiles), desc="Featurizing tiles", unit="tile") if (show_progress and tqdm) else None

        def _process_batch(tile_batch: list[np.ndarray]) -> np.ndarray:
            """Process a batch of tiles through the featurizer."""
            batch_array = np.stack(tile_batch, axis=0)  # (batch_size, H, W, C)
            try:
                features = featurizer(batch_array)
            except (RuntimeError, ValueError, IndexError) as e:
                # Common error: featurizer expects single tiles but got batch
                error_msg = str(e).lower()
                if "dimension" in error_msg or "permute" in error_msg or "input.dim()" in error_msg:
                    raise ValueError(
                        f"Featurizer failed with batched input. Your featurizer appears to expect "
                        f"single tiles (3D: H, W, C) but received a batch (4D: batch_size={batch_array.shape[0]}, H, W, C). "
                        f"Either:\n"
                        f"  1. Modify your featurizer to handle batches (4D input with permute(0, 3, 1, 2)), OR\n"
                        f"  2. Set batch_size=None to use single-tile processing"
                    ) from e
                raise

            features = np.asarray(features)
            # Expect 2D output (batch_size, feature_dim) or 3D that we can reshape
            if features.ndim == 1:
                # Single feature vector for whole batch - not expected but handle it
                features = features[np.newaxis, :]
            elif features.ndim > 2:
                # Flatten extra dimensions
                features = features.reshape(features.shape[0], -1)
            return features

        current_batch: list[np.ndarray] = []
        batch_tile_ids: list[str] = []

        # mypy: batch_size is Optional, but guarded by use_batching
        assert batch_size is not None
        for tile, tile_id in zip(valid_tiles, valid_tile_ids, strict=True):
            current_batch.append(tile)
            batch_tile_ids.append(tile_id)

            if len(current_batch) >= batch_size:
                # Process batch
                batch_features = _process_batch(current_batch)
                all_features.append(batch_features)
                all_tile_ids.extend(batch_tile_ids)
                if pbar:
                    pbar.update(len(current_batch))
                current_batch = []
                batch_tile_ids = []

        # Process remaining tiles
        if current_batch:
            batch_features = _process_batch(current_batch)
            all_features.append(batch_features)
            all_tile_ids.extend(batch_tile_ids)
            if pbar:
                pbar.update(len(current_batch))

        if pbar:
            pbar.close()

        # Concatenate batch results
        features_array = np.concatenate(all_features, axis=0)

    elif use_parallel:
        # CPU parallel mode: use multiprocessing
        def _process_single_tile(args: tuple[np.ndarray, str], queue: SigQueue | None = None) -> tuple[np.ndarray, str]:
            """Process a single tile - for parallel execution."""
            tile, tile_id = args
            features = featurizer(tile)
            features = np.asarray(features)
            if features.ndim > 1:
                features = features.flatten()
            if queue is not None:
                queue.put(Signal.UPDATE)  # Signal completion
            return features, tile_id

        # Set up progress bar with queue-based updates (following pattern from squidpy._utils.parallelize)
        pbar = None
        queue: SigQueue | None = None
        update_thread = None

        if show_progress and tqdm:
            pbar = tqdm(total=len(valid_tiles), desc="Featurizing tiles", unit="tile")
            queue = Manager().Queue()  # type: ignore[assignment]

            def update_pbar() -> None:
                """Update progress bar from queue signals."""
                n_completed = 0
                q = queue
                if q is None:
                    return
                while n_completed < len(valid_tiles):
                    try:
                        res = q.get()
                    except EOFError:
                        break
                    assert isinstance(res, Signal), f"Invalid type `{type(res).__name__}`."
                    if res == Signal.UPDATE:
                        if pbar is not None:
                            pbar.update(1)
                        n_completed += 1

                if pbar is not None:
                    # Ensure progress bar reaches 100%
                    if n_completed < len(valid_tiles):
                        pbar.update(len(valid_tiles) - n_completed)
                    pbar.close()

            update_thread = Thread(target=update_pbar, name="FeaturizeTilesUpdateThread")
            update_thread.start()

        # Process in parallel
        with jl.parallel_config(backend="loky", n_jobs=n_jobs):
            results = jl.Parallel(n_jobs=n_jobs)(
                jl.delayed(_process_single_tile)((tile, tile_id), queue=queue)
                for tile, tile_id in zip(valid_tiles, valid_tile_ids, strict=True)
            )

        # Wait for progress bar thread to finish
        if update_thread is not None:
            update_thread.join()

        # Extract results
        all_features = [feat for feat, _ in results]
        all_tile_ids = [tid for _, tid in results]

        features_array = np.vstack(all_features)

    else:
        # Sequential mode: process one tile at a time
        all_features = []
        all_tile_ids = []
        pbar = tqdm(total=len(valid_tiles), desc="Featurizing tiles", unit="tile") if (show_progress and tqdm) else None

        for tile, tile_id in zip(valid_tiles, valid_tile_ids, strict=True):
            # Process single tile - featurizer receives (H, W, C)
            features = featurizer(tile)
            features = np.asarray(features)
            # Ensure 1D output (feature_dim,) - flatten if needed
            if features.ndim > 1:
                features = features.flatten()
            all_features.append(features)
            all_tile_ids.append(tile_id)
            if pbar:
                pbar.update(1)

        if pbar:
            pbar.close()

        features_array = np.vstack(all_features)

    if skipped_tiles > 0:
        logger.warning(f"Skipped {skipped_tiles} tiles with invalid bounds")

    logger.info(f"Extracted features with shape: {features_array.shape} (feature_dim={features_array.shape[1]})")

    # Create AnnData object with features
    adata_features = AnnData(X=features_array)
    adata_features.obs = adata_tissue.obs.reindex(all_tile_ids)
    adata_features.obs.index = pd.Index(all_tile_ids)

    # Copy spatial coordinates for processed tiles
    if "spatial" in adata_tissue.obsm:
        # Create mapping from tile_id to index position in original adata_tissue
        id_to_idx = {tid: idx for idx, tid in enumerate(adata_tissue.obs.index)}
        spatial_indices = [id_to_idx[tid] for tid in all_tile_ids]
        adata_features.obsm["spatial"] = adata_tissue.obsm["spatial"][spatial_indices]

    # Link to shapes via spatialdata_attrs
    if "spatialdata_attrs" in adata_tissue.uns:
        adata_features.uns["spatialdata_attrs"] = adata_tissue.uns["spatialdata_attrs"].copy()
    else:
        # Fallback: create spatialdata_attrs from shapes
        adata_features.uns["spatialdata_attrs"] = {
            "region": shapes_key,
            "region_key": "grid_name",
            "instance_key": "tile_id",
        }
        if "grid_name" not in adata_features.obs.columns:
            adata_features.obs["grid_name"] = pd.Categorical([shapes_key] * len(adata_features))
        if "tile_id" not in adata_features.obs.columns:
            adata_features.obs["tile_id"] = adata_features.obs.index

    # Store metadata
    adata_features.uns["tile_featurization"] = {
        "image_key": image_key,
        "shapes_key": shapes_key,
        "source_table_key": table_key,
        "scale": scale,
        "n_tiles": len(all_tile_ids),
        "n_skipped": skipped_tiles,
        "feature_dim": features_array.shape[1],
        "filter_tissue_only": filter_tissue_only,
        "batch_size": batch_size,
        "n_jobs": n_jobs,
        "device": device,
    }

    # Add feature names
    adata_features.var_names = [f"feature_{i}" for i in range(features_array.shape[1])]

    # Save to sdata
    sdata.tables[features_table_key] = TableModel.parse(adata_features)
    logger.info(f"Saved tile features to 'sdata.tables[\"{features_table_key}\"]'")
