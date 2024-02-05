"""Utils for plotting functions."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from cycler import Cycler, cycler
from matplotlib.colors import ListedColormap, to_hex, to_rgba
from scanpy import logging as logg
from scanpy.plotting._utils import add_colors_for_categorical_sample_annotation

from squidpy._constants._pkg_constants import Key

Palette_t = Union[str, ListedColormap, None]


def _maybe_set_colors(
    source: AnnData, target: AnnData, key: str, palette: str | ListedColormap | Cycler | Sequence[Any] | None = None
) -> None:
    color_key = Key.uns.colors(key)
    try:
        if palette is not None:
            raise KeyError("Unable to copy the palette when there was other explicitly specified.")
        target.uns[color_key] = source.uns[color_key]
    except KeyError:
        if isinstance(palette, ListedColormap):  # `scanpy` requires it
            palette = cycler(color=palette.colors)
        add_colors_for_categorical_sample_annotation(target, key=key, force_update_colors=True, palette=palette)


def _get_palette(
    adata: AnnData,
    cluster_key: str | None,
    categories: Sequence[Any],
    palette: Palette_t = None,
    alpha: float = 1.0,
) -> Mapping[str, str] | None:
    if palette is None:
        try:
            palette = adata.uns[Key.uns.colors(cluster_key)]  # type: ignore[arg-type]
            if len(palette) != len(categories):
                raise ValueError(
                    f"Expected palette to be of length `{len(categories)}`, found `{len(palette)}`. "
                    + f"Removing the colors in `adata.uns` with `adata.uns.pop('{cluster_key}_colors')` may help."
                )
            return {cat: to_hex(to_rgba(col)[:3] + (alpha,), keep_alpha=True) for cat, col in zip(categories, palette)}
        except KeyError as e:
            logg.error(f"Unable to fetch palette, reason: {e}. Using `None`.")
            return None

    len_cat = len(categories)
    if isinstance(palette, str):
        cmap = plt.get_cmap(palette)
        palette = [to_hex(x, keep_alpha=True) for x in cmap(np.linspace(0, 1, len_cat), alpha=alpha)]
    elif isinstance(palette, ListedColormap):
        palette = [to_hex(x, keep_alpha=True) for x in palette(np.linspace(0, 1, len_cat), alpha=alpha)]
    else:
        raise TypeError(f"Palette is {type(palette)} but should be string or `ListedColormap`.")

    return dict(zip(categories, palette))
