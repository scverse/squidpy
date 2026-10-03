from __future__ import annotations

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from matplotlib.collections import PathCollection

from squidpy import pl


def _adata(values):
    values = np.asarray(values)
    adata = AnnData(values[:, None].copy(), var=pd.DataFrame(index=["gene"]))
    adata.obs["value"] = values
    adata.obsm["spatial"] = np.column_stack((np.arange(len(values)), np.zeros(len(values))))
    adata.uns["spatial"] = {"sample": {"images": {}, "scalefactors": {}}}
    return adata


def _plot(adata, **kwargs):
    return pl.spatial_scatter(adata, img=False, img_res_key=None, size=0.4, return_ax=True, **kwargs)


def _centers(collection):
    if isinstance(collection, PathCollection):
        return np.asarray(collection.get_offsets())
    return np.array([path.get_extents().get_points().mean(axis=0) for path in collection.get_paths()])


@pytest.mark.parametrize("shape", [None, "circle", "square", "hex"])
@pytest.mark.parametrize("sort_order", [True, False])
@pytest.mark.parametrize("source", ["obs", "X", "layer", "raw"])
def test_continuous_sort_order(shape, sort_order, source):
    values = np.array([8.0, np.nan, 1.0, 4.0, 4.0, -2.0])
    adata = _adata(values)
    kwargs = {"color": "gene", "use_raw": False}
    if source == "obs":
        kwargs["color"] = "value"
    elif source == "layer":
        adata.layers["expression"] = adata.X.copy()
        adata.X[:] = 0
        kwargs["layer"] = "expression"
    elif source == "raw":
        adata.raw = adata.copy()
        adata.X[:] = 0
        kwargs["use_raw"] = True
    coords = adata.obsm["spatial"].copy()
    ax = _plot(adata, shape=shape, sort_order=sort_order, **kwargs)
    try:
        collection = ax.collections[-1]
        expected = [1, 5, 2, 3, 4, 0] if sort_order else list(range(6))
        np.testing.assert_allclose(collection.get_array().filled(np.nan), values[expected])
        np.testing.assert_allclose(_centers(collection), coords[expected], atol=1e-12)
        np.testing.assert_array_equal(adata.obsm["spatial"], coords)
        np.testing.assert_array_equal(adata.obs["value"], values)
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("color", [None, "category"])
@pytest.mark.parametrize("shape", [None, "circle", "square", "hex"])
def test_noncontinuous_preserves_order(color, shape):
    adata = _adata([8, 1, 4])
    adata.obs["category"] = pd.Categorical(["b", "a", "b"])
    ax = _plot(adata, color=color, shape=shape)
    try:
        np.testing.assert_allclose(_centers(ax.collections[0]), adata.obsm["spatial"], atol=1e-12)
    finally:
        plt.close(ax.figure)


@pytest.mark.parametrize("shape", [None, "circle", "square", "hex"])
@pytest.mark.parametrize("sort_order", [None, False])
def test_overlap_pixel(shape, sort_order):
    adata = _adata([8.0, 1.0])
    adata.obsm["spatial"][:] = 0
    fig, ax = plt.subplots(figsize=(2, 2), dpi=100)
    kwargs = {} if sort_order is None else {"sort_order": sort_order}
    try:
        pl.spatial_scatter(
            adata,
            color="value",
            shape=shape,
            img=False,
            img_res_key=None,
            size=400 if shape is None else 0.4,
            ax=ax,
            colorbar=False,
            cmap="viridis",
            **kwargs,
        )
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        fig.canvas.draw()
        x, y = np.rint(ax.transData.transform((0, 0))).astype(int)
        pixels = np.asarray(fig.canvas.buffer_rgba())
        actual = pixels[pixels.shape[0] - 1 - y, x, :3]
        top = 1.0 if sort_order is None else 0.0
        expected = np.array(plt.get_cmap("viridis")(top)[:3]) * 255
        np.testing.assert_allclose(actual, expected, atol=1)
    finally:
        plt.close(fig)


def test_sort_order_signature():
    assert inspect.signature(pl.spatial_scatter).parameters["sort_order"].default is True
    assert "sort_order" not in inspect.signature(pl.spatial_segment).parameters
    assert "sort_order" in pl.spatial_scatter.__doc__


@pytest.mark.parametrize("shape", [None, "circle", "square", "hex"])
@pytest.mark.parametrize("outline", [False, True])
@pytest.mark.parametrize(
    "styles",
    [
        {"edgecolors": ["red", "green", "blue"], "linewidths": [1.0, 2.0, 3.0]},
        {"ec": ["red", "blue"], "lw": [1.0, 2.0], "ls": ["solid", "dashed"]},
        {"edgecolor": (0.2, 0.4, 0.6), "linewidth": 2.0, "linestyle": (0, (2, 3))},
        {"edgecolors": "face", "aa": [True, False, True], "urls": ["a", "b", "c"]},
        {"edgecolors": "none", "linewidths": [2.0], "linestyles": [(0, (1, 2)), (1, (3, 4))]},
        {"edgecolors": [[1, 0, 0, 1], [0, 0, 1, 0.5]], "linewidths": np.array(2.0)},
    ],
)
def test_sort_preserves_point_styles(shape, outline, styles):
    adata = _adata([8.0, 1.0, 4.0])
    original = _plot(adata, color="value", shape=shape, outline=outline, sort_order=False, **styles)
    sorted_ax = _plot(adata, color="value", shape=shape, outline=outline, sort_order=True, **styles)
    try:
        original.figure.canvas.draw()
        sorted_ax.figure.canvas.draw()
        for before, after in zip(original.collections, sorted_ax.collections, strict=True):
            for getter in ["get_edgecolors", "get_linewidths", "get_linestyles", "get_antialiaseds", "get_urls"]:
                expected = getattr(before, getter)()
                actual = getattr(after, getter)()
                if not len(expected):
                    assert len(actual) == 0
                    continue
                for drawn_index, original_index in enumerate([1, 2, 0]):
                    left = actual[drawn_index % len(actual)]
                    right = expected[original_index % len(expected)]
                    if getter == "get_linestyles":
                        assert left[0] == right[0]
                        np.testing.assert_equal(left[1], right[1])
                    else:
                        np.testing.assert_equal(left, right)
    finally:
        plt.close(original.figure)
        plt.close(sorted_ax.figure)


@pytest.mark.parametrize("offsets", [[1.0, 2.0], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
def test_sort_preserves_patch_offsets(offsets):
    adata = _adata([8.0, 1.0, 4.0])
    original = _plot(adata, color="value", shape="circle", sort_order=False, offsets=offsets)
    sorted_ax = _plot(adata, color="value", shape="circle", sort_order=True, offsets=offsets)
    try:
        before = original.collections[-1].get_offsets()
        after = sorted_ax.collections[-1].get_offsets()
        np.testing.assert_array_equal(after, before[[1, 2, 0]] if len(before) > 1 else before)
    finally:
        plt.close(original.figure)
        plt.close(sorted_ax.figure)


@pytest.mark.parametrize("outline", [True, False])
def test_sort_each_library_and_color_after_crop(outline):
    adata = _adata([8.0, 1.0, 4.0, 3.0, 9.0, 2.0])
    adata.obs["library"] = pd.Categorical(["a", "a", "a", "b", "b", "b"])
    adata.obs["other"] = -adata.obs["value"]
    adata.uns["spatial"] = dict.fromkeys(["a", "b"], {"images": {}, "scalefactors": {}})
    axes = _plot(
        adata,
        color=["value", "other"],
        shape="circle",
        library_key="library",
        crop_coord=(0, -1, 4, 1),
        outline=outline,
    )
    try:
        for ax, expected in zip(axes, [[1, 2, 0], [0, 2, 1], [3, 4], [4, 3]], strict=True):
            coords = adata.obsm["spatial"][expected].copy()
            coords[:, 1] += 1
            np.testing.assert_allclose(_centers(ax.collections[-1]), coords, atol=1e-12)
    finally:
        plt.close(axes[0].figure)
