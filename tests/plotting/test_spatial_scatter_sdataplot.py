"""Smoke tests for the spatialdata-plot delegation pipeline.

Covers the three happy paths identified in plans/delegate-plots-to-sdata-plot.md:
- Path 1: Visium spots over H&E, categorical coloring, single + multi-library.
- Path 2: Visium spots over H&E, continuous gene-expression coloring, N-gene grids.
- Path 3: Segmentation masks colored by cell type (MIBI-TOF-style).
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
from anndata import AnnData
from matplotlib.figure import Figure

from squidpy.pl._sdata_delegation import (
    _spatial_scatter_via_sdata_plot,
    _spatial_segment_via_sdata_plot,
)
from squidpy.pl._sdata_delegation._capture import (
    capture_scatter_intent,
    capture_segment_intent,
)

matplotlib.use("Agg")


@pytest.fixture()
def adata_hne_with_cluster(adata_hne: AnnData) -> AnnData:
    a = adata_hne.copy()
    a.obs["cluster_path1"] = (a.obs["array_col"] > a.obs["array_col"].median()).astype(str).astype("category")
    return a


@pytest.fixture()
def adata_hne_concat_with_cluster(adata_hne_concat: AnnData) -> AnnData:
    a = adata_hne_concat.copy()
    a.obs["cluster_path1"] = (a.obs["array_col"] > a.obs["array_col"].median()).astype(str).astype("category")
    return a


class TestCaptureIntent:
    def test_single_library_resolved_from_uns(self, adata_hne_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1")
        assert intent.data.library_ids == ("V1_Adult_Mouse_Brain",)
        assert len(intent.panels) == 1
        assert intent.panels[0].color == "cluster_path1"
        assert intent.data.element_kind == "shapes"
        assert intent.data.needs_image is True

    def test_multi_library_via_library_key(self, adata_hne_concat_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(adata_hne_concat_with_cluster, color="cluster_path1", library_key="library_id")
        assert set(intent.data.library_ids) == {"V1_Adult_Mouse_Brain", "V2_Adult_Mouse_Brain"}
        assert len(intent.panels) == 2

    def test_no_color_is_allowed(self, adata_hne_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(adata_hne_with_cluster)
        assert intent.panels[0].color is None

    def test_multi_color_expands_panels(self, adata_hne_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(adata_hne_with_cluster, color=["a", "b", "c"])
        assert len(intent.panels) == 3
        assert tuple(p.color for p in intent.panels) == ("a", "b", "c")

    def test_panel_iteration_order_library_first(self, adata_hne_concat_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(
            adata_hne_concat_with_cluster,
            color=["g1", "g2"],
            library_key="library_id",
            library_first=True,
        )
        assert len(intent.panels) == 4
        # library_first=True: V1, V1, V2, V2 with colors g1, g2, g1, g2
        first_lib_colors = [p.color for p in intent.panels if p.library_id == intent.data.library_ids[0]]
        assert first_lib_colors == ["g1", "g2"]

    def test_panel_iteration_order_color_first(self, adata_hne_concat_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(
            adata_hne_concat_with_cluster,
            color=["g1", "g2"],
            library_key="library_id",
            library_first=False,
        )
        assert len(intent.panels) == 4
        # library_first=False: g1/V1, g1/V2, g2/V1, g2/V2
        first_two = [(p.library_id, p.color) for p in intent.panels[:2]]
        assert {p[1] for p in first_two} == {"g1"}

    def test_unsupported_kwarg_rejected(self, adata_hne_with_cluster: AnnData) -> None:
        with pytest.raises(NotImplementedError, match="does not yet support"):
            capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", some_future_kwarg=True)

    def test_legend_loc_on_data_deprecated(self, adata_hne_with_cluster: AnnData) -> None:
        with pytest.warns(DeprecationWarning, match="on data"):
            capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", legend_loc="on data")

    def test_size_per_library_sequence(self, adata_hne_concat_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(
            adata_hne_concat_with_cluster,
            color="cluster_path1",
            library_key="library_id",
            size=[0.5, 1.5],
        )
        sizes_by_lib = {p.library_id: p.size for p in intent.panels}
        assert sizes_by_lib == {"V1_Adult_Mouse_Brain": 0.5, "V2_Adult_Mouse_Brain": 1.5}

    def test_size_scalar_broadcasts(self, adata_hne_concat_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(
            adata_hne_concat_with_cluster,
            color="cluster_path1",
            library_key="library_id",
            size=0.75,
        )
        assert all(p.size == 0.75 for p in intent.panels)

    def test_size_wrong_length_rejected(self, adata_hne_concat_with_cluster: AnnData) -> None:
        with pytest.raises(ValueError, match="size"):
            capture_scatter_intent(
                adata_hne_concat_with_cluster,
                color="cluster_path1",
                library_key="library_id",
                size=[0.5, 0.5, 0.5],
            )

    def test_palette_as_colormap_routes_to_cmap(self, adata_hne_with_cluster: AnnData) -> None:
        from matplotlib.colors import ListedColormap

        palette = ListedColormap(["#ff0000", "#00ff00", "#0000ff"])
        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", palette=palette)
        # Colormap routes to cmap; palette stays None so sdata-plot doesn't require groups.
        assert intent.render.palette is None
        assert isinstance(intent.render.cmap, ListedColormap)

    def test_palette_as_string_list_wraps_as_cmap(self, adata_hne_with_cluster: AnnData) -> None:
        from matplotlib.colors import ListedColormap

        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", palette=["#aabbcc", "#ddeeff"])
        assert intent.render.palette is None
        assert isinstance(intent.render.cmap, ListedColormap)

    def test_palette_dict_keeps_palette(self, adata_hne_with_cluster: AnnData) -> None:
        palette = {"True": "#ff0000", "False": "#0000ff"}
        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", palette=palette)
        assert intent.render.palette == palette
        assert intent.render.groups == ("True", "False")

    def test_vmin_vmax_folded_into_norm(self, adata_hne_with_cluster: AnnData) -> None:
        from matplotlib.colors import Normalize

        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", vmin=0.0, vmax=5.0)
        assert isinstance(intent.render.norm, Normalize)
        assert intent.render.norm.vmin == 0.0
        assert intent.render.norm.vmax == 5.0

    def test_vcenter_uses_twoslope(self, adata_hne_with_cluster: AnnData) -> None:
        from matplotlib.colors import TwoSlopeNorm

        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", vmin=-1.0, vmax=1.0, vcenter=0.0)
        assert isinstance(intent.render.norm, TwoSlopeNorm)

    def test_norm_and_vmin_conflict_rejected(self, adata_hne_with_cluster: AnnData) -> None:
        from matplotlib.colors import Normalize

        with pytest.raises(ValueError, match="not both"):
            capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", norm=Normalize(0, 1), vmin=0)

    def test_shape_none_routes_to_points(self, adata_hne_with_cluster: AnnData) -> None:
        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", shape=None)
        assert intent.data.element_kind == "points"


class TestRender:
    def test_single_library_renders_one_panel(self, adata_hne_with_cluster: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1")
        assert isinstance(fig, Figure)
        assert len(fig.axes) >= 1  # at least the plot axis; legend axes are extra
        plt.close(fig)

    def test_multi_library_renders_two_panels(self, adata_hne_concat_with_cluster: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(
            adata_hne_concat_with_cluster, color="cluster_path1", library_key="library_id"
        )
        assert isinstance(fig, Figure)
        panel_axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        assert len(panel_axes) == 2
        plt.close(fig)

    def test_no_image_renders_only_shapes(self, adata_hne_with_cluster: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1", img=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_return_ax_returns_axes(self, adata_hne_with_cluster: AnnData) -> None:
        result = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1", return_ax=True)
        from matplotlib.axes import Axes

        assert isinstance(result, Axes)
        plt.close("all")

    def test_palette_dict_applied(self, adata_hne_concat_with_cluster: AnnData) -> None:
        palette = {"True": "#ff0000", "False": "#0000ff"}
        fig = _spatial_scatter_via_sdata_plot(
            adata_hne_concat_with_cluster,
            color="cluster_path1",
            library_key="library_id",
            palette=palette,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestConnectivityEdges:
    @pytest.fixture()
    def adata_hne_with_neighbors(self, adata_hne: AnnData) -> AnnData:
        from squidpy.gr import spatial_neighbors

        a = adata_hne.copy()
        spatial_neighbors(a)
        a.obs["cluster_path1"] = (a.obs["array_col"] > a.obs["array_col"].median()).astype(str).astype("category")
        return a

    def test_capture_sets_needs_graph(self, adata_hne_with_neighbors: AnnData) -> None:
        intent = capture_scatter_intent(
            adata_hne_with_neighbors, color="cluster_path1", connectivity_key="spatial_connectivities"
        )
        assert intent.data.needs_graph is True
        assert intent.data.graph_layer == "spatial_connectivities"

    def test_no_connectivity_means_no_graph(self, adata_hne_with_neighbors: AnnData) -> None:
        intent = capture_scatter_intent(adata_hne_with_neighbors, color="cluster_path1")
        assert intent.data.needs_graph is False

    def test_edges_render_single_library(self, adata_hne_with_neighbors: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(
            adata_hne_with_neighbors,
            color="cluster_path1",
            connectivity_key="spatial_connectivities",
            img=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_edges_with_custom_width_color(self, adata_hne_with_neighbors: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(
            adata_hne_with_neighbors,
            color="cluster_path1",
            connectivity_key="spatial_connectivities",
            edges_width=2.0,
            edges_color="red",
            img=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPath2Continuous:
    def test_single_gene_renders(self, adata_hne: AnnData) -> None:
        gene = adata_hne.var_names[0]
        fig = _spatial_scatter_via_sdata_plot(adata_hne, color=gene, cmap="viridis")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multi_gene_grid_panels(self, adata_hne: AnnData) -> None:
        genes = list(adata_hne.var_names[:3])
        fig = _spatial_scatter_via_sdata_plot(adata_hne, color=genes, cmap="viridis")
        assert isinstance(fig, Figure)
        plot_axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        assert len(plot_axes) == 3
        plt.close(fig)

    def test_multi_gene_multi_library_grid(self, adata_hne_concat: AnnData) -> None:
        genes = list(adata_hne_concat.var_names[:2])
        fig = _spatial_scatter_via_sdata_plot(adata_hne_concat, color=genes, library_key="library_id", cmap="viridis")
        assert isinstance(fig, Figure)
        plot_axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        assert len(plot_axes) == 4  # 2 libraries x 2 genes
        plt.close(fig)

    def test_vmin_vmax_applied_at_render(self, adata_hne: AnnData) -> None:
        gene = adata_hne.var_names[0]
        fig = _spatial_scatter_via_sdata_plot(adata_hne, color=gene, vmin=0.0, vmax=2.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_layer_passthrough(self, adata_hne: AnnData) -> None:
        a = adata_hne.copy()
        a.layers["scaled"] = a.X.copy()
        gene = a.var_names[0]
        fig = _spatial_scatter_via_sdata_plot(a, color=gene, layer="scaled")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPath3Segmentation:
    @pytest.fixture()
    def mibitof(self) -> AnnData:
        import squidpy as sq

        # Function-scoped + copy so tests that mutate obs (e.g. adding _sq_region via the
        # adapter) don't leak state into siblings.
        return sq.datasets.mibitof().copy()

    def test_capture_requires_seg_cell_id(self, mibitof: AnnData) -> None:
        with pytest.raises(TypeError):
            capture_segment_intent(mibitof)  # type: ignore[call-arg]

    def test_capture_rejects_seg_contourpx_1(self, mibitof: AnnData) -> None:
        with pytest.raises(ValueError, match="seg_contourpx=1"):
            capture_segment_intent(mibitof, seg_cell_id="cell_id", seg_contourpx=1)

    def test_capture_element_kind_is_labels(self, mibitof: AnnData) -> None:
        intent = capture_segment_intent(mibitof, seg_cell_id="cell_id", color="Cluster")
        assert intent.data.element_kind == "labels"
        assert intent.data.seg_cell_id == "cell_id"

    def test_single_library_segment_renders(self, mibitof: AnnData) -> None:
        a = mibitof[mibitof.obs["library_id"] == "point16"].copy()
        fig = _spatial_segment_via_sdata_plot(a, seg_cell_id="cell_id", color="Cluster")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_multi_library_segment_renders(self, mibitof: AnnData) -> None:
        fig = _spatial_segment_via_sdata_plot(mibitof, seg_cell_id="cell_id", color="Cluster", library_key="library_id")
        assert isinstance(fig, Figure)
        plot_axes = [ax for ax in fig.axes if ax.get_subplotspec() is not None]
        assert len(plot_axes) == 3
        plt.close(fig)

    def test_seg_contourpx_passthrough(self, mibitof: AnnData) -> None:
        a = mibitof[mibitof.obs["library_id"] == "point16"].copy()
        fig = _spatial_segment_via_sdata_plot(a, seg_cell_id="cell_id", color="Cluster", seg_contourpx=3)
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestWiredKwargs:
    """M1: kwargs previously captured-then-dropped now produce an observable effect."""

    def _panel_ax(self, fig: Figure):
        return next(ax for ax in fig.axes if ax.get_subplotspec() is not None)

    def test_save_writes_file(self, adata_hne_with_cluster: AnnData, tmp_path) -> None:
        out = tmp_path / "scatter.png"
        fig = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1", save=str(out))
        assert out.exists() and out.stat().st_size > 0
        plt.close(fig)

    def test_colorbar_toggle(self, adata_hne: AnnData) -> None:
        gene = adata_hne.var_names[0]
        fig_on = _spatial_scatter_via_sdata_plot(adata_hne, color=gene, colorbar=True)
        fig_off = _spatial_scatter_via_sdata_plot(adata_hne, color=gene, colorbar=False)
        # continuous color: colorbar=True adds a dedicated colorbar axes, False does not.
        assert len(fig_on.axes) > len(fig_off.axes)
        plt.close(fig_on)
        plt.close(fig_off)

    def test_legend_toggle(self, adata_hne_with_cluster: AnnData) -> None:
        fig_on = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1")
        fig_off = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1", legend_loc=None)
        assert self._panel_ax(fig_on).get_legend() is not None
        assert self._panel_ax(fig_off).get_legend() is None
        plt.close(fig_on)
        plt.close(fig_off)

    def test_axis_label_sets_labels(self, adata_hne_with_cluster: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1", axis_label=["myX", "myY"])
        ax = self._panel_ax(fig)
        assert ax.get_xlabel() == "myX"
        assert ax.get_ylabel() == "myY"
        plt.close(fig)

    def test_img_channel_and_alpha_render(self, adata_hne_with_cluster: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(
            adata_hne_with_cluster, color="cluster_path1", img_channel=0, img_alpha=0.5
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_edges_kwargs_valid(self, adata_hne: AnnData) -> None:
        from squidpy.gr import spatial_neighbors

        a = adata_hne.copy()
        spatial_neighbors(a)
        a.obs["cluster_path1"] = (a.obs["array_col"] > a.obs["array_col"].median()).astype(str).astype("category")
        fig = _spatial_scatter_via_sdata_plot(
            a,
            color="cluster_path1",
            connectivity_key="spatial_connectivities",
            edges_kwargs={"edge_alpha": 0.5},
            img=False,
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_edges_kwargs_unknown_raises(self, adata_hne: AnnData) -> None:
        from squidpy.gr import spatial_neighbors

        a = adata_hne.copy()
        spatial_neighbors(a)
        a.obs["cluster_path1"] = (a.obs["array_col"] > a.obs["array_col"].median()).astype(str).astype("category")
        with pytest.raises(NotImplementedError, match="edges_kwargs"):
            _spatial_scatter_via_sdata_plot(
                a,
                color="cluster_path1",
                connectivity_key="spatial_connectivities",
                edges_kwargs={"bogus_key": 1},
                img=False,
            )

    def test_wspace_hspace_accepted(self, adata_hne_with_cluster: AnnData) -> None:
        fig = _spatial_scatter_via_sdata_plot(
            adata_hne_with_cluster, color=["cluster_path1", "cluster_path1"], wspace=0.4, hspace=0.3
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_scale_factor_accepted_and_stored(self, adata_hne_with_cluster: AnnData) -> None:
        # previously rejected via **unsupported; now an image-scalef override (V1)
        intent = capture_scatter_intent(adata_hne_with_cluster, color="cluster_path1", scale_factor=2.0)
        assert intent.data.scale_factor == 2.0
        fig = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1", scale_factor=2.0)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_use_raw_default_matches_legacy(self, adata_hne: AnnData) -> None:
        """Default (use_raw=None) plots raw counts when adata.raw exists, like legacy;
        use_raw=False plots .X. Guards against a silent value-source change under the flag."""
        assert adata_hne.raw is not None
        gene = adata_hne.var_names[0]

        def _color_vmax(fig: Figure) -> float:
            vs = [
                coll.norm.vmax
                for ax in fig.axes
                for coll in ax.collections
                if coll.norm is not None and coll.norm.vmax is not None
            ]
            return max(vs)

        fig_default = _spatial_scatter_via_sdata_plot(adata_hne, color=gene, img=False)
        fig_x = _spatial_scatter_via_sdata_plot(adata_hne, color=gene, img=False, use_raw=False)
        # raw counts have a larger dynamic range than normalized .X for this gene
        assert _color_vmax(fig_default) > _color_vmax(fig_x)
        plt.close(fig_default)
        plt.close(fig_x)


class TestSpatialDataNativeInput:
    """M2/M3: render directly from a user's SpatialData, no AnnData shim."""

    @pytest.fixture()
    def sdata_visium_like(self):
        import anndata as ad
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        from shapely.geometry import Point
        from spatialdata import SpatialData
        from spatialdata.models import Image2DModel, ShapesModel, TableModel
        from spatialdata.transformations import Identity, set_transformation

        cs = "lib1"
        n = 20
        rng = np.random.default_rng(0)
        xy = rng.uniform(5, 95, size=(n, 2))
        spots = ShapesModel.parse(gpd.GeoDataFrame({"radius": np.full(n, 2.0)}, geometry=[Point(*p) for p in xy]))
        set_transformation(spots, Identity(), to_coordinate_system=cs)
        img = Image2DModel.parse(np.zeros((3, 100, 100), dtype=np.float32), dims=("c", "y", "x"))
        set_transformation(img, Identity(), to_coordinate_system=cs)
        obs = pd.DataFrame(
            {
                "region": pd.Categorical(["spots"] * n),
                "inst": np.arange(n),
                "ct": pd.Categorical(["a", "b"] * (n // 2)),
                "score": rng.random(n),
            }
        )
        adata = ad.AnnData(X=np.zeros((n, 3), dtype=np.float32), obs=obs)
        tab = TableModel.parse(adata, region="spots", region_key="region", instance_key="inst")
        return SpatialData(images={"he": img}, shapes={"spots": spots}, tables={"table": tab})

    def test_categorical_renders(self, sdata_visium_like) -> None:
        fig = _spatial_scatter_via_sdata_plot(sdata_visium_like, color="ct", library_id="lib1")
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_continuous_renders(self, sdata_visium_like) -> None:
        fig = _spatial_scatter_via_sdata_plot(sdata_visium_like, color="score", library_id="lib1", img=False)
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_use_raw_rejected(self, sdata_visium_like) -> None:
        with pytest.raises(ValueError, match="use_raw"):
            _spatial_scatter_via_sdata_plot(sdata_visium_like, color="ct", library_id="lib1", use_raw=True)

    def test_library_key_rejected(self, sdata_visium_like) -> None:
        with pytest.raises(ValueError, match="library_key"):
            _spatial_scatter_via_sdata_plot(sdata_visium_like, color="ct", library_key="foo")

    def test_ambiguous_shapes_raises(self, sdata_visium_like) -> None:
        # add a second shapes element to the same coordinate system -> ambiguous without shapes_layer
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Point
        from spatialdata.models import ShapesModel
        from spatialdata.transformations import Identity, set_transformation

        extra = ShapesModel.parse(
            gpd.GeoDataFrame({"radius": np.full(3, 1.0)}, geometry=[Point(i, i) for i in range(3)])
        )
        set_transformation(extra, Identity(), to_coordinate_system="lib1")
        sdata_visium_like.shapes["spots2"] = extra
        with pytest.raises(ValueError, match="Multiple shapes"):
            _spatial_scatter_via_sdata_plot(sdata_visium_like, color="ct", library_id="lib1", img=False)

    def test_shapes_layer_disambiguates(self, sdata_visium_like) -> None:
        import geopandas as gpd
        import numpy as np
        from shapely.geometry import Point
        from spatialdata.models import ShapesModel
        from spatialdata.transformations import Identity, set_transformation

        extra = ShapesModel.parse(
            gpd.GeoDataFrame({"radius": np.full(3, 1.0)}, geometry=[Point(i, i) for i in range(3)])
        )
        set_transformation(extra, Identity(), to_coordinate_system="lib1")
        sdata_visium_like.shapes["spots2"] = extra
        fig = _spatial_scatter_via_sdata_plot(
            sdata_visium_like, color="ct", library_id="lib1", img=False, shapes_layer="spots"
        )
        assert isinstance(fig, Figure)
        plt.close(fig)

    def test_anndata_input_deprecated(self, adata_hne_with_cluster: AnnData) -> None:
        with pytest.warns(DeprecationWarning, match="deprecated"):
            fig = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1")
        plt.close(fig)

    @pytest.mark.parametrize("use_sdata", [False, True])
    def test_render_parametrized_over_input_type(
        self, use_sdata: bool, adata_hne_with_cluster: AnnData, sdata_visium_like
    ) -> None:
        """Both input types share a categorical-render assertion (W4.3)."""
        if use_sdata:
            fig = _spatial_scatter_via_sdata_plot(sdata_visium_like, color="ct", library_id="lib1")
        else:
            with pytest.warns(DeprecationWarning):
                fig = _spatial_scatter_via_sdata_plot(adata_hne_with_cluster, color="cluster_path1")
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPublicAPIFlag:
    """The SQUIDPY_USE_SDATAPLOT flag routes the public sq.pl entrypoint through delegation."""

    def test_public_spatial_scatter_routes_and_warns(self, adata_hne_with_cluster: AnnData, monkeypatch) -> None:
        import squidpy as sq

        monkeypatch.setenv("SQUIDPY_USE_SDATAPLOT", "1")
        with pytest.warns(DeprecationWarning, match="deprecated"):
            fig = sq.pl.spatial_scatter(adata_hne_with_cluster, color="cluster_path1")
        assert isinstance(fig, Figure)
        plt.close(fig)
