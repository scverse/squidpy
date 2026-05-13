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
