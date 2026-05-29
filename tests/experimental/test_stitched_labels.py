"""Tests for sq.experimental.im.make_stitched_labels."""

from __future__ import annotations

import numpy as np
import pytest

import squidpy as sq


def _qc_and_stitch(sdata, **stitch_kwargs):
    sq.experimental.tl.calculate_tiling_qc(
        sdata,
        labels_key="labels",
        tile_size=200,
        nmads_cut=1.0,
        nmads_smoothed=1.5,
    )
    sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels", **stitch_kwargs)


class TestMakeStitchedLabels:
    def test_creates_new_labels_element(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata)
        assert "labels_stitched" not in sdata.labels
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")
        assert "labels_stitched" in sdata.labels

    def test_original_labels_unchanged(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        original_arr = np.asarray(sdata.labels["labels"].values).copy()
        _qc_and_stitch(sdata)
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")
        after_arr = np.asarray(sdata.labels["labels"].values)
        np.testing.assert_array_equal(original_arr, after_arr)

    def test_remap_unifies_stitched_pieces(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        stitched = adata.obs[adata.obs["is_stitched"].astype(bool)]
        if len(stitched) == 0:
            pytest.skip("no stitched cells in this fixture realisation")

        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")
        new_arr = np.asarray(sdata.labels["labels_stitched"].values)
        old_arr = np.asarray(sdata.labels["labels"].values)

        # Pick one stitched group with >= 2 pieces
        gid = int(stitched["stitch_group_id"].iloc[0])
        pieces = stitched.loc[stitched["stitch_group_id"] == gid, "label_id"].astype(int).tolist()
        assert len(pieces) >= 2

        # All original pixels of those pieces should now carry the group id
        for piece_id in pieces:
            mask = old_arr == piece_id
            assert mask.any()
            assert (new_arr[mask] == gid).all(), f"piece {piece_id} not remapped to {gid}"

    def test_unstitched_pieces_keep_their_id(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata)
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")
        old_arr = np.asarray(sdata.labels["labels"].values)
        new_arr = np.asarray(sdata.labels["labels_stitched"].values)
        # Pixels with label 0 (background) stay 0
        bg = old_arr == 0
        assert (new_arr[bg] == 0).all()
        # Cells whose group_id == label_id are unchanged in the remap
        adata = sdata.tables["labels_qc"]
        unstitched = adata.obs[adata.obs["stitch_group_id"].astype(int) == adata.obs["label_id"].astype(int)]
        # Spot-check the first 5 unstitched
        for lid in unstitched["label_id"].astype(int).iloc[:5]:
            mask = old_arr == lid
            if mask.any():
                assert (new_arr[mask] == lid).all()

    def test_collapsed_table_one_row_per_group(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True)
        assert "labels_stitched_table" in sdata.tables
        agg = sdata.tables["labels_stitched_table"]
        adata = sdata.tables["labels_qc"]
        # Output has one row per unique stitch_group_id (unstitched cells stay
        # as singleton groups, stitched groups collapse to one row).
        n_groups = adata.obs["stitch_group_id"].nunique()
        assert agg.n_obs == n_groups
        for col in ("label_id", "stitch_group_id", "n_pieces", "is_stitched", "stitch_confidence"):
            assert col in agg.obs.columns

    def test_collapsed_table_includes_unstitched_cells(self, sdata_tile_boundary):
        """Both stitched (collapsed) and unstitched (passthrough) rows present."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True)
        agg = sdata.tables["labels_stitched_table"]
        # At least some unstitched cells should be in the output.
        assert (~agg.obs["is_stitched"].astype(bool)).sum() > 0, "expected unstitched rows"
        # The is_stitched column flags which rows are collapsed groups.
        if agg.obs["is_stitched"].astype(bool).sum() > 0:
            assert (agg.obs.loc[agg.obs["is_stitched"].astype(bool), "n_pieces"] >= 2).all()

    def test_merge_strategy_sum_aggregates_numeric_columns(self, sdata_tile_boundary):
        """For a stitched group, a synthetic numeric column should sum across pieces."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        adata.obs["fake_area"] = 100.0
        sdata.tables["labels_qc"] = adata
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True, merge_strategy="sum")
        agg = sdata.tables["labels_stitched_table"]
        stitched = agg.obs[agg.obs["is_stitched"].astype(bool)]
        if len(stitched) == 0:
            pytest.skip("no stitched groups in this realisation")
        # Each stitched group has n_pieces members each contributing 100.
        np.testing.assert_array_equal(
            stitched["fake_area"].to_numpy(),
            stitched["n_pieces"].to_numpy() * 100.0,
        )

    def test_merge_strategy_mean_aggregates_numeric_columns(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        adata.obs["fake_intensity"] = 42.0
        sdata.tables["labels_qc"] = adata
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True, merge_strategy="mean")
        agg = sdata.tables["labels_stitched_table"]
        stitched = agg.obs[agg.obs["is_stitched"].astype(bool)]
        if len(stitched) > 0:
            np.testing.assert_allclose(stitched["fake_intensity"].to_numpy(), 42.0)

    def test_merge_strategy_callable(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        adata.obs["fake_count"] = 1
        sdata.tables["labels_qc"] = adata
        sq.experimental.im.make_stitched_labels(
            sdata,
            labels_key="labels",
            write_table=True,
            merge_strategy=lambda s: len(s),
        )
        agg = sdata.tables["labels_stitched_table"]
        # Callable returns len of group, so fake_count == n_pieces post-merge.
        np.testing.assert_array_equal(
            agg.obs["fake_count"].to_numpy(),
            agg.obs["n_pieces"].to_numpy(),
        )

    def test_group_invariant_columns_take_first(self, sdata_tile_boundary):
        """is_stitched, n_pieces, stitch_confidence are not affected by sum strategy."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata_orig = sdata.tables["labels_qc"]
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True, merge_strategy="sum")
        agg = sdata.tables["labels_stitched_table"]
        # n_pieces should be in {1, 2, 3, 4} -- if "sum" had been applied to it,
        # a 4-piece group would show n_pieces = 16.
        assert (agg.obs["n_pieces"].astype(int) <= 4).all()
        # Members of a stitch group share is_stitched value; collapsed row should match.
        stitched = adata_orig.obs[adata_orig.obs["is_stitched"].astype(bool)]
        for gid in stitched["stitch_group_id"].astype(int).unique():
            row = agg.obs[agg.obs["stitch_group_id"].astype(int) == gid]
            assert len(row) == 1
            assert bool(row["is_stitched"].iloc[0]) is True

    def test_aggregated_table_preserves_qc_columns_and_uns(self, sdata_tile_boundary):
        """The reduced table must keep the QC table's obs columns and uns
        instead of constructing a fresh AnnData from scratch."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        # User adds a custom obs column to simulate downstream annotation.
        adata.obs["my_custom_flag"] = True
        sdata.tables["labels_qc"] = adata
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True)
        agg = sdata.tables["labels_stitched_table"]
        # Original QC obs columns survive
        for col in (
            "max_straight_edge_ratio",
            "cardinal_alignment_score",
            "cut_score",
            "smoothed_cut_score",
            "is_outlier",
            "nhood_outlier_fraction",
            "centroid_y",
            "centroid_x",
            "my_custom_flag",
        ):
            assert col in agg.obs.columns, f"missing preserved column: {col}"
        # Uns surfaces survive (tiling_qc params, tiling_stitch params)
        assert "tiling_qc" in agg.uns
        assert "tiling_stitch" in agg.uns
        # spatialdata_attrs now points at the stitched labels element
        attrs = agg.uns["spatialdata_attrs"]
        assert attrs["region"] == "labels_stitched"
        assert attrs["instance_key"] == "label_id"

    def test_aggregated_table_label_id_matches_new_element_ids(self, sdata_tile_boundary):
        """label_id values in the table must equal the IDs in the new labels
        element (the stitch_group_id values become the new instance keys)."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True)
        agg = sdata.tables["labels_stitched_table"]
        new_arr = np.asarray(sdata.labels["labels_stitched"].values)
        unique_in_image = set(np.unique(new_arr).tolist()) - {0}
        unique_in_table = set(agg.obs["label_id"].astype(int).tolist())
        # Every row in the table must reference an existing instance in the labels element.
        assert unique_in_table.issubset(unique_in_image), f"orphan rows: {unique_in_table - unique_in_image}"

    @pytest.mark.parametrize(
        ("setup", "kwargs", "match"),
        [
            ("qc_only", {"labels_key": "labels"}, "stitch_group_id"),
            ("qc_and_stitch", {"labels_key": "bogus"}, "not found"),
            ("qc_and_stitch", {"labels_key": "labels", "merge_strategy": "bogus"}, "Unknown merge_strategy"),
        ],
        ids=["stitch_not_run", "missing_labels_key", "invalid_merge_strategy"],
    )
    def test_invalid_input_raises(self, sdata_tile_boundary, setup, kwargs, match):
        sdata, _ = sdata_tile_boundary
        if setup == "qc_only":
            sq.experimental.tl.calculate_tiling_qc(sdata, labels_key="labels", tile_size=200)
        else:
            _qc_and_stitch(sdata)
        with pytest.raises(ValueError, match=match):
            sq.experimental.im.make_stitched_labels(sdata, **kwargs)

    def test_idempotent(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata)
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")
        first = np.asarray(sdata.labels["labels_stitched"].values).copy()
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")
        second = np.asarray(sdata.labels["labels_stitched"].values)
        np.testing.assert_array_equal(first, second)

    def test_join_labels_false_keeps_multi_component(self, sdata_tile_boundary):
        from skimage.measure import label as cc_label

        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        stitched = adata.obs[adata.obs["is_stitched"].astype(bool)]
        if len(stitched) == 0:
            pytest.skip("no stitched cells in this fixture realisation")
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", join_labels=False)
        arr = np.asarray(sdata.labels["labels_stitched"].values)
        # At least one stitched group should have >1 connected component
        # (the unjoined behaviour leaves the cut stripe as background).
        any_multi = False
        for gid in stitched["stitch_group_id"].astype(int).unique()[:5]:
            mask = arr == gid
            if mask.any():
                ncc = int(cc_label(mask).max())
                if ncc > 1:
                    any_multi = True
                    break
        assert any_multi, "expected at least one multi-component stitched group with join_labels=False"

    def test_join_labels_true_unifies_components(self, sdata_tile_boundary):
        from skimage.measure import label as cc_label

        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        stitched = adata.obs[adata.obs["is_stitched"].astype(bool)]
        if len(stitched) == 0:
            pytest.skip("no stitched cells in this fixture realisation")
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", join_labels=True)
        arr = np.asarray(sdata.labels["labels_stitched"].values)
        for gid in stitched["stitch_group_id"].astype(int).unique():
            mask = arr == gid
            if not mask.any():
                continue
            ncc = int(cc_label(mask).max())
            assert ncc == 1, f"group {gid} still has {ncc} components after join_labels=True"

    def test_join_labels_does_not_overwrite_other_cells(self, sdata_tile_boundary):
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        adata = sdata.tables["labels_qc"]
        # Snapshot every non-stitched cell's pixel set before joining, then
        # confirm none of those pixels changed identity afterwards.
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", join_labels=False)
        before_arr = np.asarray(sdata.labels["labels_stitched"].values).copy()
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", join_labels=True)
        after_arr = np.asarray(sdata.labels["labels_stitched"].values)
        non_stitched_gids = (
            adata.obs.loc[~adata.obs["is_stitched"].astype(bool), "stitch_group_id"].astype(int).unique()
        )
        for gid in non_stitched_gids[:20]:
            before_mask = before_arr == gid
            if not before_mask.any():
                continue
            # Non-stitched cells must keep all their original pixels.
            assert (after_arr[before_mask] == gid).all(), f"non-stitched cell {gid} was overwritten"


class TestScaleRework:
    """Lock the scale/correctness rework: out-of-range passthrough, lazy join,
    sparse-safe and vectorised aggregation."""

    def test_unmapped_image_label_passes_through(self):
        """A label present in the image but absent from the QC table (e.g. a
        min_area-filtered cell) must survive the LUT remap, not crash (C1)."""
        import pandas as pd
        import xarray as xr

        from squidpy.experimental.im._stitched_labels import _apply_lut, _build_lookup

        obs = pd.DataFrame({"label_id": [1, 2], "stitch_group_id": [1, 1]})
        labels = np.array([[0, 1, 2], [5, 5, 0]], dtype=np.int32)  # label 5 not in table
        lut = _build_lookup(obs, labels.dtype)
        out = np.asarray(_apply_lut(xr.DataArray(labels, dims=("y", "x")), lut))
        assert (out[labels == 5] == 5).all(), "unmapped label 5 was not preserved"
        assert (out[labels == 2] == 1).all(), "label 2 should remap to group 1"

    def test_join_labels_stays_lazy_on_dask_input(self, sdata_tile_boundary):
        import dask.array as da

        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        res = sq.experimental.im.make_stitched_labels(
            sdata, labels_key="labels", join_labels=True, write_table=False, inplace=False
        )
        # The fixture labels are dask-backed; the joined output must remain lazy.
        assert isinstance(res["labels"].data, da.Array), "join_labels materialised the full array"

    @pytest.mark.parametrize("strategy", ["sum", "mean", "first", "max"])
    def test_aggregate_X_sparse_matches_dense(self, strategy):
        from scipy import sparse

        from squidpy.experimental.im._stitched_labels import _aggregate_X

        rng = np.random.default_rng(0)
        dense = rng.integers(0, 5, size=(6, 4)).astype(np.float64)
        groups = [np.array([0, 1, 2]), np.array([3]), np.array([4, 5])]
        out_dense = np.asarray(_aggregate_X(dense, groups, strategy))
        out_sparse = _aggregate_X(sparse.csr_matrix(dense), groups, strategy)
        # Sparse input must NOT be densified into a dense result.
        assert sparse.issparse(out_sparse), "sparse input should yield sparse output"
        np.testing.assert_allclose(out_dense, out_sparse.toarray())
