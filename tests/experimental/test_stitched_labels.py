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


class TestReviewFixes:
    """Regression locks for the review findings (each fails on the pre-fix code)."""

    def test_centroid_is_mean_not_sum_and_in_bounds(self, sdata_tile_boundary):
        """M1: QC contract columns must not be summed by the default merge_strategy."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        qc = sdata.tables["labels_qc"].obs
        stitched = qc[qc["is_stitched"].astype(bool)]
        if stitched.empty:
            pytest.skip("no stitched cells in this realisation")
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True, merge_strategy="sum")
        agg = sdata.tables["labels_stitched_table"].obs
        h, w = np.asarray(sdata.labels["labels"].values).shape[-2:]
        gid = int(stitched["stitch_group_id"].iloc[0])
        members = qc[qc["stitch_group_id"].astype(int) == gid]
        assert len(members) >= 2
        row = agg[agg["stitch_group_id"].astype(int) == gid]
        assert len(row) == 1
        # centroid -> mean of pieces (a sum would roughly double it and leave the image)
        np.testing.assert_allclose(row["centroid_y"].iloc[0], members["centroid_y"].mean(), rtol=1e-6)
        np.testing.assert_allclose(row["centroid_x"].iloc[0], members["centroid_x"].mean(), rtol=1e-6)
        assert 0.0 <= row["centroid_y"].iloc[0] <= h
        assert 0.0 <= row["centroid_x"].iloc[0] <= w
        # cut_score -> max of pieces (a sum would push it past its natural range)
        np.testing.assert_allclose(row["cut_score"].iloc[0], members["cut_score"].max(), rtol=1e-6)

    def test_aggregate_X_integer_sum_no_overflow(self):
        """M3: summing an integer .X must not wrap on cast-back to the input dtype."""
        from scipy import sparse

        from squidpy.experimental.im._stitched_labels import _aggregate_X

        X = np.array([[40000], [30000], [20000]], dtype=np.uint16)  # sum 90000 > uint16 max
        groups = [np.array([0, 1, 2])]
        dense = np.asarray(_aggregate_X(X, groups, "sum"))
        assert dense[0, 0] == 90000, "uint16 sum wrapped (90000 % 65536 == 24464)"
        sp_out = _aggregate_X(sparse.csr_matrix(X), groups, "sum")
        assert sp_out.toarray()[0, 0] == 90000

    def test_aggregate_X_callable_applied_to_singletons(self):
        """M5: a callable strategy must be applied to singleton groups too (obs/X parity)."""
        from squidpy.experimental.im._stitched_labels import _aggregate_X

        X = np.array([[10.0], [20.0], [30.0]])
        groups = [np.array([0]), np.array([1, 2])]  # group 0 is a singleton
        out = np.asarray(_aggregate_X(X, groups, lambda s: float(len(s))))
        assert out[0, 0] == 1.0, "callable bypassed for singleton (got the raw row, not len)"
        assert out[1, 0] == 2.0

    def test_int_obs_column_mean_not_truncated(self, sdata_tile_boundary):
        """M6: aggregating an integer obs column with mean must keep the fractional value."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata, min_confidence=0.5)
        qc = sdata.tables["labels_qc"]
        qc.obs["int_feature"] = np.arange(1, qc.n_obs + 1, dtype=np.int64)
        sdata.tables["labels_qc"] = qc
        sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=True, merge_strategy="mean")
        agg = sdata.tables["labels_stitched_table"]
        assert agg.obs["int_feature"].dtype.kind == "f", "mean of an int column was truncated back to int"
        stitched = qc.obs[qc.obs["is_stitched"].astype(bool)]
        if not stitched.empty:
            gid = int(stitched["stitch_group_id"].iloc[0])
            members = qc.obs[qc.obs["stitch_group_id"].astype(int) == gid]
            got = agg.obs.loc[agg.obs["stitch_group_id"].astype(int) == gid, "int_feature"].iloc[0]
            np.testing.assert_allclose(got, members["int_feature"].mean(), rtol=1e-6)

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [("duplicate", "duplicate 'label_id'"), ("zero", "non-positive")],
        ids=["duplicate", "zero"],
    )
    def test_invalid_label_id_raises(self, sdata_tile_boundary, mutate, match):
        """M7: a duplicated / non-positive label_id must raise an actionable error, not mis-map.

        (NaN label_id is also guarded in-code but is unreachable through the public
        path -- SpatialData's TableModel rejects a null instance key on assignment.)
        """
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata)
        qc = sdata.tables["labels_qc"]
        lid = qc.obs["label_id"].to_numpy().copy()
        if mutate == "duplicate":
            lid[1] = lid[0]
        else:
            lid[0] = 0
        qc.obs["label_id"] = lid
        sdata.tables["labels_qc"] = qc
        with pytest.raises(ValueError, match=match):
            sq.experimental.im.make_stitched_labels(sdata, labels_key="labels")

    def test_invalid_merge_strategy_raises_without_table(self, sdata_tile_boundary):
        """M9: merge_strategy is validated eagerly even when write_table=False."""
        sdata, _ = sdata_tile_boundary
        _qc_and_stitch(sdata)
        with pytest.raises(ValueError, match="Unknown merge_strategy"):
            sq.experimental.im.make_stitched_labels(
                sdata, labels_key="labels", write_table=False, merge_strategy="bogus"
            )

    def test_multiscale_output_carries_resolved_scale_transform(self):
        """M2/M4: the stitched element must use the resolved scale's transform, not the
        DataTree base (Identity), and sit at that scale's resolution."""
        import dask.array as da
        import xarray as xr
        from spatialdata import SpatialData
        from spatialdata.models import Labels2DModel
        from spatialdata.transformations import get_transformation

        from squidpy.experimental.utils._labels import resolve_labels_array
        from tests.experimental.conftest import make_tile_boundary_sdata

        base, _ = make_tile_boundary_sdata()
        arr = np.asarray(base.labels["labels"].values)
        ms = Labels2DModel.parse(
            xr.DataArray(da.from_array(arr, chunks=(200, 200)), dims=("y", "x")), scale_factors=[2]
        )
        sdata = SpatialData(images={"image": base.images["image"]}, labels={"labels": ms})
        sq.experimental.tl.calculate_tiling_qc(
            sdata, labels_key="labels", scale="scale1", tile_size=100, nmads_cut=1.0, nmads_smoothed=1.5
        )
        sq.experimental.tl.assign_stitch_groups(sdata, labels_key="labels")
        res = sq.experimental.im.make_stitched_labels(sdata, labels_key="labels", write_table=False, inplace=False)
        out = res["labels"]
        resolved = resolve_labels_array(sdata, "labels", "scale1")

        def _affine(elem):
            t = next(iter(get_transformation(elem, get_all=True).values()))
            return t.to_affine_matrix(("y", "x"), ("y", "x"))

        # output overlays at the resolved scale (scale1 carries a 2x Scale to global)
        np.testing.assert_allclose(_affine(out), _affine(resolved))
        # the bug attached the DataTree base (Identity) transform instead
        assert not np.allclose(_affine(out), _affine(sdata.labels["labels"]))
        assert out.shape[-2:] == resolved.shape[-2:]
