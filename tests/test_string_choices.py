"""String choices are case-insensitive, and an unknown one names the valid options."""

from __future__ import annotations

import numpy as np
import pytest
import spatialdata as sd
from spatialdata.models import Image2DModel, Labels2DModel

import squidpy as sq
from squidpy.experimental.im import StainFit
from squidpy.experimental.im._utils import flatten_channels
from squidpy.gr._clusterers import LeidenClusterer


def _rgb_sdata() -> sd.SpatialData:
    values = np.random.default_rng(0).uniform(40, 200, size=(3, 64, 64)).astype(np.uint8)
    sdata = sd.SpatialData(images={"img": Image2DModel.parse(values, dims=("c", "y", "x"))})
    sdata.labels["img_tissue"] = Labels2DModel.parse(np.ones((64, 64), dtype=np.uint32), dims=("y", "x"))
    return sdata


def _visium(monkeypatch: pytest.MonkeyPatch, sample_id: str) -> str:
    monkeypatch.setattr("squidpy.datasets._datasets.download", lambda name, *_, **__: name)
    return sq.datasets.visium(sample_id)


def _pl_nhood(adata, mode):
    sq.gr.nhood_enrichment(adata, cluster_key="leiden", n_perms=2, rng=0)
    sq.pl.nhood_enrichment(adata, cluster_key="leiden", mode=mode)


def _pl_qc_image(_, metric):
    sdata = _rgb_sdata()
    sq.experimental.im.qc_image(
        sdata, image_key="img", metrics=["tenengrad"], tile_size=(32, 32), tissue_mask_key="img_tissue", progress=False
    )
    sq.experimental.pl.qc_image(sdata, image_key="img", metrics=metric)


def _pl_ripley(adata, mode):
    sq.gr.ripley(adata, cluster_key="leiden", mode="G", n_simulations=2, n_steps=5, rng=0)
    sq.pl.ripley(adata, cluster_key="leiden", mode=mode)


# (data factory name, call(data, value), upper-case spelling, error name in the message)
_CASES = {
    "gr.nhood_enrichment-normalization": (
        "nhood_data",
        lambda a, v: sq.gr.nhood_enrichment(a, cluster_key="leiden", normalization=v, n_perms=2, rng=0, copy=True),
        "TOTAL",
        "`normalization`",
    ),
    "gr.nhood_enrichment-handle_nan": (
        "nhood_data",
        lambda a, v: sq.gr.nhood_enrichment(a, cluster_key="leiden", handle_nan=v, n_perms=2, rng=0, copy=True),
        "ZERO",
        "`handle_nan`",
    ),
    "gr.spatial_autocorr-attr": (
        "nhood_data",
        lambda a, v: sq.gr.spatial_autocorr(a, attr=v, genes=a.var_names[:2], copy=True),
        "x",
        "`attr`",
    ),
    "gr.spatial_autocorr-mode": (
        "nhood_data",
        lambda a, v: sq.gr.spatial_autocorr(a, mode=v, genes=a.var_names[:2], copy=True),
        "GEARY",
        "`SpatialAutocorr`",
    ),
    "gr.ripley-mode": (
        "nhood_data",
        lambda a, v: sq.gr.ripley(a, cluster_key="leiden", mode=v, n_simulations=2, n_steps=5, rng=0, copy=True),
        "l",
        "`RipleyStat`",
    ),
    "gr.spatial_neighbors-coord_type": (
        "nhood_data",
        lambda a, v: sq.gr.spatial_neighbors(a, coord_type=v, copy=True),
        "GENERIC",
        "`CoordType`",
    ),
    "gr.spatial_neighbors_knn-transform": (
        "nhood_data",
        lambda a, v: sq.gr.spatial_neighbors_knn(a, transform=v, copy=True),
        "COSINE",
        "`Transform`",
    ),
    "gr.calculate_niche_cellcharter-aggregation": (
        "dummy_adata2",
        lambda a, v: sq.gr.calculate_niche_cellcharter(a, distance=2, aggregation=v, rng=0, copy=True),
        "MEAN",
        "`aggregation`",
    ),
    "gr.LeidenClusterer-flavor": (
        None,
        lambda _, v: LeidenClusterer(flavor=v, n_neighbors=3).fit(np.random.default_rng(0).normal(size=(20, 3))),
        "IGRAPH",
        "`flavor`",
    ),
    "pl.nhood_enrichment-mode": ("nhood_data", _pl_nhood, "COUNT", "`mode`"),
    "pl.ripley-mode": ("nhood_data", _pl_ripley, "g", "`RipleyStat`"),
    "experimental.pl.qc_image-metrics": (None, _pl_qc_image, "TENENGRAD", "`metrics`"),
    "datasets.visium-sample_id": (
        "monkeypatch",
        _visium,
        "V1_HUMAN_HEART",
        "`sample_id`",
    ),
    "experimental.im.detect_tissue-method": (
        None,
        lambda _, v: sq.experimental.im.detect_tissue(_rgb_sdata(), image_key="img", method=v, inplace=False),
        "OTSU",
        "`method`",
    ),
    "experimental.im.detect_tissue-channel_format": (
        None,
        lambda _, v: flatten_channels(img=_rgb_sdata().images["img"], channel_format=v),
        "RGB",
        "`channel_format`",
    ),
    "experimental.im.qc_image-metrics": (
        None,
        lambda _, v: sq.experimental.im.qc_image(
            _rgb_sdata(), image_key="img", metrics=[v], tile_size=(32, 32), tissue_mask_key="img_tissue", progress=False
        ),
        "TENENGRAD",
        "Unknown metrics",
    ),
    "experimental.im.calculate_image_features-align_mode": (
        None,
        lambda _, v: sq.experimental.im.calculate_image_features(
            _rgb_sdata(),
            image_key="img",
            labels_key="img_tissue",
            features=["skimage:morphology"],
            align_mode=v,
            inplace=False,
        ),
        "RASTERIZE",
        "`align_mode`",
    ),
    "experimental.im.fit_stain_reference-method": (
        None,
        lambda _, v: sq.experimental.im.fit_stain_reference(_rgb_sdata(), image_key="img", method=v),
        "REINHARD",
        "`method`",
    ),
    "experimental.im.StainFit-method": (
        None,
        lambda _, v: StainFit(method=v, mu=np.zeros(3), sigma=np.ones(3)),
        "Reinhard",
        "`method`",
    ),
}


@pytest.mark.parametrize("case", list(_CASES), ids=list(_CASES))
def test_string_choice_is_case_insensitive(request: pytest.FixtureRequest, case: str) -> None:
    data_name, call, upper, name = _CASES[case]
    data = None if data_name is None else request.getfixturevalue(data_name)
    call(data, upper)
    with pytest.raises(ValueError, match=f"{name}.*bogus|bogus.*{name}"):
        call(data, "bogus")


def test_pl_centrality_scores_score_is_case_insensitive(nhood_data) -> None:
    # Unknown names are dropped, not raised on (as before), so this is outside the table above.
    sq.gr.centrality_scores(nhood_data, cluster_key="leiden")
    sq.pl.centrality_scores(nhood_data, cluster_key="leiden", score="DEGREE_CENTRALITY")
    with pytest.raises(ValueError, match="No valid values"):
        sq.pl.centrality_scores(nhood_data, cluster_key="leiden", score="bogus")
