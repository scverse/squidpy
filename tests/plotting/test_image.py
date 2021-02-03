from anndata import AnnData
import scanpy as sc

from squidpy.im import ImageContainer
import squidpy as sq


def test_extract(adata: AnnData, cont: ImageContainer, caplog):
    """
    Calculate features and extract columns to obs
    """
    # get obsm
    sq.im.calculate_image_features(adata, cont, features=["summary"])

    # extract columns (default values)
    extr_adata = sq.pl.extract(adata)
    # Test that expected columns exist
    for col in [
        "summary_quantile_0.9_ch_0",
        "summary_quantile_0.5_ch_0",
        "summary_quantile_0.1_ch_0",
        "summary_quantile_0.9_ch_1",
        "summary_quantile_0.5_ch_1",
        "summary_quantile_0.1_ch_1",
        "summary_quantile_0.9_ch_2",
        "summary_quantile_0.5_ch_2",
        "summary_quantile_0.1_ch_2",
    ]:
        assert col in extr_adata.obs.columns

    # get obsm that is a numpy array
    adata.obsm["pca_features"] = sc.pp.pca(adata.obsm["img_features"], n_comps=3)
    # extract columns
    extr_adata = sq.pl.extract(adata, obsm_key="pca_features", prefix="pca_features")
    # Test that expected columns exist
    for col in ["pca_features_0", "pca_features_1", "pca_features_2"]:
        assert col in extr_adata.obs.columns

    # extract multiple obsm at once (no prefix)
    extr_adata = sq.pl.extract(adata, obsm_key=["img_features", "pca_features"])
    # Test that expected columns exist
    for col in [
        "summary_quantile_0.9_ch_0",
        "summary_quantile_0.5_ch_0",
        "summary_quantile_0.1_ch_0",
        "summary_quantile_0.9_ch_1",
        "summary_quantile_0.5_ch_1",
        "summary_quantile_0.1_ch_1",
        "summary_quantile_0.9_ch_2",
        "summary_quantile_0.5_ch_2",
        "summary_quantile_0.1_ch_2",
        "0",
        "1",
        "2",
    ]:
        assert col in extr_adata.obs.columns

    # TODO: test similarly to ligrec
    # currently logging to stderr, and not captured by caplog
    # extract obsm twice and make sure that warnings are issued
    # with caplog.at_level(logging.WARNING):
    #    extr2_adata = sq.pl.extract(extr_adata, obsm_key=['pca_features'])
    #    log = caplog.text
    #    assert "will be overwritten by extract" in log
