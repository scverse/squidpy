import pandas as pd
import anndata as ad
import os


def read_seqfish(base_path: str, dataset: str):

    if dataset == "ob":
        counts_path = os.path.join(base_path, "sourcedata", "ob_counts.csv")
        clusters_path = os.path.join(base_path, "OB_cell_type_annotations.csv")
        centroids_path = os.path.join(base_path, "sourcedata", "ob_cellcentroids.csv")
    elif dataset == "svz":
        counts_path = os.path.join(base_path, "sourcedata", "cortex_svz_counts.csv")
        clusters_path = os.path.join(base_path, "cortex_svz_cell_type_annotations.csv")
        centroids_path = os.path.join(
            base_path, "sourcedata", "cortex_svz_cellcentroids.csv"
        )
    else:
        print("Dataset not available")

    counts = pd.read_csv(counts_path)
    clusters = pd.read_csv(clusters_path)
    centroids = pd.read_csv(centroids_path)

    adata = ad.AnnData(counts, obs=pd.concat([clusters, centroids], axis=1))
    adata.obsm["spatial"] = adata.obs[["X", "Y"]].to_numpy()

    adata.obs["louvain"] = pd.Categorical(adata.obs["louvain"])

    return adata
