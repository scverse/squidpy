import numpy as np
import pandas as pd
import anndata as ad
from pointpats import ripley

def ripley_c(adata: ad.AnnData, dist_name: str, cluster_key: str, r_name: str, support: int):

    df_lst = []
    for c in adata.obs[cluster_key].unique():
        coord = adata[adata.obs[cluster_key]==c].obsm["spatial"]
        adj = adata[adata.obs[cluster_key]==c].obsp[dist_name].todense()

        rip = _ripley_fun(coord, adj, name=r_name, support=support)
        df_rip = pd.DataFrame(rip)
        df_rip.columns = ["distance",f"ripley_{r_name}"]
        df_rip[cluster_key]=c
        df_lst.append(df_rip)

    return pd.concat(df_lst,axis=0)



def _ripley_fun(coord: np.array, dist: np.array, name: str, support: int):

    if name == "k":
        rip = ripley.k_function(coord, distances=dist, support=support,)
    elif name == "l":
        rip = ripley.l_function(coord, distances=dist, support=support,)
    else:
        print("Function not implemented")

    return np.stack([rip[0], np.squeeze(np.array(rip[1]), axis=1)], axis=1)
