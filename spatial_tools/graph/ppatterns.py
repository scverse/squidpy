import numpy as np
import anndata as ad
from pointpats import ripley

def ripley_fun(coord: np.array, dist: np.array,name: str, support:int):

    if name == "f":
        rip = ripley.k_function(
            coord,
            distances=dist,
            support=support,)
    elif name == "l":
        rip = ripley.l_function(
            coord,
            distances=dist,
            support=support,)
    else:
        print("Function not implemented")
    
    return np.stack([*rip], axis = 0)