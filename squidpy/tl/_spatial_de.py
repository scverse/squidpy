from anndata import AnnData
import scanpy as sc
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
from typing import List
import pandas as pd
import numpy as np

def spatial_de(adata: AnnData,
                design_matrix_key: str,
                smooth: str,
                covariates: str | List[str] | None = None,
                correction: str = "fdr_bh",
                copy: bool = False) -> pd.DataFrame:
    from joblib import delayed, Parallel
    """Get spatially DEGs."""
    #write formula
    formula = "gene ~"
    if isinstance(covariates, list):
        for covariate in covariates:
            formula += covariate + " + "
        formula.removesuffix("+")
    else:
        formula += " " + covariates

    #fit models in parallel
    f_test_results = Parallel(n_jobs=-1)(delayed(_build_model)
    (adata, design_matrix_key, smooth, response_var = gene, formula = formula, correction = correction) 
    for gene in adata.var_names)
    
    #get table of DEGs with adjusted p-values
    results = pd.DataFrame(f_test_results, columns=['gene','p-val','lfc'])
    results = results.sort_values('p-val', ascending=False)
    vals = results[['p-val']].values.flatten()
    adjusted = sm.stats.multipletests(vals, method=correction)
    results['p-val_adj'] = adjusted[1]
    adata.uns[design_matrix_key + "_DEGs"] = results

def _build_model(adata: AnnData,
                design_matrix_key: str,
                smooth: str,
                response_var: str,
                formula: str,
                correction: str = "fdr_bh"):
    """Model fitting."""
    design_matrix = adata.obsm[design_matrix_key].copy()
    design_matrix[response_var] = sc.get.obs_df(adata, response_var).to_numpy()
    design_matrix = design_matrix.rename(columns={response_var: 'gene'})
    x_spline = design_matrix[[smooth]]
    bs = BSplines(x_spline, df=6, degree=3)
    gam_bs = GLMGam.from_formula(formula=formula, data=design_matrix, smoother=bs)
    res_bs = gam_bs.fit()
    res_bs.summary()
    A = np.identity(len(res_bs.params))
    print(res_bs.params)
    A = A[1:,:]
    return (response_var, res_bs.f_test(A).pvalue, res_bs.params[1])