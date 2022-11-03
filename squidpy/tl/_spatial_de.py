# from joblib import delayed, Parallel

# from typing import List

# from anndata import AnnData
# import scanpy as sc

# from statsmodels.gam.api import GLMGam, BSplines
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm


# def spatial_de(
#     adata: AnnData,
#     design_matrix_key: str,
#     smooth: str,
#     covariates: str | List[str] | None = None,
#     correction: str = "fdr_bh",
#     restrict_to: str | None = None,
#     copy: bool = False,
# ) -> pd.DataFrame:

#     """Get spatially DEGs."""
#     # write formula
#     formula = "gene ~"
#     if isinstance(covariates, list):
#         for covariate in covariates:
#             formula += covariate + " + "
#         formula.removesuffix("+")
#     else:
#         formula += " " + covariates

#     # fit models in parallel
#     f_test_results = Parallel(n_jobs=-1)(
#         delayed(_build_model)(adata, design_matrix_key, smooth, response_var=gene, formula=formula)
#         for gene in adata.var_names
#     )

#     # get table of DEGs with adjusted p-values and a dictionary of genes and their fitted values
#     DEGs = [x[0] for x in f_test_results]
#     fitted_values = [x[1] for x in f_test_results]
#     results = pd.DataFrame(DEGs, columns=["gene", "p-val", "fc"])
#     fitted_genes = dict(zip(results[["gene"]].to_numpy().ravel(), fitted_values))
#     results = results.sort_values("p-val", ascending=False)
#     vals = results[["p-val"]].values.flatten()
#     adjusted = sm.stats.multipletests(vals, method=correction)
#     results["p-val_adj"] = adjusted[1]

#     # store DEGs
#     adata.uns[design_matrix_key + "_DEGs"] = results

#     # store fitted values
#     adata.uns[design_matrix_key + "_fitted_values"] = pd.DataFrame.from_dict(fitted_genes)


# def _build_model(adata: AnnData, design_matrix_key: str, smooth: str, response_var: str, formula: str):
#     """Model fitting."""
#     design_matrix = adata.obsm[design_matrix_key].copy()
#     design_matrix[response_var] = sc.get.obs_df(adata, response_var).to_numpy()
#     design_matrix = design_matrix.rename(columns={response_var: "gene"})
#     x_spline = design_matrix[[smooth]]
#     bs = BSplines(x_spline, df=6, degree=3)
#     gam_bs = GLMGam.from_formula(formula=formula, data=design_matrix, smoother=bs)
#     res_bs = gam_bs.fit()
#     res_bs.summary()
#     A = np.identity(len(res_bs.params))
#     print(res_bs.params)
#     A = A[1:, :]
#     return ([response_var, res_bs.f_test(A).pvalue, res_bs.params[1]], res_bs.fittedvalues)
