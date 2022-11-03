from anndata import AnnData
import plotly.graph_objects as go
import numpy as np

def pl_spatial_de(adata: AnnData):
    adata.uns["design_matrix_DEGs"]['log_pval'] = np.log10(adata.uns["design_matrix_DEGs"]['p-val_adj']) * -1
    adata.uns["design_matrix_DEGs"]["l2fc"] = np.log2(np.abs(adata.uns["design_matrix_DEGs"]['fc']))
    adata.uns["design_matrix_DEGs"]["l2fc"][adata.uns["design_matrix_DEGs"]['fc'] < 0] = adata.uns["design_matrix_DEGs"]["l2fc"][adata.uns["design_matrix_DEGs"]['fc'] < 0]*-1
    #adata.uns["design_matrix_DEGs"]['lfc'] = np.log2(adata.uns["design_matrix_DEGs"]['fc'] )
    fig = go.Figure()
    trace1 = go.Scatter(
    x=adata.uns["design_matrix_DEGs"]['fc'],
    y=adata.uns["design_matrix_DEGs"]['log_pval'],
    mode='markers',
    hovertext=list(adata.uns["design_matrix_DEGs"]['gene'])
    )
    fig.add_trace(trace1)
    fig.update_layout(title='Volcano plot for spatial DEGs')
    fig.show()