import os

from squidpy._constants._pkg_constants import Key
from squidpy.read import nanostring

# !mkdir test_nanostring
# !wget -P test_nanostring/nanostring_data https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/Lung5_Rep2/Lung5_Rep2+SMI+Flat+data.tar.gz
# !wget -P test_nanostring/nanostring_data https://nanostring-public-share.s3.us-west-2.amazonaws.com/SMI-Compressed/Lung6/Lung6+SMI+Flat+data.tar.gz
# !tar -xzf tutorial_data/nanostring_data/Lung5_Rep2+SMI+Flat+data.tar.gz -C tutorial_data/nanostring_data/
# !tar -xzf tutorial_data/nanostring_data/Lung6+SMI+Flat+data.tar.gz -C tutorial_data/nanostring_data/


# def test_read_nanostring_lung():
#     # Test that reading nanostring files work
#     samples = ["Lung5_Rep2", "Lung6"]
#     nanostring_path = "test_nanostring"
#     for sample in samples:
#         adata = nanostring(
#             path=os.path.join(nanostring_path, sample),
#             counts_file=f"{sample}_exprMat_file.csv",
#             meta_file=f"{sample}_metadata_file.csv",
#             fov_file=f"{sample}_fov_positions_file.csv",
#         )
#         assert "spatial" in adata.uns.keys()
#         assert "spatial" in adata.uns.keys()
#         lib_id = list(adata.uns[Key.uns.spatial].keys())[0]
#         assert Key.uns.image_key in adata.uns[Key.uns.spatial][lib_id].keys()
#         assert Key.uns.scalefactor_key in adata.uns[Key.uns.spatial][lib_id].keys()
#         assert Key.uns.image_res_key in adata.uns[Key.uns.spatial][lib_id][Key.uns.image_key]
#         assert Key.uns.size_key in adata.uns[Key.uns.spatial][lib_id][Key.uns.scalefactor_key]
