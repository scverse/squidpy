from __future__ import annotations

__all__ = ["KNOWN_HASHES"]


KNOWN_HASHES: dict[str, str] = {
    # 10x V1 – V1_Mouse_Kidney
    'https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Mouse_Kidney/V1_Mouse_Kidney_filtered_feature_bc_matrix.h5': 'sha256:5e0b1d1c51c4e8759cd623d212573e1c28daf95d66e0d25a8e4488f6bed3831a',
    'https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Mouse_Kidney/V1_Mouse_Kidney_spatial.tar.gz': 'sha256:91570548eae3d2bcf738af45e9dc463547a01669841db43ff20afb41b7cc0539',
    # 10x V1 – V1_Adult_Mouse_Brain
    'https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_filtered_feature_bc_matrix.h5': 'sha256:eb78379e02dcf48036abf05b67233e73ecb0d880787feb82f76ff16f6ce01eb3',
    'https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_image.tif': 'sha256:39d0a85a7cecb0bde9ad2566260d571bb49834d26fc443cb32b96475f30668b2',
    'https://cf.10xgenomics.com/samples/spatial-exp/1.1.0/V1_Adult_Mouse_Brain/V1_Adult_Mouse_Brain_spatial.tar.gz': 'sha256:46d6b05ba740f232d6bf4b27b9a8846815851e000985fb878f1364bab04e5bd4',
    # 10x V2 - Parent_Visium_Human_Cerebellum
    'https://cf.10xgenomics.com/samples/spatial-exp/1.2.0/Parent_Visium_Human_Cerebellum/Parent_Visium_Human_Cerebellum_filtered_feature_bc_matrix.h5': 'sha256:05c137dd74623e748558c60a99d8e19749cbd073d070ce827aec73cee899f1d0',
    'https://cf.10xgenomics.com/samples/spatial-exp/1.2.0/Parent_Visium_Human_Cerebellum/Parent_Visium_Human_Cerebellum_spatial.tar.gz': 'sha256:7a8a42ad53d93776b7b21b31c3727d76a8ed6c332e2f39b6b056b52ef41eeea0',
    # 10x V3 - Visium_FFPE_Mouse_Brain
    'https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Mouse_Brain/Visium_FFPE_Mouse_Brain_filtered_feature_bc_matrix.h5': 'sha256:f5a5d0fafeab6259ded1c4883b255ef57557b81f32774513594e23a49e8352ce',
    'https://cf.10xgenomics.com/samples/spatial-exp/1.3.0/Visium_FFPE_Mouse_Brain/Visium_FFPE_Mouse_Brain_spatial.tar.gz': 'sha256:e4e1b845fd078946c6f8b61bd8d1927c0ce2395c3730f602cd80ef439d4a9d73',
    # figshare
    'https://ndownloader.figshare.com/files/26098124': 'sha256:39d0a85a7cecb0bde9ad2566260d571bb49834d26fc443cb32b96475f30668b2',
    'https://ndownloader.figshare.com/files/26098328': 'sha256:56d379d96da859ea963c4349bbc8de07da9b68ce133839ebef5fe1b033c9e7bb',
    'https://ndownloader.figshare.com/files/26098364': 'sha256:2929fdd06e32fa25b38493e67f301fc5b22b1a32bfbe48ab7237d8d85fe8982d',
    'https://ndownloader.figshare.com/files/52370645': 'sha256:6f88b1624d072a362cb2b40a12f86b7649d3d2f2cc762dd6be23a078ac3093b6',
}
