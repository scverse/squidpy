# Roadmap for Spatial Tools Package & Publication
The aim is to build a modular and flexible package for doing analysis of spatial transcriptomics data.
This package extends scanpy by 1. brining features from the tissue image in adata and 2. calculating statistics of the spatial graph using the transcriptome (and potentially imaging features).

In detail, spatial-tools will have the following functionalities:
1. extract features from tissue image in `obs x features` matrix in adata. These features are calculated from the tissue image or the cell-segmented tissue image using crops centered around each transcriptomics observation (e.g. a spot in 10x). They summarize tissue information and bring it together with the transcriptomics information
2. calculate statistics from the spatial graph. These include neighborhood enrichment analysis tools like a permuation-based test and assortativity measures.
In addition, all these functionalities will be documented and motivated in exploratory analysis notebooks to help users to choose between different parameters.
New code should contain unit-tests, and scanpy-like docstrings for each function.

## Timeline

### Due 8.10.
- [ ] clean up + unit tests.

### Open
- [ ] implement different image features
- [ ] evaluation of features
- [ ] robust cell segmentation from h&e and fluorescence and calculation of features from cell segmentation
- [ ] speed up calculation of features (parallelize?)
- [ ] prepare tutorial of feature extraction & analysis
- [ ] ...
