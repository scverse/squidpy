# Scanpy meets Spatial Transcriptomics (2 challenges)

### Introduction and outline of the challenges

The aim of these challenges is to build preprocessing and analysis tools for spatial modalities: the spatial graph and the tissue image. The topics of the two challenges are the following:

**Spatial graph**

- Build graph from spatial coordinates that account for different neighborhood size (wrt to coordinate distance)
- Tools for neighborhood analysis
- Permutation-based test (e.g. HistoCAT)
- Homophily measure
- Assortativity
- Exploratory analysis on neighbourhood enrichment on mouse brain -*> draft tutorial*

**Tissue image**

- Efficiently access and crop imate tile under spot, accounting for different resolutions (on-disk)?
- Extract image features from image tiles for each spot (scikit-image.features), save them in either new adata, or as obs.
- Assess feasibility of nuclei segmentation modules in scikit-image for H&E and fluorescent images
- Exploratory analysis of extracted image features *→ draft tutorial*

### Logistics

The Hackathon will be organized in an ***agile development format***, where issues that refer to specific tasks are grouped together in **milestones**. What you will find in the repo:

- Skeleton of modules/functions
- Issues with description of the task, reference to code, reference to milestone and (potentially) metrics to evaluate the tool
- Details of datasets: how to access, what's inside etc.

### Anticipated outcomes

We'll try to implement and evaluate as many tools as possible. On the spatial graph side, what's interesting will potentially land to scanpy eventually. On the image side, it will build up as an external package. In both cases, we'll use the tools implemented here to wrap up a collaborative protocol article (F1000/Nature Protocols etc.)