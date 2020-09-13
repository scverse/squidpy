# Scanpy meets Spatial Transcriptomics (2 challenges)

## Contributors
David, Louis, Olle, Sabrina, Anna, Sergei, Mohammad, Ignacio, Giovanni, Hannah

## Installation
```
git clone https://github.com/theislab/spatial-tools.git
cd spatial-tools
pip install -e .
```

## Workflow
Now you can find 2 branches `images` and `graph`. Switch to that branch when you have decided your issue: e.g. `git switch graph`.
After than, create a new branch for the specific subtask you decide to tackle: e.g. `git checkout -b perm-test`. When you have something, push it to github:

```
git commit -m "init permutation test"
git push origin perm-test
```

You can then open a PR from github while workin on it, so people can see the code, comment, review etc. Whatever you push, will appear on the open PR. When you are done, assign somebody to review your code.

Before starting a new task, remember to switch to master and fetch and pull
```
git switch master
git fetch
git pull
```

## Introduction and outline of the challenges

The aim of these challenges is to build preprocessing and analysis tools for spatial modalities: the spatial graph and the tissue image. The topics of the two challenges are the following:

### Spatial graph

- Build graph from spatial coordinates that account for different neighborhood size (wrt to coordinate distance)
- Tools for neighborhood enrichment analysis
    - Permutation-based test (e.g. HistoCAT)
    - Assortativity measure
- Exploratory analysis on neighbourhood enrichment on mouse brain *→ draft tutorial*

### Tissue image

- Efficiently access and crop image tile under spot, accounting for different resolutions (on-disk)?
- Extract image features from image tiles for each spot (scikit-image.features), save them in either new adata, or as obs.
- Assess feasibility of nuclei segmentation modules in scikit-image for H&E and fluorescent images
- Exploratory analysis of extracted image features *→ draft tutorial*

## Logistics

The Hackathon is organized in an ***agile development format***, where issues that refer to specific tasks are grouped together in **milestones**. What you will find in the repo:

- Skeleton of modules/functions
- Issues with description of the task, reference to code, reference to milestone and (potentially) metrics to evaluate the tool
- Details of datasets: how to access, what's inside etc.

## Anticipated outcomes

We'll try to implement and evaluate as many tools as possible. On the spatial graph side, what's interesting will potentially land to Scanpy eventually. On the image side, it will build up as an external package. In both cases, we'll use the tools implemented here to wrap up a collaborative protocol article (F1000/Nature Protocols etc.)