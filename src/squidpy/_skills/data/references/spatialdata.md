# SpatialData interoperability — Squidpy 1.8.4

Read when a request mentions SpatialData, a SpatialData table, coordinate systems, images/labels/shapes, cell IDs, or mismatched regions.

## Object and region selection

- Inventory `sdata.images`, `sdata.labels`, `sdata.shapes`, and `sdata.tables`. Names like `"image"`, `"cells"`, and `"table"` in tutorials are not conventions to assume.
- Determine which table annotates which region(s), and how observation `instance_id` values correspond to segmentation label IDs or shape IDs. Never assume `.obs` row order corresponds to label IDs.
- Inspect spatial transformations and coordinate-system names before treating physical units as pixels; centroid coordinates depend on the desired coordinate system.
- If the input is AnnData with `adata.obsm["spatial"]`, the graph functions can use the coordinates directly. If it is SpatialData, specify both `table_key` and `elements_to_coordinate_systems` for its annotated Shapes/Labels regions. All annotated regions of the table must be compatible with the mapping.
- Example of a _call shape_ only (replace the keys after inspecting the object):

```python
sq.gr.spatial_neighbors_knn(
    sdata,
    table_key="cell_table",
    elements_to_coordinate_systems={"cell_shapes": "global"},
    n_neighs=6,
)
```

## Shuffled instance IDs in 1.8.4

The SpatialData graph dispatcher can construct coordinates in element order while leaving the table in its original observation order. A graph can have the correct shape and still connect the wrong cells. Verify that each coordinate row belongs to the table row's region and instance ID, rather than validating distances only against the dispatcher's own `.obsm` output.

For a table annotating one 2D shapes region, explicitly align public SpatialData centroids before calling the graph constructor on the AnnData table. The variable names below stand for keys and metadata inspected from the actual object:

```python
from spatialdata import get_centroids

table = sdata.tables[table_key]
attrs = table.uns["spatialdata_attrs"]
instance_ids = table.obs[attrs["instance_key"]]
centroids = get_centroids(sdata.shapes[region_element], coordinate_system=coordinate_system).compute()
table.obsm["spatial"] = centroids.loc[instance_ids, ["x", "y"]].to_numpy()
sq.gr.spatial_neighbors_knn(table, n_neighs=n_neighs)
```

Here `region_element` names the shapes element. This example covers a single 2D region. For multiple regions, align each region separately and join on **both region and instance ID**, because different regions can reuse instance IDs. Preserve observation order and apply the intended library separation when building the graph. Do not replace SpatialData transformations with untransformed shape centroids or discard additional coordinate axes from 3D data.

## Coordinate/data integrity checks

1. Verify table annotations point to existing region elements and selected coordinate systems are supported by their transforms.
2. Verify centroid calculation is performed in the intended coordinate system; graph distance units follow its coordinates.
3. For images and labels, check both pixel-grid alignment and SpatialData transforms. Equal shapes alone do not imply the same physical alignment.
4. For multiscale data, determine whether `"auto"` chooses coarsest or finest for the _specific_ function. There is no universal scale policy: `detect_tissue` / stain fitting choose coarse, stain transform / decompose choose fine.
5. For mutations, preserve originals unless a replacement is explicitly desired; use unique keys and verify `sdata.tables`, `.images`, `.labels`, or `.shapes` after completion.

## Ecosystem boundaries

- SpatialData owns spatial containers and coordinate transformations.
- Squidpy supplies spatial analyses and experimental image / QC algorithms.
- `spatialdata-plot` handles layered rendering of SpatialData image/label/shape elements; `napari-spatialdata` handles interactive exploration.

Sources:
- https://squidpy.readthedocs.io/en/stable/api/squidpy.gr.spatial_neighbors_knn.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.make_tiles.html
- https://squidpy.readthedocs.io/en/stable/api/squidpy.experimental.im.StainFit.transform.html
