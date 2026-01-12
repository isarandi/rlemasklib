Convert Between Polygons and Masks
===================================

Polygons and RLE masks are two common representations for regions.
Here's how to convert between them.

Polygon to mask
---------------

::

    import numpy as np
    from rlemasklib import RLEMask

    # Polygon as Nx2 array of (x, y) vertices
    polygon = np.array([[10, 10], [100, 10], [100, 80], [10, 80]])

    mask = RLEMask.from_polygon(polygon, imshape=(480, 640))

Multiple polygons (with holes)
------------------------------

For a polygon with holes, pass a list of vertex arrays.
The first is the outer boundary, the rest are holes::

    outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
    hole = np.array([[30, 30], [70, 30], [70, 70], [30, 70]])

    mask = RLEMask.from_polygon([outer, hole], imshape=(200, 200))

Union of separate polygons
--------------------------

For multiple disconnected regions, create masks and combine::

    polygons = [poly1, poly2, poly3]
    masks = [RLEMask.from_polygon(p, imshape) for p in polygons]
    combined = RLEMask.union(masks)

From shapely geometry
---------------------

::

    import shapely

    # Shapely Polygon
    shape = shapely.Polygon([(10, 10), (100, 10), (100, 80), (10, 80)])
    mask = RLEMask.from_polygon(np.array(shape.exterior.coords), imshape)

    # MultiPolygon
    masks = [RLEMask.from_polygon(np.array(p.exterior.coords), imshape)
             for p in multi_polygon.geoms]
    combined = RLEMask.union(masks)

COCO format
-----------

COCO stores polygons as flat lists ``[x1, y1, x2, y2, ...]``::

    # From COCO polygon
    coco_poly = [10, 10, 100, 10, 100, 80, 10, 80]
    polygon = np.array(coco_poly).reshape(-1, 2)
    mask = RLEMask.from_polygon(polygon, imshape)
