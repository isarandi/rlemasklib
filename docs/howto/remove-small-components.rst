Remove Small Components from a Mask
====================================

Segmentation outputs often have noise: small disconnected regions that
aren't part of the main object. This guide shows how to filter them.

By minimum size
---------------

Remove all connected components smaller than a threshold::

    cleaned = mask.remove_small_components(min_size=100)

This keeps only components with at least 100 pixels.

Keep only the largest
---------------------

If you know there's exactly one object, keep only the largest component::

    largest = mask.largest_connected_component()

Connectivity
------------

Both methods accept a ``connectivity`` parameter:

- ``connectivity=4``: only horizontal/vertical neighbors count (default)
- ``connectivity=8``: diagonal neighbors also count

::

    # Stricter connectivity (no diagonals)
    cleaned = mask.remove_small_components(min_size=100, connectivity=4)

Get all components separately
-----------------------------

To inspect or filter components with custom logic::

    components = mask.connected_components()

    # Filter by your own criteria
    kept = [c for c in components if c.area() > 50 and c.bbox()[2] > 10]

    # Recombine
    result = RLEMask.union(kept) if kept else RLEMask.zeros(mask.shape)

Get components with stats
-------------------------

For efficient filtering based on component properties, get stats in a single pass::

    components, stats = mask.connected_components_with_stats()
    areas, bboxes, centroids = stats
    # areas: shape (n,) - pixel count per component
    # bboxes: shape (n, 4) - [x, y, width, height] per component
    # centroids: shape (n, 2) - [x, y] center of mass per component

Filter with a custom function
-----------------------------

Pass a filter function that receives stats arrays and returns a boolean mask.
Only matching components are extracted (more efficient than extracting all
then filtering)::

    # Keep components that are large enough and roughly square
    def my_filter(areas, bboxes, centroids):
        widths, heights = bboxes[:, 2], bboxes[:, 3]
        aspect_ratios = widths / np.maximum(heights, 1)
        return (areas > 100) & (aspect_ratios > 0.5) & (aspect_ratios < 2.0)

    components, stats = mask.connected_components_with_stats(filter_fn=my_filter)

Filter by location
------------------

Keep only components whose centroid is in a region of interest::

    def in_roi(areas, bboxes, centroids):
        x, y = centroids[:, 0], centroids[:, 1]
        return (x > 100) & (x < 500) & (y > 50) & (y < 400)

    components, _ = mask.connected_components_with_stats(filter_fn=in_roi)
