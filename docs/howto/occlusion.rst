Compute Occlusion Between Objects
==================================

When objects overlap in an image, compute which parts are visible
and how much is occluded.

Occlusion fraction
------------------

Given an object's full mask and what's blocking it::

    # Full extent of the object (if nothing was in front)
    instance_mask = ...

    # Mask of everything in front of this object
    occluder_mask = ...

    # Visible portion
    visible_mask = instance_mask - occluder_mask

    # Fraction occluded
    if instance_mask.area() > 0:
        occ_fraction = 1 - visible_mask.area() / instance_mask.area()
    else:
        occ_fraction = 1.0

Layer multiple objects
----------------------

When processing objects sorted by depth (back to front)::

    objects = sorted(objects, key=lambda o: o.depth, reverse=True)

    occluder_mask = RLEMask.zeros(image_shape)

    for obj in objects:
        # Visible = full mask minus occluders
        obj.visible_mask = obj.instance_mask - occluder_mask

        # Bounding box of visible portion
        obj.visible_bbox = obj.visible_mask.bbox()

        # This object now occludes things behind it
        occluder_mask |= obj.instance_mask

Modal vs amodal masks
---------------------

- **Amodal mask**: full object extent (including hidden parts)
- **Modal mask**: only the visible portion

::

    amodal_mask = full_segmentation
    modal_mask = amodal_mask - occluder_mask

    # Or compute modal from instance masks
    modal_masks = []
    accumulated = RLEMask.zeros(shape)

    for instance in reversed(depth_sorted_instances):
        modal = instance.amodal_mask - accumulated
        modal_masks.append(modal)
        accumulated |= instance.amodal_mask

Check if point is occluded
--------------------------

::

    point = (x, y)

    # Check if point is in the visible region
    is_visible = visible_mask[point[1], point[0]]

Occlusion between two specific objects
--------------------------------------

::

    # Overlap between object A and B
    overlap = mask_a & mask_b

    # If A is in front of B, this much of B is hidden by A
    b_hidden_by_a = overlap.area()

    # Fraction of B occluded by A
    b_occ_by_a = b_hidden_by_a / mask_b.area() if mask_b.area() > 0 else 0
