Combine Multiple Masks
======================

Common operations for merging masks from multiple sources.

Union (logical OR)
------------------

Combine regions from any of the masks::

    combined = mask1 | mask2

    # Or for many masks:
    combined = RLEMask.union([mask1, mask2, mask3, mask4])

Intersection (logical AND)
--------------------------

Keep only regions present in all masks::

    overlap = mask1 & mask2

    # Or for many masks:
    overlap = RLEMask.intersection([mask1, mask2, mask3])

Difference (subtract)
---------------------

Remove one mask's region from another::

    # Pixels in mask1 but not in mask2
    difference = mask1 - mask2

XOR (exclusive or)
------------------

Pixels in exactly one mask, not both::

    xor = mask1 ^ mask2

Layer compositing
-----------------

When stacking objects front-to-back, compute visible portions::

    # Objects sorted back-to-front (furthest first)
    occluder = RLEMask.zeros(shape)

    for obj in reversed(objects):
        # What's visible = instance minus everything in front
        obj.visible_mask = obj.instance_mask - occluder
        occluder = occluder | obj.instance_mask
