Compute IoU Between Masks
=========================

Intersection over Union (IoU) measures how much two masks overlap.
It's the standard metric for comparing segmentations.

Two masks
---------

::

    from rlemasklib import RLEMask

    iou = RLEMask.iou(mask1, mask2)

This returns a float between 0 (no overlap) and 1 (identical).

Batch computation
-----------------

For comparing many masks against many others::

    # Returns a matrix of shape (len(masks_a), len(masks_b))
    iou_matrix = RLEMask.iou_matrix(masks_a, masks_b)

Using the functional API
------------------------

With COCO-format dicts::

    import rlemasklib

    iou_matrix = rlemasklib.iou([mask1_dict, mask2_dict], [mask3_dict, mask4_dict])

Manual computation
------------------

If you need the intersection and union areas separately::

    intersection = mask1 & mask2
    union = mask1 | mask2

    iou = intersection.area() / union.area() if union.area() > 0 else 0.0
