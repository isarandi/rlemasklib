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

    iou = rlemasklib.iou([mask1_dict, mask2_dict])

Manual computation
------------------

If you need the intersection and union areas separately::

    intersection = mask1 & mask2
    union = mask1 | mask2

    iou = intersection.area() / union.area() if union.area() > 0 else 0.0

COCO's "crowd" IoU
------------------

``pycocotools`` computes a modified IoU against crowd regions (``iscrowd=1``),
dividing by the detection's area instead of the union. rlemasklib is a
general-purpose mask library and does not special-case this, but it is a
one-liner::

    # Fraction of the detection covered by the crowd region
    iou_crowd = (dt & gt).area() / dt.area() if dt.area() > 0 else 0.0
