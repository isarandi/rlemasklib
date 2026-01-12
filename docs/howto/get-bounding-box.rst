Get the Bounding Box of a Mask
==============================

The bounding box is the smallest axis-aligned rectangle containing all
foreground pixels.

Basic usage
-----------

::

    bbox = mask.bbox()  # Returns [x, y, width, height]

The format matches COCO's bounding box convention.

Components
----------

::

    x, y, w, h = mask.bbox()

    # Top-left corner
    top_left = (x, y)

    # Bottom-right corner (exclusive)
    bottom_right = (x + w, y + h)

Empty masks
-----------

If the mask has no foreground pixels, ``bbox()`` returns ``[0, 0, 0, 0]``.

Check first if needed::

    if mask.area() > 0:
        bbox = mask.bbox()
    else:
        # Handle empty mask

Crop to bounding box
--------------------

To extract just the region containing the mask and get both
the cropped mask and the box coordinates::

    cropped, bbox = mask.tight_crop()

Functional API
--------------

With COCO-format dicts::

    import rlemasklib

    bbox = rlemasklib.toBbox(mask_dict)  # Single mask
    bboxes = rlemasklib.toBbox([mask1, mask2, mask3])  # Multiple masks
