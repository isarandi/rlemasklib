Find the Largest Rectangle Inside a Mask
=========================================

Find the largest axis-aligned rectangle that fits entirely within
the foreground region. Useful for cropping to usable content.

Basic usage
-----------

::

    bbox = mask.largest_interior_rectangle()
    x, y, w, h = bbox

Returns ``[x, y, width, height]`` of the largest rectangle.

With aspect ratio constraint
----------------------------

Find the largest rectangle with a specific aspect ratio::

    # 16:9 aspect ratio
    bbox = mask.largest_interior_rectangle(aspect_ratio=16/9)

The returned rectangle will have exactly that width/height ratio.

Centered on a point
-------------------

Find the largest rectangle that must contain a specific center point::

    center = (320, 240)  # x, y
    bbox = mask.largest_interior_rectangle_around(center)

With both constraints
---------------------

::

    bbox = mask.largest_interior_rectangle_around(
        center=(320, 240),
        aspect_ratio=4/3
    )

Use case: optimal crop after warping
------------------------------------

After warping an image, the valid region may be non-rectangular.
Find the best crop that avoids black borders::

    # valid_mask marks pixels that have real data (not border fill)
    crop_box = valid_mask.largest_interior_rectangle(aspect_ratio=16/9)

    x, y, w, h = crop_box
    cropped_image = image[y:y+h, x:x+w]

Use case: undistortion
----------------------

When removing lens distortion, find the largest usable rectangle::

    # valid_region is where the undistortion mapping is defined
    inner_crop = valid_region.largest_interior_rectangle()

    # Or keep principal point centered
    principal_point = (intrinsic_matrix[0, 2], intrinsic_matrix[1, 2])
    centered_crop = valid_region.largest_interior_rectangle_around(principal_point)
