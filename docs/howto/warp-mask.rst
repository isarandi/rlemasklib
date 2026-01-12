Warp a Mask with a Transformation
==================================

Apply geometric transforms (affine, perspective) to masks.

Affine transform
----------------

For rotation, scaling, shearing, translation::

    import numpy as np

    # 2x3 affine matrix (same as OpenCV)
    M = np.array([
        [0.9, -0.1, 10],
        [0.1,  0.9, 20]
    ], dtype=np.float32)

    warped = mask.warp_affine(M, dst_shape=(480, 640))

Perspective transform
---------------------

For 3x3 homography matrices::

    # 3x3 perspective matrix
    H = np.array([
        [1.1, 0.1, 5],
        [0.05, 1.2, 10],
        [0.0001, 0.0002, 1]
    ], dtype=np.float32)

    warped = mask.warp_perspective(H, dst_shape=(480, 640))

From OpenCV
-----------

If you have a transform matrix from OpenCV::

    import cv2

    # Get affine transform from point correspondences
    src_pts = np.array([[0, 0], [100, 0], [0, 100]], dtype=np.float32)
    dst_pts = np.array([[10, 10], [110, 20], [5, 115]], dtype=np.float32)
    M = cv2.getAffineTransform(src_pts, dst_pts)

    warped = mask.warp_affine(M, dst_shape)

    # Or perspective from 4 points
    H = cv2.getPerspectiveTransform(src_4pts, dst_4pts)
    warped = mask.warp_perspective(H, dst_shape)

Resize
------

Simple scaling is a special case::

    # Resize to specific dimensions
    resized = mask.resize((new_height, new_width))

    # Or use warp_affine with a scale matrix
    scale = 0.5
    M = np.array([[scale, 0, 0], [0, scale, 0]], dtype=np.float32)
    resized = mask.warp_affine(M, dst_shape)

Decode-warp-encode fallback
---------------------------

For complex warps not directly supported, decode to array first::

    import cv2

    # Decode
    arr = mask.to_array()

    # Warp with OpenCV (use INTER_NEAREST for binary masks)
    warped_arr = cv2.warpPerspective(
        arr, H, (width, height),
        flags=cv2.INTER_NEAREST
    )

    # Re-encode
    warped_mask = RLEMask.from_array(warped_arr)

Performance note
----------------

Direct RLE warping (``warp_affine``, ``warp_perspective``) avoids
decoding to a dense array. For sparse masks this can be much faster
than the decode-warp-encode approach.
