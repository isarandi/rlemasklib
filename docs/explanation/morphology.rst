Morphological Operations on RLE
===============================

Morphological operations—erosion and dilation—are fundamental in image
processing. Erosion shrinks foreground regions; dilation grows them.
Opening (erosion then dilation) removes small protrusions. Closing
(dilation then erosion) fills small holes.

The dense approach applies a structuring element at every pixel: O(pixels × kernel_size).
With RLE, we want O(runs).

Vertical operations are easy
----------------------------

Consider dilating vertically by 1 pixel: each foreground pixel spreads up
and down by one row. In RLE terms, each run of 1s extends by 1 at both ends.

Simple implementation::

    for each run i:
        if i is odd (foreground):
            cnts[i-1] -= 1  # shrink preceding zeros
            cnts[i] += 2    # extend foreground
            cnts[i+1] -= 1  # shrink following zeros

Handle boundary conditions (can't shrink below 0, runs at image edges).
Cost: O(runs).

Erosion is the dual: shrink foreground runs, grow background runs.

Horizontal operations are harder
--------------------------------

Horizontal dilation spreads left and right. In column-major RLE, that means
pixels in column x spread to columns x-1 and x+1. But columns are encoded
separately—we can't just adjust run lengths.

The naive approach: decode one column, dilate, re-encode. Repeat for all
columns. That's O(h × w).

The better approach: use boolean operations. Dilating right by 1 pixel is::

    dilate_right(mask) = mask | shift_left(mask)

Shifting an RLE left by one column is O(runs): subtract h from the first run
(or remove it if it becomes empty), add h to the last run. The OR is O(runs).

So horizontal dilation by 1 is O(runs). Dilating by k pixels chains k shifts
and ORs: O(k × runs).

Combining directions
--------------------

A 3×3 square dilation combines both:

1. Dilate vertically by 1: O(runs)
2. Dilate horizontally by 1 (both directions): O(runs)

Total: O(runs), not O(pixels).

For larger kernels, the cost grows with kernel size. A 5×5 kernel needs
dilation by 2 in each direction. An arbitrary kernel might need O(kernel_area)
shift-and-OR operations.

Optimizing repeated operations
------------------------------

Dilating by k pixels naively does k shifts and ORs. But we can do better
using doubling::

    dilate_by_1 = mask | shift(mask, 1)
    dilate_by_2 = dilate_by_1 | shift(dilate_by_1, 2)
    dilate_by_4 = dilate_by_2 | shift(dilate_by_2, 4)
    ...

This gives dilation by k in O(log k) operations instead of O(k).

For the separable square and cross kernels there is an even better option:
the O(runs) vertical-dilation primitive, applied to the mask and its
transpose, has cost independent of the kernel size altogether. The library
uses this separable approach for larger square and cross kernels.

Erosion via complement
----------------------

Erosion and dilation are duals::

    erode(mask) = complement(dilate(complement(mask)))

Eroding foreground is the same as dilating background, then swapping.
The library implements erosion this way rather than duplicating the
dilation logic.

The contour operation
---------------------

A useful derived operation: find the contour (outline) of a mask.

The inner contour is foreground pixels adjacent to background::

    inner_contour = mask - erode(mask)

The outer contour is background pixels adjacent to foreground::

    outer_contour = dilate(mask) - mask

Both are O(runs) using the RLE operations.

Separable kernels
-----------------

Many useful kernels are separable: a 3×3 cross (no corners) is just
horizontal and vertical lines. A 3×3 square is the dilation of a cross.

The library provides ``erode3x3`` and ``dilate3x3``, whose kernel is selected by the
``connectivity`` argument:

- ``connectivity=4`` (the default): cross kernel (no corners)
- ``connectivity=8``: square kernel

The cross kernel is faster (fewer shift operations). Use it when diagonal
connectivity doesn't matter. For larger kernels, ``erode``/``dilate`` take a
``kernel_shape`` (``'circle'``, ``'square'``, ``'diamond'``, ``'cross'``) and ``kernel_size``.

Opening and closing
-------------------

Opening removes small foreground regions (noise)::

    open(mask) = dilate(erode(mask))

Closing fills small holes::

    close(mask) = erode(dilate(mask))

Both are O(runs) for fixed kernel sizes.

For removing regions smaller than a threshold, connected component analysis
with size filtering is often more appropriate than morphological opening.

Non-separable kernel shapes
---------------------------

The circle and diamond kernels are not separable, so ``erode``/``dilate``
decompose them by columns: each kernel column is a vertical segment, so the
mask is vertically dilated per distinct column height, shifted horizontally
into place, and the shifted copies are OR-ed together.

Cost: O(kernel_width × runs). Efficient for the moderate kernel sizes typical
in mask processing.

Arbitrary structuring elements (any 0/1 array) are not part of the public
morphology API — only the named shapes ``'circle'``, ``'square'``,
``'diamond'`` and ``'cross'`` are. General weighted kernels are available
through thresholded convolution (``conv2d_valid``), and for very large dense
kernels the dense approach (decode, filter with OpenCV, re-encode) may be
faster.

OpenCV parity
-------------

The ``'square'`` kernel matches OpenCV's ``cv2.MORPH_RECT`` exactly:
eroding or dilating with ``kernel_shape='square'`` gives pixel-identical
results to ``cv2.erode``/``cv2.dilate`` with a rectangular structuring
element of the same size. The ``'circle'`` kernel contains the pixels within
Euclidean distance ``kernel_size / 2`` of the center, which differs slightly
from ``cv2.MORPH_ELLIPSE``'s rasterization, so results can differ in a thin
ring of boundary pixels.

Run count growth
----------------

Dilation can increase run count. Each foreground region might sprout new
runs at its edges. In the worst case::

    output_runs ≈ input_runs × kernel_complexity

For most real masks, growth is modest. The library allocates conservatively
and shrinks after each operation.

Erosion typically decreases run count as small regions disappear entirely.
