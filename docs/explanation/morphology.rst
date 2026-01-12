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

The library uses this for large kernels.

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

The library provides:

- ``erode3x3``, ``dilate3x3``: square kernel
- ``erode_cross3x3``, ``dilate_cross3x3``: cross kernel

The cross kernel is faster (fewer shift operations). Use it when diagonal
connectivity doesn't matter.

Opening and closing
-------------------

Opening removes small foreground regions (noise)::

    open(mask) = dilate(erode(mask))

Closing fills small holes::

    close(mask) = erode(dilate(mask))

Both are O(runs) for fixed kernel sizes.

For removing regions smaller than a threshold, connected component analysis
with size filtering is often more appropriate than morphological opening.

Arbitrary structuring elements
------------------------------

For non-rectangular kernels, the library falls back to a general approach:

1. For each pixel in the structuring element, compute the shifted mask
2. Combine all shifted masks with OR (dilation) or AND (erosion)

Cost: O(kernel_nonzero_pixels × runs). Efficient for small sparse kernels,
but grows linearly with kernel complexity.

Very large or dense kernels might be faster with the dense approach.
The library doesn't automatically switch; choose based on your use case.

Run count growth
----------------

Dilation can increase run count. Each foreground region might sprout new
runs at its edges. In the worst case::

    output_runs ≈ input_runs × kernel_complexity

For most real masks, growth is modest. The library allocates conservatively
and shrinks after each operation.

Erosion typically decreases run count as small regions disappear entirely.
