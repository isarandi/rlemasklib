Largest Interior Rectangle
==========================

Given a binary mask, we want to find the largest axis-aligned rectangle that fits
entirely inside the foreground. This comes up when cropping images to
the visible region after warping, or when placing text or overlays
inside a segmented object.

The dense approach computes, for each pixel, how far up the foreground
extends. This gives a histogram per row. Then we solve "largest rectangle
in histogram" for each row. Total cost: O(h × w).

With RLE, we want to do better when the mask has few runs.

The histogram approach
----------------------

The classic "largest rectangle in histogram" problem has an O(n) solution
using a stack. For a histogram with n bars::

    heights = [2, 1, 5, 6, 2, 3]

We scan left to right. The stack holds indices of bars in increasing height
order. When we see a bar shorter than the stack top, we pop and compute the
rectangle that had the popped bar as its shortest bar. The width extends
from the new stack top to the current position.

For 2D masks, we compute a histogram for each row: for column x, the
histogram value is how many consecutive foreground pixels are above
(including the current row). Then we run the 1D algorithm on each row's
histogram. This gives O(h × w) total.

Adapting to RLE
---------------

The histogram approach touches every pixel. With RLE, we want to skip
uniform regions.

The key observation: a candidate rectangle's right edge must align with
where a horizontal run ends. If a rectangle doesn't extend to a run
boundary, we could extend it rightward and get a larger rectangle.

Similarly for the top and bottom edges—they must align with where
vertical extent changes. But vertical extent changes exactly where
horizontal runs start or end for some row.

So we only need to check rectangles at "events" where the mask structure
changes. Between events, nothing interesting happens.

Left and right contours
-----------------------

Define the left contour as foreground pixels that have background to their
left (or are at the image boundary). These are pixels where a horizontal
run starts. The right contour is foreground with background to the right—where
runs end.

We compute these using RLE boolean operations::

    left_contour = mask - shift_right(mask)
    right_contour = mask - shift_left(mask)

Shifting an RLE by one column is cheap: adjust the first run's length by h.
The subtraction is a merge operation, O(runs).

The column sweep
----------------

We process columns left to right, maintaining state::

    start_x[y] = the x coordinate where row y's current horizontal run started
                 (UINT_MAX if row y is currently in background)

At each column:

1. For pixels in the left contour: set ``start_x[y] = x``
2. For pixels in the right contour: check candidate rectangles
3. For pixels in neither: do nothing

Step 3 is the payoff. Columns with no contour pixels cost nothing. We
skip over uniform regions without decoding them.

The stack trick
---------------

When we hit a right contour pixel, we need to find all maximal rectangles
ending at this column. The candidates differ in their top edge position.

We scan down the column, tracking where the width changes. When ``start_x``
decreases (run started earlier, so width increases), we push the row index
onto a stack. When ``start_x`` increases (run started later, width decreases),
we pop and check the rectangle.

The stack holds row indices in order of increasing ``start_x``. Each pop
gives us a candidate rectangle::

    right edge: current x
    left edge: start_x[popped_row]
    top edge: row below the new stack top (or 0)
    bottom edge: current row

We check if this rectangle is the best so far. The algorithm is analogous
to the 1D histogram solution, but we run it per-column only when there
are contour pixels.

Aspect ratio constraint
-----------------------

Sometimes we want the largest rectangle with a fixed aspect ratio, say 16:9.
The algorithm is the same, but when we evaluate a candidate, we constrain it.

Given a bounding rectangle of width w and height h with aspect ratio r = w/h,
and target aspect ratio t:

- If r > t (too wide): shrink width to h × t, center horizontally
- If r < t (too tall): shrink height to w / t, center vertically

The constrained rectangle is the largest with aspect ratio t that fits
inside the bounding rectangle. We track the best constrained area instead
of the best raw area.

Center constraint
-----------------

A different problem: find the largest rectangle containing a specific
center point (cx, cy), for example the principal point of an image with a calibrated camera. This requires a different algorithm.

The rectangle must be symmetric around the center, i.e., if it extends d pixels
left of center, it must extend d pixels right. Same for up and down.

We build a 1D histogram indexed by x-distance from center::

    hist[d] = minimum y-extent at x-distance d

For each d from 0 to the maximum possible, hist[d] tells us how far
we can go up and down from cy while staying in foreground, across all
columns at distance d from cx.

Scanning the RLE
~~~~~~~~~~~~~~~~

We scan runs, looking for those that intersect the center row (y = cy).
For each such run:

- Compute which columns it covers at row cy
- For each column, compute the y-extent (distance from cy to run boundary)
- Update hist[d] with the minimum

Runs of zeros set hist[d] = 0 for the affected distances. We track
``xdist_earliest_zero``—beyond this distance, no valid rectangle exists.

Finding the best rectangle
~~~~~~~~~~~~~~~~~~~~~~~~~~

The histogram is symmetric by construction. A rectangle centered at
(cx, cy) with half-width d has::

    width = 2d + 1
    height = 2 × min(hist[0], hist[1], ..., hist[d]) + 1

We scan d from 0 to ``xdist_earliest_zero``, tracking the running minimum
of hist values, and compute the area for each candidate. The best wins.

Combining center and aspect ratio
---------------------------------

When both constraints apply, we use the center algorithm to generate
candidates, then apply aspect ratio constraints when evaluating.

For a candidate with half-width d and half-height min_hist::

    width = 2d + 1
    height = 2 × min_hist + 1
    ratio = width / height

If ratio > target: constrain width, compute area as height × (height × target).
If ratio < target: constrain height, compute area as width × (width / target).

The position adjusts to keep the rectangle centered at (cx, cy) after
constraining.

Cost analysis
-------------

The unconstrained algorithm costs O(h + runs) per column with contour pixels,
and O(1) per column without. Total: O(w × h + runs × h) in the worst case,
but often much better when contours are sparse.

The center-constrained algorithm is O(runs) to scan and O(bbox_width) to
evaluate the histogram. For masks with few runs relative to their bounding
box, this is faster than the general algorithm.
