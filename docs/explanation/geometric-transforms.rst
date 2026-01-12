Geometric Transforms
====================

Some operations on RLE are really simple, while others require some ingenuity. This page walks
through several geometric transforms, starting with the easy ones and building
up to the tricky transpose algorithm that makes many others others possible.

Complement
----------

With the complement operation, each pixel's value flips. Ones become zeros, zeros become ones.
On a dense array, we would visit every pixel and flip it. On RLE, we can do this much simpler.

Recall that RLE stores run lengths, alternating between background and foreground,
always starting with background. To complement the mask, we just need to swap
which runs are background and which are foreground. We can do this by prepending
or appending a zero-length run.

For example, if the original runs are::

    [3, 5, 2]

meaning 3 zeros, 5 ones, 2 zeros, then the complement is::

    [0, 3, 5, 2]

meaning 0 zeros (we start immediately with foreground), then 3 ones, 5 zeros,
2 ones. The runs themselves are unchanged. We just shifted which color each
run represents.

If the mask already starts with foreground (first run is zero), we remove
that leading zero instead of adding one. Either way, the operation is O(1)
or O(n) for a copy, but no arithmetic on the run values themselves.

To avoid the copy, we actually allocate the RLE data such that we have space for one extra run before the start of the array. Then complement is always O(1): just adjust the start pointer and length.

Rotation by 180 degrees
-----------------------

Rotating a mask by 180 degrees is equivalent to reversing the pixel order.
If we read the flattened column-major sequence backwards, we get the rotated
result.

On RLE, reversing the pixel order means reversing the run sequence. If the
original runs are::

    [3, 5, 2]

then the reversed runs are::

    [2, 5, 3]

There's only one minor catch. RLE always starts with a background run, so if the original
mask ended with foreground (even number of runs), then reversing would make it
start with background. We fix this the same way as complement: prepend a
zero-length run in such cases.

So rotation by 180 degrees is just reversing the run array, with some care
around the leading zero. Still O(n) where n is the number of runs, and we
never touch individual pixels.

A challenge: rotation by 90 degrees
-----------------------------------

Now try to think of how we would rotate by 90 degrees, say, clockwise.

It is not obvious. The column-major flattening interleaves the data in a way
that does not simply reverse or shift. A pixel at position (row, col) moves
to (col, height - 1 - row). The new columns were old rows, and vice versa.
The run structure gets scrambled in a complicated way.

We will return to this. But first, let us look at some other operations that
turn out to be more manageable.

Binary operations
-----------------

Binary operations combine two masks, and correspond to set operations such as union, intersection, difference, XOR.
On dense arrays, we visit every pixel and apply the boolean function.
On RLE, we can do better.

The key observation is that both masks change value only at run boundaries.
We don't have to visit every pixel; we only need to visit the boundaries (of either mask).
We can do this in one fast pass.

Each time we jump ahead to the nearest run boundary in either mask. When we hit a boundary,
we update the current value of that mask. We then compute the output value of the binary operation
on the two current values. If the output value changes, we record a new run boundary.

The result has at most as many runs as the two inputs combined. For masks
with few runs, this is much faster than touching every pixel.

The difference between union, intersection, and XOR is just the boolean function
we apply at each step, but the overall algorithm is the same.

Shifting
--------

Shifting a mask moves it by some offset, so pixels that were at (row, col) move to
(row, col + offset) for a horizontal shift, or (row + offset, col) for a vertical shift.

Horizontal shifts (along rows) are fairly simple. All we need to do is adjust the beginning
and end of the run counts sequence. If the height is H, and we want to shift by one column to the right,
we need to add H background pixels at the start, and remove H pixels at the end. This doesn't change
the bulk of the run lengths; we just need to adjust the edges (though copies can be necessary, but copying is much faster than detailed processing).

For a vertical shift (along columns), things are trickier because columns
are contiguous in our column-major layout. We can have runs that start at the bottom of one column
and continue at the top of the next. To shift vertically, we need to break these runs at column boundaries.
This may require splitting runs that cross column boundaries. This is somewhat more work, but still manageable.

So, shifting is O(n) in the number of runs, not the number of pixels, and in practice often even better,
for example if we have a large background and a small foreground object, shifting just involves adjusting a the first and last run.


Vertical flip
-------------

Flipping a mask vertically means reversing each column. In column-major order,
each column is a contiguous segment of the flattened array. To flip vertically,
we reverse each segment independently.

On RLE, this means we partition the runs by column, reverse each partition,
and concatenate. The partitioning is not free, because as we saw with vertical shifts,
runs can cross column boundaries. But in addition to managing these splits, we also need to
keep track of the runs we saw in the current column so we can reverse them.

The total work is proportional to the number of runs plus the width (since each column
boundary might cause a split, though in natural masks where the foreground
does not reach the image boundary, the processing is proportional to the
number of runs).

This is more work than complement or rot180, but still avoids touching
every pixel.

Note that now horizontal flipping became easy: rotate 180, then vertical flip.

Transpose: the key operation
----------------------------

Transpose swaps rows and columns. A pixel at (row, col) moves to (col, row).
That may not seem consequential, but if we can implement transpose, we get
the other hard operations for free.

We already have vertical flip and 180 degree rotation. If we get transpose,
we will have all eight operations in the dihedral group D4, the symmetry
group of a square:

- Rotate 90 clockwise = transpose, then flip horizontally
- Rotate 90 counter-clockwise = transpose, then flip vertically
- Flip horizontally = transpose, then flip vertically, then transpose

Mathematically, {transpose, vflip} is a generating set for D4. But we keep
rot180 as a primitive because building it from transpose and vflip would be
far more expensive than just reversing the run array.

So if we can transpose efficiently, we get the remaining operations too.

Naive approaches
~~~~~~~~~~~~~~~~

The simplest approach: decode to a dense array, transpose the array, encode
back to RLE. This works, but it is O(width * height), touching every pixel.
We can do better.

A smarter approach: go column by column in the output (which is row by row
in the input). For each output column, we need to know where the runs are.
We could track, for each row in the input, whether we are currently in
foreground or background, and how long the current run has been going.

As we step through the input columns, we update the state for each row.
When a run boundary crosses a row, we record it. After processing all input
columns, we concatenate the per-row run lists into the final output.

But this is still O(width * height) because we must examine every input
column for every row. We cannot skip ahead to the next run boundary; we
must check each position to update the per-row state.

The XOR trick
~~~~~~~~~~~~~

We can do better. The key insight is that the transposed run boundaries are
exactly the positions where a pixel differs from its left neighbor (in the
original image). If we can find these positions fast, we can build the
transposed RLE directly.

And we can find them fast: we shift the mask right by one column and XOR with
the original. Each position in the shifted mask holds what was in the left
neighbor's position. XOR marks where they differ.

For example, consider a small mask (shown row by row for clarity)::

    0 0 1 1
    0 1 1 1
    0 0 1 0

Shift right by one column::

    ? 0 0 1
    ? 0 1 1
    ? 0 0 1

But what about the first column (marked with ?)? In the flattened column-major
array, runs are continuous: a run can start at the bottom of one column and
continue at the top of the next. So the "left neighbor" of a pixel at the start of a row is actually the last pixel in the row above.

This means the shift is circular: the last column wraps around to become the
first column, but shifted down by one row (with the top left corner filled with 0)::

    0 0 0 1
    1 0 1 1
    1 0 0 1


XOR with original::

    0 0 1 0
    1 1 0 0
    1 0 1 1

The 1s mark where the value in the original mask differs from its left neighbor: exactly the
vertical boundaries that become horizontal boundaries after transpose.

Both the shift and XOR operations work on RLE without decoding. The shift
is somewhat intricate to implement because of the wrap-around and vertical
offset, but it remains O(n) in the number of runs. The result is a sparse
RLE with 1s only at boundary positions.

From boundaries to runs
~~~~~~~~~~~~~~~~~~~~~~~

After the XOR, we have a sparse mask marking boundary crossings. To build
the transposed RLE, we need to collect these boundaries column by column
in the transposed orientation, which means row by row in the original.
But this is not the order they appear in the XOR result. We need to reorder them.
A naive idea wold be to do this with a sort like quicksort, but that is O(n log n).

Instead, we can do this with a bucket sort. Each boundary position has a (row, col)
coordinate in the original mask. In the transposed mask, it will be at
(col, row). We bucket the boundaries by their new column (which is the
original row), then within each bucket sort by the new row (original column).

Bucket sort runs in O(n + height) time, where n is the number of boundaries.
After sorting, we walk through the boundaries in transposed order. Between
consecutive boundaries, the mask value is constant. We compute the run lengths
as the differences between consecutive boundary positions.

Why this works well
~~~~~~~~~~~~~~~~~~~

The XOR and bucket sort approach has a key advantage. The number of boundaries
is proportional to the number of runs, not the number of pixels. For a smooth
mask, this is much smaller.

Consider a solid rectangle. The dense representation has width * height pixels.
But the RLE has only a handful of runs (one per column that intersects the
rectangle). The transpose operation touches only these few boundaries, not
every pixel.

This is the sqrt benefit again. Storage and processing scale with perimeter,
not area. The transpose algorithm preserves this property.

Putting it together
-------------------

With transpose in hand, we can now implement the 90 degree rotations:

- **Rotate 90 CW**: transpose, horizontal flip
- **Rotate 90 CCW**: transpose, vertical flip

Each building block is O(n) in the number of runs. The compositions are still
efficient, and we never decode to a dense array.

The complement was trivial. Rot180 was almost trivial. Binary ops and shifting
were manageable. Vertical flip required splitting runs at column boundaries.
Transpose required some tricks with the shift and XOR  and bucket sort. But
together, they give us all the geometric transforms we need, all without
touching individual pixels.

Crop and pad
------------

Cropping extracts a rectangular region from the mask and padding adds (background)
pixels around the edges. These operations are conceptually simple but quite tedious
to implement correctly on RLE.

Cropping columns only
~~~~~~~~~~~~~~~~~~~~~

The easiest case is when we crop columns but keep all rows. In column-major
order, each column is a contiguous segment of the flattened array. Cropping
to columns c1 through c2 means extracting a contiguous slice of pixels.

We walk through the runs until we find the one containing the first pixel of
the box. We note how much of that run falls inside the box. Then we walk
backwards from the end to find the run containing the last pixel of the box.
The runs in between can be copied unchanged. We just need to trim the first
and last runs to the box boundaries.

There is a special case: if the box falls entirely within a single run, the
output is just one run (or two if we need a leading zero to start with
background). And if the first run inside the box is foreground, we prepend
a zero-length background run.

Cropping rows too
~~~~~~~~~~~~~~~~~

When we also crop rows, things get messier. A single run in the input can
span multiple columns, and within each column it may partially overlap the
row range we want to keep.

For each run, we compute how much of it intersects the crop box. The run
starts at some position in the flattened array. We compute where that maps
to in the cropped coordinate system: which column, which row within that
column. We do the same for the end of the run. The contribution of this run
to the output depends on how many complete columns it spans inside the box,
plus the partial contributions at the start and end columns.

The formula involves integer division and modulo by the column height, with
careful clamping to the box boundaries. Runs entirely outside the box become
zero-length and are later eliminated.

Zero padding
~~~~~~~~~~~~

Padding adds background pixels around the edges. The horizontal case (adding
columns on the left or right) is easy: we just add to the first run (for left
padding) or the last run (for right padding). If the original mask ended with
foreground, we may need to append a new background run.

Vertical padding (adding rows at the top or bottom) is harder. We need to
insert background pixels at the start and end of each column. But runs can
span column boundaries.

The key insight: runs of background (zeros) just get longer. If a background
run spans k column boundaries, it grows by k times the vertical padding. But
runs of foreground (ones) that span column boundaries must be split. A
foreground run that starts in one column and continues into the next becomes
multiple runs: foreground in the first column, then background (the padding),
then foreground in the next column.

The implementation does two passes. First, it counts how many runs the output
will have, since foreground runs may split. Then it walks through the input
runs, expanding background runs and splitting foreground runs at column
boundaries, inserting the padding between the pieces.

There is also a "carry over" flag for when a foreground run ends exactly at
a column boundary. The next background run needs extra padding because a new
column started.

Replicate padding
~~~~~~~~~~~~~~~~~

Zero padding is not the only option. Replicate padding extends the edge pixels
outward: the left column is repeated for the left padding, and so on. This is
useful for convolutions where we want to avoid edge artifacts.

This is where the implementation gets truly tedious. We need to identify the
first and last columns of the original mask, determine which runs touch the
top and bottom of those columns, and replicate their structure. If the left
column has multiple runs (say, background at top, foreground in middle,
background at bottom), each replicated column must have the same structure,
with the top and bottom portions extended by the padding amount.

The code tracks j_toplef, j_botlef (runs containing top-left and bottom-left
pixels), j_toprig, j_botrig (same for right), and whether each corner is
foreground or background. It handles special cases like when an entire edge
column is uniform, when the mask is only one column wide, and so on.

There is no clever trick here, unfortunately. It is careful bookkeeping through many cases.

Complexity
~~~~~~~~~~

Crop and pad are O(n) in the number of runs. For most masks, this is much
better than O(width * height). But unlike the elegant transpose algorithm,
the code is not pretty. It is the kind of code where you write it, test it,
find an off-by-one error, fix it, find another, and repeat until the tests
pass.

Two-pass warps
--------------

Now for something surprising: we can do affine and perspective warps directly
on RLE without decoding.

The key insight is that these transforms can be decomposed into two 1D
operations. An affine transform is a combination of scaling, rotation, and
shear. But it can also be viewed as two successive shears along perpendicular
axes. Each shear can be applied column by column, transforming run boundaries
without touching individual pixels.

The algorithm
~~~~~~~~~~~~~

The warp proceeds in five steps:

1. **Pre-rotate**: Compute the dominant direction of the transform (how a
   horizontal unit vector maps). Rotate the coordinate system by the nearest
   multiple of 90 degrees to make the transform roughly axis-aligned. This
   minimizes the shear in each pass.

2. **Pre-flip**: If the transform would flip the Y axis, apply a flip to the
   matrix. This ensures runs always move "forward" and we never need to go
   backwards in the output.

3. **Pass 1**: For each foreground run, transform its Y coordinates. The run
   at (x, y_start) to (x, y_end) becomes a run at (x, y'_start) to (x, y'_end)
   in the intermediate image. We emit the appropriate number of zeros (gap
   from the previous run) and ones (run length).

4. **Transpose**: Swap rows and columns in the intermediate result.

5. **Pass 2**: Same as Pass 1, but now we are transforming what was the X
   coordinate (which is now Y after the transpose). The transformation formula
   is derived from the original matrix.

6. **Undo rotation and flip**: Apply the inverse of the pre-rotation and
   pre-flip to get the final result in the correct orientation.

Why decomposition works
~~~~~~~~~~~~~~~~~~~~~~~

Consider an affine transform::

    x' = a*x + b*y + c
    y' = d*x + e*y + f

Pass 1 transforms Y within each column. For a fixed x, the transformation
is just::

    y' = e*y + (d*x + f)

This is a linear function of y. The run boundaries (y_start, y_end) map to
(y'_start, y'_end) via this linear function. We can compute the output run
boundaries directly without iterating over pixels.

After transposing, Pass 2 does the same for X. The formula for x' given the
intermediate (transposed) coordinates requires a bit of algebra, but it is
still a linear function within each column.

For perspective transforms, the formulas involve division (homogeneous
coordinates), but the same decomposition applies. Each pass computes a
rational function of the run boundaries.

Handling backwards motion
~~~~~~~~~~~~~~~~~~~~~~~~~

There is a subtlety: the shear might cause a run to start before the previous
run ended. This would require "going backwards" in the output, which the
streaming algorithm cannot do.

The code handles this gracefully:

- If the new run would start before the previous one ended, it reduces the
  run length so it at least ends in the right place.
- If even the end would go backwards, it skips the run entirely.

This produces an approximate result, but maintains a valid RLE. For most
practical transforms (small shears, no extreme distortion), this does not
happen.

Complexity
~~~~~~~~~~

Each pass processes each run once, so the algorithm is O(n) in the number of
runs. The pre-rotation and transpose are also O(n). For smooth masks with
few runs, this is much faster than decoding to a dense array, warping pixel
by pixel, and re-encoding.

The two-pass warp is perhaps the most interesting algorithm in the library.
Affine and perspective transforms seem inherently 2D, yet we can decompose
them into 1D operations and never touch individual pixels.
