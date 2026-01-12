Connected Components in RLE
===========================

Given a binary mask, we want to find connected components: maximal regions
of foreground pixels that are all reachable from each other. The standard
approach for dense arrays is a two-pass algorithm or flood fill, both O(pixels).

But we have RLE-encoded masks. The whole point of RLE is that operations
should be O(runs), which grows with the perimeter, not the area. Can we
find connected components without decoding?

Union-find on runs
------------------

The idea is to treat each run of 1s as a node in a graph. Two runs are
connected if they touch. Once we know the graph, we use union-find to
group runs into components.

The important question is: which runs in *adjacent columns* touch?

The two-pointer scan
--------------------

Runs are stored column by column, top to bottom. Given a run at position
r (counting from the start of the flattened array), it lives in column
r // h and spans rows r % h through (r + length - 1) % h.

To find adjacencies between neighboring columns, we use two pointers into
the run array. Pointer i1 walks through runs in column c, pointer i2 walks
through runs in column c+1. We track r1 and r2, the starting positions of
each run.

Column c+1 starts h pixels after column c in the flattened array.
So when r2 catches up to r1 + h, we're comparing runs in adjacent columns.
We check if they overlap vertically::

    overlap_start = max(r1 + h, r2)
    overlap_end = min(r1 + cnts[i1] + h, r2 + cnts[i2])
    if overlap_start < overlap_end:
        union(i1, i2)

We advance whichever pointer's run ends first (accounting for the h offset),
and repeat until we've processed all runs.

4-connectivity vs 8-connectivity
--------------------------------

With 4-connectivity, two pixels are neighbors if they share an edge.
Runs must actually overlap, share at least one row, to be connected.

With 8-connectivity, diagonal neighbors count. Two runs that merely touch
at a corner are connected. This means we also union when::

    overlap_start == overlap_end and overlap_start % h != 0

The second condition excludes the case where both runs end exactly at a
column boundary, which would be a corner touch at the image edge—not a
real diagonal adjacency.

The splitting problem
---------------------

Consider a run of 1s that straddles a column boundary:

This single run covers pixels in two columns. If the bottom part connects
to something in column 0 but the top part connects to something different
in column 1, the run should be split between two components.

When can this happen? Only if the run is short enough that it doesn't
connect to itself across the column boundary. If a run spans the full
column height h, its top and bottom are in adjacent columns and touch
(8-connectivity) or almost touch (4-connectivity), so it stays together.

Before running union-find, we scan for runs that:

- Are runs of 1s (odd index)
- Span a column boundary: (r + length - 1) / h > r / h
- Are short enough to possibly split: length ≤ h (or h+1 for 8-connectivity)

Such runs get split into two runs with a zero-length gap between them.
After union-find, if both parts ended up in the same component, we merge
them back.

Tracking component sizes
------------------------

Union-find nodes carry a size field, initialized to the run's length.
When two nodes are unioned, the new root's size becomes the sum::

    void union(x, y):
        x = find(x)
        y = find(y)
        if x == y: return
        if x.size < y.size:
            x.parent = y
            y.size += x.size
        else:
            y.parent = x
            x.size += y.size

After the scan, each root node knows the total pixel count of its component.
This costs nothing extra beyond what's needed for union-find and will come in
handy for filtering.

Filtering small components
--------------------------

A common operation is to remove small components (noise). We could extract
all components, measure each, and discard the small ones. But that builds
RLEs we'll throw away.

Instead, filter during output. After union-find completes, scan the runs
again. For each run, find its root and check the size::

    for each run i:
        root = find(i)
        if root.size >= min_size:
            output run to its component
        else:
            extend previous zero-run

Same cost as full extraction, just with a size check. Runs from small
components become background.

Largest component only
----------------------

Similar idea. After union-find, scan the root nodes to find the one with
maximum size::

    max_root = None
    for each root node n:
        if n.size > max_size:
            max_size = n.size
            max_root = n

Then in the output phase, only keep runs whose root is max_root. Everything
else becomes background. One union-find pass, one output pass.

Reassembling output RLEs
------------------------

After union-find, we know which component each run belongs to. Now we need
to build an RLE for each component.

The naive approach allocates one output RLE per component, then scans the
input runs. For each run, append it to the correct component's RLE. Runs
not in that component extend the previous zero-run.

One complication: runs we split earlier might end up in the same component.
When appending a run, check if the previous run of zeros in that component
has length zero. If so, merge the new run of 1s with the previous one
instead of adding a new run pair.

The output phase is O(runs × components). For a single component (largest
or filtered), it's O(runs). For full extraction with k components, it's
O(k × runs). If k is large, this can dominate. But in practice, masks from
segmentation models have few components, and the union-find phase is the
main cost.
