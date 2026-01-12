Boolean Operations on RLE
=========================

Given two RLE-encoded masks, we can compute their union, intersection, difference,
or any other boolean combination efficiently in RLE. The naive approach where we decode both masks to
dense arrays has linear cost in the number of pixels of the image.

With RLE, we can achieve O(runs).

The merge algorithm
-------------------

RLE stores alternating run lengths: background, foreground, background, ...
At any position in the flattened array, we know which mask is 0 and which is 1
by tracking the parity of the run's index.

We walk through both masks simultaneously, processing one "switch event" at
a time, i.e. where either mask's run ends and a new run with the opposite value starts. Between events, both masks
have constant values, so the output has constant value too.

Two-pointer scan
----------------

We maintain:

- ``a``, ``b``: current run indices in each mask
- ``ra``, ``rb``: positions where current runs started
- Output runs accumulated in result

At each step::

    # Current values (even index = 0, odd index = 1)
    va = a & 1
    vb = b & 1
    vout = f(va, vb)  # boolean function

    # Find where next run ends
    end_a = ra + cnts_a[a]
    end_b = rb + cnts_b[b]
    next_event = min(end_a, end_b)

    # Extend output
    run_length = next_event - current_pos
    if vout == (output_run_index & 1):
        output[-1] += run_length  # extend current run
    else:
        output.append(run_length)  # start new run

    # Advance pointer(s)
    if end_a == next_event:
        ra = end_a
        a += 1
    if end_b == next_event:
        rb = end_b
        b += 1

This processes each input run exactly once: O(runs_a + runs_b).

The truth table encoding
------------------------

The boolean function f(va, vb) could be any of 16 possibilities (2^4 for
two binary inputs). Rather than writing 16 separate functions, we encode
the truth table in 4 bits::

    bit 0: f(0, 0)
    bit 1: f(0, 1)
    bit 2: f(1, 0)
    bit 3: f(1, 1)

To evaluate: ``f(va, vb) = (truth_table >> (va + 2*vb)) & 1``

Standard operations::

    AND:        0b1000 = 8   (only 1 when both are 1)
    OR:         0b1110 = 14  (1 unless both are 0)
    XOR:        0b0110 = 6   (1 when different)
    DIFF (A-B): 0b0010 = 2   (1 when A=1 and B=0)
    DIFF (B-A): 0b0100 = 4   (1 when B=1 and A=0)

This lets us write one merge function that handles all boolean operations.

Same-size requirement
---------------------

The merge algorithm assumes all masks have identical dimensions. The output
takes its height and width from the first mask. If masks differ in size,
results are undefined.

This is intentional: masks represent the same image region, so they should
be the same size. If you need to combine masks of different sizes, pad or
crop them first to match.

Multi-mask operations
---------------------

For more than two masks, we could chain pairwise merges::

    result = merge(merge(merge(m1, m2), m3), m4)

But that's O(n × runs) and builds intermediate results.

Better: merge all at once with k pointers, one per mask. At each step,
find the earliest event across all masks, compute the output value from
all k inputs, and advance the relevant pointers.

The truth table now has 2^k entries. For k=4 masks with AND, the table is
a single bit at position 15 (all inputs 1). For OR, it's all bits except
position 0.

Cost is O(total_runs) with O(k) work per event for finding the minimum
and computing the output. For small k, this is efficient.

Run count bounds
----------------

How many runs can the output have? In the worst case, every run boundary
in either input creates a new run in the output. So::

    output_runs <= runs_a + runs_b

But often it's much less. AND of two sparse masks produces a very sparse
result. OR of similar masks produces something close to either input.

The library allocates conservatively (sum of input runs) and shrinks
after the merge completes.

Special cases
-------------

**Complement**: Flip all bit values by shifting the run interpretation.
If the first run is zero-length, remove it (shifting indices down).
Otherwise, prepend a zero-length run (shifting indices up). This is O(1)
for the inplace version since extra space is pre-allocated.

**Self-operations**: A & A = A, A | A = A, A ^ A = 0, A - A = 0. The
library doesn't special-case these; the merge algorithm handles them
correctly with the same cost as any other merge.

**Empty masks**: A mask with no foreground is just one run of all zeros.
Merging with an empty mask is still O(runs) in the non-empty mask.

The code structure
------------------

The implementation in ``boolfuncs.c``::

    void rleMerge(RLE *a, RLE *b, RLE *out, uint truth_table) {
        // Allocate output (worst case: sum of input runs)
        // Two-pointer scan with truth table lookup
        // Shrink output to actual size
    }

    void rleAnd(RLE *a, RLE *b, RLE *out) {
        rleMerge(a, b, out, 0b1000);
    }

    void rleOr(RLE *a, RLE *b, RLE *out) {
        rleMerge(a, b, out, 0b1110);
    }

Higher-level operations (complement, difference, XOR) are built on the
same merge primitive with different truth tables.
