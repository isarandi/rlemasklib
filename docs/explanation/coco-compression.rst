COCO's LEB128-like Compression
==============================

Run-length encoding already compresses masks by storing run lengths instead
of pixels. But how do we store the run lengths themselves?

The naive approach uses fixed-width integers. A uint32 per run gives us
4 bytes × number of runs. For a mask with 100 runs, that's 400 bytes.
But most run lengths are small—a few hundred pixels at most. We're wasting
bits on leading zeros.

Variable-length encoding saves space by using fewer bytes for smaller
numbers.

LEB128 background
-----------------

LEB128 (Little Endian Base 128) is a standard variable-length integer
encoding used in DWARF debug info, WebAssembly, and Protocol Buffers.

The idea: use 7 bits per byte for data, and the high bit as a continuation
flag. If the high bit is 1, more bytes follow. If 0, this is the last byte.

Example: encoding 300 (binary: 100101100)

- Split into 7-bit chunks from the right: 0000010, 0101100
- Reverse order (little endian): 0101100, 0000010
- Add continuation bits: 10101100, 00000010
- Result: two bytes, 0xAC 0x02

Small numbers (0-127) fit in one byte. Larger numbers grow as needed.

COCO's modification
-------------------

COCO stores masks in JSON. Binary bytes would need base64 encoding or
escaping, adding overhead and complexity.

COCO's solution: use 6 bits per character instead of 8 bits per byte.
5 data bits + 1 continuation bit = 6 bits, which maps to 64 values.
Add 48 to get ASCII codes 48-111, which are the characters '0' through 'o'.

All printable, all JSON-safe, no escaping needed. The mask becomes a
readable string like ``"oW3a0f2N"`` in the JSON file.

The tradeoff: 5 data bits per character vs 7 data bits per byte means
roughly 40% more characters than a binary encoding would need bytes.
But JSON overhead for binary data (base64) would be 33% anyway, and this
stays human-inspectable.

Delta encoding
--------------

COCO adds another layer: delta encoding. Instead of storing raw run lengths,
store the difference from the previous run of the same type.

Runs alternate between 0s and 1s. Consecutive runs of 0s (indices 0, 2, 4, ...)
often have similar lengths. Same for runs of 1s (indices 1, 3, 5, ...).

So we encode::

    encoded[i] = cnts[i] - cnts[i-2]    (for i >= 2)
    encoded[i] = cnts[i]                (for i < 2)

For typical blob-like objects, this works well: RLE scans column by column,
and a continuous shape has similar vertical extent in neighboring columns.
Each column's foreground run length changes gradually, so deltas cluster
near zero. Small deltas need fewer characters.

Signed representation
---------------------

Deltas can be negative. The encoding handles this with sign extension.

Each 5-bit chunk has a sign bit (the 5th bit, value 0x10). When decoding
the last chunk, if this bit is set, we fill all higher bits with 1s,
giving a negative two's complement number.

Example: encoding -3

- -3 in two's complement (infinite width): ...11111101
- Take last 5 bits: 11101
- High bit (0x10) is set, so this signals negative
- No more chunks needed (remaining bits are all 1s)
- Add continuation bit (0): 011101
- Add 48: ASCII 77, which is 'M'

Decoding 'M':

- 'M' is ASCII 77, subtract 48: 29 = 0b011101
- Continuation bit (bit 5, 0x20): 0, so this is the last chunk
- Data bits (bits 0-4, 0x1f): 0b11101 = 29
- Sign bit (bit 4, 0x10): set, so sign-extend with 1s
- Result: ...11111 11101 = -3

Encoding walkthrough
--------------------

Let's encode the run lengths [8, 12, 6, 15].

First, compute deltas::

    cnts[0] = 8   → delta = 8   (no previous run of same type)
    cnts[1] = 12  → delta = 12  (no previous run of same type)
    cnts[2] = 6   → delta = 6 - 8 = -2
    cnts[3] = 15  → delta = 15 - 12 = 3

Now encode each delta:

**8**: binary 01000

- Fits in 5 bits, sign bit (0x10) is 0, matches sign (positive)
- No continuation needed
- Chunk: 01000 = 8, add 48 → ASCII 56 = '8'

**12**: binary 01100

- Fits in 5 bits, sign bit is 0, matches sign
- Chunk: 01100 = 12, add 48 → ASCII 60 = '<'

**-2**: binary ...111110

- Last 5 bits: 11110
- Sign bit is 1, remaining bits are all 1s, so we're done
- Chunk: 11110 = 30, add 48 → ASCII 78 = 'N'

**3**: binary 00011

- Fits in 5 bits, sign bit is 0, matches sign
- Chunk: 00011 = 3, add 48 → ASCII 51 = '3'

Result: ``"8<N3"``

Decoding walkthrough
--------------------

Decode ``"8<N3"`` back to run lengths.

**'8'**: 56 - 48 = 8 = 0b01000

- Continuation bit: 0, last chunk
- Data: 8, sign bit 0, no extension
- Delta: 8, cnts[0] = 8

**'<'**: 60 - 48 = 12 = 0b01100

- Continuation bit: 0, last chunk
- Data: 12, sign bit 0, no extension
- Delta: 12, cnts[1] = 12

**'N'**: 78 - 48 = 30 = 0b11110

- Continuation bit: 0, last chunk
- Data: 30 & 0x1f = 30, sign bit 1, extend with 1s
- Delta: -2, cnts[2] = cnts[0] + (-2) = 8 - 2 = 6

**'3'**: 51 - 48 = 3 = 0b00011

- Continuation bit: 0, last chunk
- Data: 3, sign bit 0, no extension
- Delta: 3, cnts[3] = cnts[1] + 3 = 12 + 3 = 15

Result: [8, 12, 6, 15]

Compression ratio
-----------------

For typical segmentation masks:

- Average ~1.5 characters per run (empirically measured on human pose datasets)
- Compare to 4 bytes per uint32: roughly 2.7× compression
- 99th percentile is still under 2 characters per run

Optional gzip compression (stored as ``zcounts`` instead of ``counts``)
adds another ~40% reduction for masks with repetitive structure.

The code
--------

Encoding in ``rleToString``::

    for each run i:
        x = cnts[i]
        if i > 2:
            x -= cnts[i-2]  # delta from same-type run

        do:
            c = x & 0x1f           # low 5 bits
            x >>= 5                # arithmetic shift
            more = (c & 0x10) ? (x != -1) : (x != 0)
            if more:
                c |= 0x20          # set continuation bit
            output(c + 48)
        while more

Decoding in ``rleFrString``::

    for each run:
        x = 0
        k = 0
        do:
            c = input() - 48
            x |= (c & 0x1f) << k
            more = c & 0x20
            k += 5
        while more

        if c & 0x10:               # sign extend
            x |= -1 << k

        if i > 2:
            x += cnts[i-2]         # undo delta
        cnts[i] = x
