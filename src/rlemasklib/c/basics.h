#pragma once

#include <stdbool.h> // for bool

typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;
typedef double *BB;

// Rules
//  - All RLEs start with a run of 0s, which may have length 0.
//  - No run except the first may have a length of 0.
//  - The final run may not have a length of 0.
//  - An empty mask which contains no pixels (height or width is 0) can be represented in two ways:
//      - With 0 runs.
//      - With 1 run of length 0.
// Note: the allocated space starts earlier than cnts to allow for efficient prepending of a run
// to perform complement.
// alloc should not be used in normal computations, just in complement and realloc etc.
typedef struct {
    siz h;
    siz w;
    siz m;
    uint *cnts;
    uint *alloc;
} RLE;

// Initialize / destruct RLE
uint *rleInit(RLE *R, siz h, siz w, siz m);

// Init by copying from existing runlength counts
uint *rleFrCnts(RLE *R, siz h, siz w, siz m, uint *cnts);

// Move the allocated pointer from one RLE to another and copy h, w, m.
// This does not copy, but R's cnts will be set to NULL, so the data is logically transferred to M.
void rleMoveTo(RLE *R, RLE *M);

// Copy the runlength counts and h, w, m from one RLE to another.
void rleCopy(const RLE *R, RLE *M);

// Free the memory allocated for the runlength counts.
void rleFree(RLE *R);


// Initialize/destroy RLE array
void rlesInit(RLE **R, siz n);

void rlesFree(RLE **R, siz n);

// Initialize RLEs full of 1s or 0s
void rleOnes(RLE *R, siz h, siz w); // this is the fully-foreground mask
void rleZeros(RLE *R, siz h, siz w); // this is the fully-background mask

// Get and set an individual pixel
byte rleGet(const RLE *R, siz i, siz j);

void rleSetInplace(RLE *R, siz y, siz x, byte value);

// Check whether two RLEs are equal (in size and pixel content)
bool rleEqual(const RLE *A, const RLE *B);

// Remove any non-first zero runlengths by adding and shifting the runs accordingly.
static void rleEliminateZeroRuns(RLE *R);

// Reallocate the runlength counts to have m runs, which may be more or less than the current
// number of runs.
static uint *rleRealloc(RLE *R, siz m);

// Swap the contents of two RLEs, i.e. the pointers and the h, w, m values.
// This is useful in double buffer ping-pong style algorithms, e.g. rleMerge.
static void rleSwap(RLE *R, RLE *M);