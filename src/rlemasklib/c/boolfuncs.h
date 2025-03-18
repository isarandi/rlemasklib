#pragma once

#include <stdint.h>
#include "basics.h"

extern const uint BOOLFUNC_AND;
extern const uint BOOLFUNC_OR;
extern const uint BOOLFUNC_XOR;
extern const uint BOOLFUNC_SUB;

void rleComplement(const RLE *R, RLE *M, siz n);
void rleComplementInplace(RLE *R, siz n);

// rleMerge merges n masks one by one into an accumulator mask, using a binary boolean function
// encoded as a uint. The integer's bits represent the truth table of the boolean function.
void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc);

// rleMerge2 is like rleMerge, but specialized for the common case of merging two masks,
// making it a bit more efficient than using rleMerge with n=2
void rleMerge2(const RLE *A, const RLE *B, RLE *M, uint boolfunc);

// Same as rleMerge, but RLEs are given as pointers, so they don't have to be contiguous in memory
void rleMergePtr(const RLE **R, RLE *M, siz n, uint boolfunc);

// rleMergeMultiFunc merges n masks one by one into an accumulator mask, using n-1 different binary
// boolean functions, so one can do A || B && C || D, for example. The boolean functions are given
// in the same format as in rleMerge.
void rleMergeMultiFunc(const RLE **R, RLE *M, siz n, uint* boolfuncs);

// The following functions use _rleMergeCustom:
// rleMergeLookup uses a n-ary boolean function encoded as a sequence of uint64_t values, whose
// bits represent the truth table of the boolean function. The output is generated in one pass
// since access to the values of each input mask is required simulataneously for n-ary functions.
void rleMergeLookup(const RLE **R, RLE *M, siz n, uint64_t *multiboolfunc, siz n_funcparts);

// DiffOr(A, B, C) = A && ~B || C
void rleMergeDiffOr(const RLE *A, const RLE *B, const RLE *C, RLE *M);

// in WeightedAtLeast, the sum of weights for 1 pixels must be at least `threshold`
void rleMergeWeightedAtLeast(const RLE **R, RLE *M, siz n, double *weights, double threshold);
// in AtLeast, at least `k` masks must have a 1 pixel at a position to set it to 1 in the output
void rleMergeAtLeast(const RLE **R, RLE *M, siz n, uint k);

// The following do not use _rleMergeCustom, but are specialized versions of rleMergePtr
// It turns out that packing the values into an int, and figuring out the number of 1s is
// not very performant, so the following are specialized versions or rleMergePtr where the counting
// is done explicitly, a counter is incremented and decremented each time we exit or enter a run of 1s
// in any of the input masks, which is faster.
void rleMergeAtLeast2(const RLE **R, RLE *M, siz n, uint k);

// rleMergeWeightedAtLeast2 adds and subtracts the weights of the input masks as we enter or exit
// runs of 1s. To make sure we don't drift much while adding weights so many times, a Kahan sum is used.
void rleMergeWeightedAtLeast2(
    const RLE **R, RLE *M, siz n, double *weights, double threshold);


