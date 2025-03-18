#pragma once

#include <stdbool.h>
#include "basics.h"

static void transformAffine(const double in[2], double out[2], const double M[6]);
static void transformPerspective(const double inp[2], double outp[2], double H[9]);

// General matrix stuff
static void invert3x3(const double A[9], double A_inv[9]);
static void transpose_3x3(const double A[9], double A_T[9]);

// Similar to modulo but works for negative a the way it works in Python
static int int_remainder(int a, int b);

// Rotate and flip back an RLE to the original orientation
static void rleBackFlipRot(RLE *tmp, RLE *M, int k, bool flip);