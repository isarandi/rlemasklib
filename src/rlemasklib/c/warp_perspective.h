#pragma once
#include "basics.h"

// Returns false if the transform is degenerate (output is then an all-zeros mask).
bool rleWarpPerspective(const RLE *R, RLE *M, siz h_out, siz w_out, double *H);
