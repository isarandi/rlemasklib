#pragma once
#include "basics.h"

void rleArea(const RLE *R, siz n, uint *a);
void rleCentroid(const RLE *R, double *xys, siz n);
void rleNonZeroIndices(const RLE *R, uint **coords_out, siz *n_out);
void rleRawMoments(const RLE *R, double *moments);
void rleMoments(const RLE *R, double *out);
void rleHuMoments(const RLE *R, double *hu);
