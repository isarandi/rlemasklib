#pragma once
#include "basics.h"

void rleToBbox(const RLE *R, BB bb, siz n);
void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n);

void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w);
void rleFrCircle(RLE *R, const double *center_xy, double radius, siz h, siz w);

void rleToUintBbox(const RLE *R, uint *bb);