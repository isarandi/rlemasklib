#pragma once
#include "basics.h"

void rleLargestInteriorRectangle(const RLE *R_, uint* rect_out);
void rleLargestInteriorRectangleAspect(const RLE *R_, double* rect_out, double aspect_ratio);
void rleLargestInteriorRectangleAroundCenter(const RLE *R, double* rect_out, uint cy, uint cx, double aspect_ratio);