#pragma once
#include "basics.h"

void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o);
void rleNms(RLE *dt, siz n, uint *keep, double thr);

void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o);
void bbNms(BB dt, siz n, uint *keep, double thr);