#include "basics.h"

void rleDilateVerticalInplace(RLE *R, uint up, uint down);
void rleConcatHorizontal(const RLE **R, RLE *M, siz n);
void rleConcatVertical(const RLE **R, RLE *M, siz n);

void rleStrideInplace(RLE *R, siz sy, siz sx);
void rleRepeatInplace(RLE *R, siz nh, siz nw);
void rleRepeat(const RLE *R, RLE *M, siz nh, siz nw);
void rleContours(const RLE *R, RLE *M);

// Separable box average-pooling (valid, strided); see the definition in misc.c.
void rleAvgPoolBoxSeparable(
    const RLE *R, RLE *M, uint kh, uint kw, uint sh, uint sw, uint threshold);