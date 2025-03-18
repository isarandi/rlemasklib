#pragma once
#include "basics.h"

// Functions for flipping, transposing and rotating
// any multiple of 90 degrees can be achieved with a combination of vertical flipping, transposing
// and rotating by 180 degrees

// This transpose algorithm is similar to [1] and [2].
// [1] Shoji, K. (1995). An algorithm for affine transformation of binary images stored in pxy tables
// by run format. Systems and Computers in Japan, 26(7), 69–78. doi:10.1002/scj.4690260707
// [2] V. Misra, J. F. Arias and A. K. Chhabra, "A memory efficient method for fast transposing
// run-length encoded images," Proceedings of the Fifth International Conference on Document
// Analysis and Recognition. ICDAR '99 (Cat. No.PR00318), Bangalore, India, 1999, pp. 161-164,
// doi: 10.1109/ICDAR.1999.791749.
void rleTranspose(const RLE *R, RLE *M);

// This is equivalent to linearizing the pixel data in row-major order then rolling the pixels
// to the right, shifting out the last pixel and shifting in a zero at the start, and then reshaping
// and re-encoding. But in this implementation this is done more efficiently by moving columns
// through padding and cropping.
// This is a first step towards obtaining the horizontal edge map that is used in transposing.
void rleRoll(const RLE *R, RLE *M);

// Vertical flip is done by scanning until a run is found that has a bottom pixel, then
// going backwards to copy the runs of the column in reverse order, so we also have to remember the
// index of the run that has the top pixel of the column and how many pixels of it are in the current column.
void rleVerticalFlip(const RLE* R, RLE* M);

// Rotation by 180 degrees is just reversing the order of the runs (paying attention that
// the first run is always a run of 0s).
void rleRotate180Inplace(RLE *R);
void rleRotate180(const RLE *R, RLE *M);

