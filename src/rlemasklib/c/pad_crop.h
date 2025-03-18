#pragma once
#include "basics.h"

// Crop the RLE mask to the bounding box. The bounding box is specified as {x, y, w, h}
// where x, y is the top-left corner of the box.
void rleCrop(const RLE *R, RLE *M, siz n, const uint *bbox);
void rleCropInplace(RLE *R, siz n, const uint *bbox);

// Pad the RLE mask with zeros. The pad amounts are {left, right, top, bottom}.
// If padding by 1 is needed, just complement the input, pad with 0s and complement again.
void rleZeroPad(const RLE *R, RLE *M, siz n, const uint *pad_amounts);
void rleZeroPadInplace(RLE *R, siz n, const uint *pad_amounts);

// Pad the RLE with the same value as the edge.
// Like cv2.BORDER_REPLICATE in OpenCV and np.pad(..., mode='edge') in numpy.
void rlePadReplicate(const RLE *R, RLE *M, const uint *pad_amounts);
// Following are helpers for rlePadReplicate

