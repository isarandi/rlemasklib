#pragma once
#include "basics.h"

// The main connected components function. Extracts components with either 4 or 8-neighborhood
// (8 means that diagonal neighbors are also considered to be connected, with 4, it's only the
// vertical and horizontal neighbors that count). Only those components are returned which have at
// least min_size pixels. The number of pixels of the components are tracked in the union-find node
// structure.
void rleConnectedComponents(
    const RLE *R_in, int connectivity, siz min_size, RLE **components, siz *n_components_out);

// Similar to rleConnectedComponents, but only keeps the largest connected component.
void rleLargestConnectedComponentInplace(RLE *R_in, int connectivity);

// Similar to rleConnectedComponents, but removes all components that are smaller than min_size.
// This is more efficient than retrieving all larger than min_size components and then merging them
// by union, since here we avoid the construction of many component RLEs and just create one
// directly. To fill small holes, one can apply this to the complement of the mask and then take the
// complement again.
void rleRemoveSmallConnectedComponentsInplace(RLE *R_in, siz min_size, int connectivity);

