#pragma once
#include "basics.h"
#include <stdbool.h>

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

// =============================================================================
// Two-phase API: compute stats first, then optionally extract selected components
// =============================================================================

// Opaque state handle for two-phase extraction
typedef struct CCState CCState;

// Phase 1: Build union-find structure and compute component stats.
// Returns a state handle and outputs stats arrays (caller owns the arrays).
// areas_out: array of n_components area values
// bboxes_out: array of n_components*4 (x, y, w, h per component)
// centroids_out: array of n_components*2 (x, y per component)
CCState* rleConnectedComponentsBegin(
    const RLE *R_in,
    int connectivity,
    siz min_size,
    siz *n_components_out,
    siz **areas_out,
    int **bboxes_out,
    double **centroids_out);

// Phase 2: Extract only the selected components as RLE masks.
// selected: boolean array of length n_components (from Phase 1)
// components_out: output array of RLE masks for selected components
// n_selected_out: number of selected components
void rleConnectedComponentsExtract(
    CCState *state,
    bool *selected,
    RLE **components_out,
    siz *n_selected_out);

// Cleanup: Free the state handle and internal resources.
void rleConnectedComponentsEnd(CCState *state);

// =============================================================================
// Fast paths: stats-only and count-only
// =============================================================================

// Get component stats without extracting component RLEs.
// Returns number of components; outputs are allocated arrays (caller owns them).
siz rleConnectedComponentStats(
    const RLE *R_in,
    int connectivity,
    siz min_size,
    siz **areas_out,
    int **bboxes_out,
    double **centroids_out);

// Count components without extracting them.
siz rleCountConnectedComponents(const RLE *R_in, int connectivity, siz min_size);

