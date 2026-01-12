#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "basics.h"
#include "minmax.h"
#include "connected_components.h"

// =============================================================================
// Types
// =============================================================================

struct UnionFindNode {
    struct UnionFindNode *parent;
    siz size;
};

struct RLEComponentStats {
    siz area;
    uint bbox_x, bbox_y, bbox_w, bbox_h;
    double centroid_x, centroid_y;
};

struct CCState {
    const RLE *R_in;
    RLE *R_split;
    const RLE *R;
    struct UnionFindNode *uf;
    uint *new_labels;
    uint *component_run_counts;
    siz n_components;
    struct RLEComponentStats *stats;
};

// =============================================================================
// Forward declarations
// =============================================================================

static struct UnionFindNode *_uf_find(struct UnionFindNode *x);
static void _uf_union(struct UnionFindNode *x, struct UnionFindNode *y);

static RLE *_rleSplitRunsThatMayBelongToDifferentComponents(const RLE *R, int connectivity);

static struct UnionFindNode *_rleBuildComponentsUF(
    const RLE *R_in, int connectivity, RLE **R_split_out, const RLE **R_out);
static void _rleCleanupUF(struct UnionFindNode *uf, RLE *R_split);

static siz _rleAssignComponentLabels(
    const RLE *R, struct UnionFindNode *uf, siz min_size,
    uint *new_labels, uint *component_run_counts);
static void _rleComputeComponentStats(
    const RLE *R, uint *new_labels, siz n_components,
    struct RLEComponentStats *stats);
static void _rleAllocateComponentRLEs(
    RLE *rles_out, siz n_components, siz h, siz w,
    uint first_zeros, uint *component_run_counts);
static void _rleFillComponentRLEs(
    RLE *rles_out, siz n_components, const RLE *R, uint *new_labels);

// =============================================================================
// Public API: Two-phase extraction with stats
// =============================================================================

CCState* rleConnectedComponentsBegin(
    const RLE *R_in,
    int connectivity,
    siz min_size,
    siz *n_components_out,
    siz **areas_out,
    int **bboxes_out,
    double **centroids_out
) {
    CCState *state = malloc(sizeof(CCState));
    state->R_in = R_in;
    state->R_split = NULL;
    state->R = NULL;
    state->uf = NULL;
    state->new_labels = NULL;
    state->component_run_counts = NULL;
    state->n_components = 0;
    state->stats = NULL;

    struct UnionFindNode *uf = _rleBuildComponentsUF(
        R_in, connectivity, &state->R_split, &state->R);

    if (!uf) {
        *n_components_out = 0;
        *areas_out = NULL;
        *bboxes_out = NULL;
        *centroids_out = NULL;
        return state;
    }

    state->uf = uf;
    siz m = state->R->m;

    state->new_labels = malloc(sizeof(uint) * (m / 2));
    state->component_run_counts = malloc(sizeof(uint) * (m / 2));

    siz n_components = _rleAssignComponentLabels(
        state->R, uf, min_size,
        state->new_labels, state->component_run_counts);

    state->n_components = n_components;
    *n_components_out = n_components;

    if (n_components == 0) {
        *areas_out = NULL;
        *bboxes_out = NULL;
        *centroids_out = NULL;
        return state;
    }

    // Compute stats
    state->stats = malloc(sizeof(struct RLEComponentStats) * n_components);
    _rleComputeComponentStats(state->R, state->new_labels, n_components, state->stats);

    // Copy to output arrays (caller owns these)
    *areas_out = malloc(sizeof(siz) * n_components);
    *bboxes_out = malloc(sizeof(int) * n_components * 4);
    *centroids_out = malloc(sizeof(double) * n_components * 2);

    for (siz i = 0; i < n_components; i++) {
        (*areas_out)[i] = state->stats[i].area;
        (*bboxes_out)[i * 4 + 0] = state->stats[i].bbox_x;
        (*bboxes_out)[i * 4 + 1] = state->stats[i].bbox_y;
        (*bboxes_out)[i * 4 + 2] = state->stats[i].bbox_w;
        (*bboxes_out)[i * 4 + 3] = state->stats[i].bbox_h;
        (*centroids_out)[i * 2 + 0] = state->stats[i].centroid_x;
        (*centroids_out)[i * 2 + 1] = state->stats[i].centroid_y;
    }

    return state;
}

void rleConnectedComponentsExtract(
    CCState *state,
    bool *selected,
    RLE **components_out,
    siz *n_selected_out
) {
    if (!state || state->n_components == 0) {
        *components_out = NULL;
        *n_selected_out = 0;
        return;
    }

    const RLE *R = state->R;
    siz m = R->m;
    uint *cnts = R->cnts;

    // Count selected and build old->new index mapping
    siz n_selected = 0;
    uint *old_to_new = malloc(sizeof(uint) * state->n_components);
    for (siz i = 0; i < state->n_components; i++) {
        if (selected[i]) {
            old_to_new[i] = n_selected++;
        } else {
            old_to_new[i] = m;  // marker for "not selected"
        }
    }

    *n_selected_out = n_selected;
    if (n_selected == 0) {
        *components_out = NULL;
        free(old_to_new);
        return;
    }

    // Recompute run counts for selected components only
    uint *selected_run_counts = calloc(n_selected, sizeof(uint));
    for (siz i = 0; i < n_selected; i++) {
        selected_run_counts[i] = 1;  // leading zeros
    }

    for (siz i = 1; i < m; i += 2) {
        uint old_label = state->new_labels[i / 2];
        if (old_label == m) continue;  // filtered by min_size

        uint new_label = old_to_new[old_label];
        if (new_label == m) continue;  // not selected by user

        if (i == 1 || old_to_new[state->new_labels[i / 2 - 1]] != new_label || cnts[i - 1] > 0) {
            selected_run_counts[new_label] += 2;
        }
    }

    // Allocate output RLEs
    rlesInit(components_out, n_selected);
    RLE *rles_out = *components_out;
    for (siz i = 0; i < n_selected; i++) {
        rleInit(&rles_out[i], R->h, R->w, selected_run_counts[i]);
        rles_out[i].cnts[0] = cnts[0];
    }

    // Track write position for each selected component
    uint **write_ptrs = malloc(sizeof(uint *) * n_selected);
    for (siz i = 0; i < n_selected; i++) {
        write_ptrs[i] = rles_out[i].cnts + 1;
    }

    // Fill selected component RLEs
    for (siz i = 1; i < m; i += 2) {
        uint current_1s = cnts[i];
        uint next_0s = (i + 1 < m) ? cnts[i + 1] : 0;

        uint old_label = state->new_labels[i / 2];
        uint new_label = (old_label != m) ? old_to_new[old_label] : m;

        if (new_label == m) {
            // Not selected - extend zeros in all selected components
            for (siz c = 0; c < n_selected; c++) {
                write_ptrs[c][-1] += current_1s + next_0s;
            }
        } else {
            // Add run to this component
            if (write_ptrs[new_label] - rles_out[new_label].cnts > 1 &&
                write_ptrs[new_label][-1] == 0) {
                write_ptrs[new_label][-2] += current_1s;
                write_ptrs[new_label][-1] += next_0s;
            } else {
                write_ptrs[new_label][0] = current_1s;
                write_ptrs[new_label][1] = next_0s;
                write_ptrs[new_label] += 2;
            }

            // Extend zeros in other selected components
            for (siz c = 0; c < n_selected; c++) {
                if (c != new_label) {
                    write_ptrs[c][-1] += current_1s + next_0s;
                }
            }
        }
    }

    // Trim trailing empty runs
    for (siz i = 0; i < n_selected; i++) {
        RLE *rle = &rles_out[i];
        if (rle->m > 0 && rle->cnts[rle->m - 1] == 0) {
            rleRealloc(rle, rle->m - 1);
        }
    }

    free(old_to_new);
    free(selected_run_counts);
    free(write_ptrs);
}

void rleConnectedComponentsEnd(CCState *state) {
    if (!state) return;

    if (state->stats) free(state->stats);
    if (state->new_labels) free(state->new_labels);
    if (state->component_run_counts) free(state->component_run_counts);
    _rleCleanupUF(state->uf, state->R_split);
    free(state);
}

// =============================================================================
// Public API: Extract all components (convenience wrapper)
// =============================================================================

void rleConnectedComponents(
    const RLE *R_in, int connectivity, siz min_size,
    RLE **components, siz *n_components_out
) {
    RLE *R_split;
    const RLE *R;
    struct UnionFindNode *uf = _rleBuildComponentsUF(R_in, connectivity, &R_split, &R);

    if (!uf) {
        *n_components_out = 0;
        *components = NULL;
        return;
    }

    siz m = R->m;
    uint *new_labels = malloc(sizeof(uint) * (m / 2));
    uint *component_run_counts = malloc(sizeof(uint) * (m / 2));

    siz n_components = _rleAssignComponentLabels(
        R, uf, min_size, new_labels, component_run_counts);

    if (n_components == 0) {
        *n_components_out = 0;
        *components = NULL;
        free(new_labels);
        free(component_run_counts);
        _rleCleanupUF(uf, R_split);
        return;
    }

    rlesInit(components, n_components);
    _rleAllocateComponentRLEs(
        *components, n_components, R->h, R->w, R->cnts[0], component_run_counts);
    _rleFillComponentRLEs(*components, n_components, R, new_labels);

    *n_components_out = n_components;
    free(new_labels);
    free(component_run_counts);
    _rleCleanupUF(uf, R_split);
}

// =============================================================================
// Public API: Stats only (no component RLEs)
// =============================================================================

siz rleConnectedComponentStats(
    const RLE *R_in,
    int connectivity,
    siz min_size,
    siz **areas_out,
    int **bboxes_out,
    double **centroids_out
) {
    RLE *R_split;
    const RLE *R;
    struct UnionFindNode *uf = _rleBuildComponentsUF(R_in, connectivity, &R_split, &R);

    if (!uf) {
        *areas_out = NULL;
        *bboxes_out = NULL;
        *centroids_out = NULL;
        return 0;
    }

    siz m = R->m;
    uint *new_labels = malloc(sizeof(uint) * (m / 2));
    uint *component_run_counts = malloc(sizeof(uint) * (m / 2));

    siz n_components = _rleAssignComponentLabels(
        R, uf, min_size, new_labels, component_run_counts);

    if (n_components == 0) {
        *areas_out = NULL;
        *bboxes_out = NULL;
        *centroids_out = NULL;
        free(new_labels);
        free(component_run_counts);
        _rleCleanupUF(uf, R_split);
        return 0;
    }

    struct RLEComponentStats *stats = malloc(sizeof(struct RLEComponentStats) * n_components);
    _rleComputeComponentStats(R, new_labels, n_components, stats);

    *areas_out = malloc(sizeof(siz) * n_components);
    *bboxes_out = malloc(sizeof(int) * n_components * 4);
    *centroids_out = malloc(sizeof(double) * n_components * 2);

    for (siz i = 0; i < n_components; i++) {
        (*areas_out)[i] = stats[i].area;
        (*bboxes_out)[i * 4 + 0] = stats[i].bbox_x;
        (*bboxes_out)[i * 4 + 1] = stats[i].bbox_y;
        (*bboxes_out)[i * 4 + 2] = stats[i].bbox_w;
        (*bboxes_out)[i * 4 + 3] = stats[i].bbox_h;
        (*centroids_out)[i * 2 + 0] = stats[i].centroid_x;
        (*centroids_out)[i * 2 + 1] = stats[i].centroid_y;
    }

    free(stats);
    free(new_labels);
    free(component_run_counts);
    _rleCleanupUF(uf, R_split);
    return n_components;
}

// =============================================================================
// Public API: Count only
// =============================================================================

siz rleCountConnectedComponents(const RLE *R_in, int connectivity, siz min_size) {
    RLE *R_split;
    const RLE *R;
    struct UnionFindNode *uf = _rleBuildComponentsUF(R_in, connectivity, &R_split, &R);
    if (!uf) return 0;

    siz count = 0;
    for (siz i = 0; i < R->m / 2; i++) {
        if (uf[i].parent == NULL && uf[i].size >= min_size) {
            count++;
        }
    }

    _rleCleanupUF(uf, R_split);
    return count;
}

// =============================================================================
// Public API: Inplace filtering
// =============================================================================

void rleRemoveSmallConnectedComponentsInplace(RLE *R_in, siz min_size, int connectivity) {
    RLE *R_split;
    const RLE *R;
    struct UnionFindNode *uf = _rleBuildComponentsUF(R_in, connectivity, &R_split, &R);
    if (!uf) return;

    siz m = R->m;
    uint *cnts = R->cnts;

    siz i_out = 1;
    for (siz i = 1; i < m; i += 2) {
        struct UnionFindNode *root = _uf_find(&uf[i / 2]);
        uint current_count_1s = cnts[i];
        uint next_count_0s = (i + 1 < m) ? cnts[i + 1] : 0;

        if (root->size < min_size) {
            R_in->cnts[i_out - 1] += current_count_1s + next_count_0s;
        } else if (i_out > 1 && R_in->cnts[i_out - 1] == 0) {
            R_in->cnts[i_out - 2] += current_count_1s;
            R_in->cnts[i_out - 1] += next_count_0s;
        } else {
            R_in->cnts[i_out++] = current_count_1s;
            if (i + 1 < m) {
                R_in->cnts[i_out++] = next_count_0s;
            }
        }
    }

    R_in->m = i_out;
    rleEliminateZeroRuns(R_in);
    _rleCleanupUF(uf, R_split);
}

void rleLargestConnectedComponentInplace(RLE *R_in, int connectivity) {
    RLE *R_split;
    const RLE *R;
    struct UnionFindNode *uf = _rleBuildComponentsUF(R_in, connectivity, &R_split, &R);
    if (!uf) return;

    siz m = R->m;
    uint *cnts = R->cnts;

    siz max_size = 0;
    struct UnionFindNode *max_node = NULL;
    for (siz i = 0; i < m / 2; i++) {
        if (uf[i].parent == NULL && uf[i].size > max_size) {
            max_size = uf[i].size;
            max_node = &uf[i];
        }
    }

    siz i_out = 1;
    for (siz i = 1; i < m; i += 2) {
        struct UnionFindNode *root = _uf_find(&uf[i / 2]);
        uint current_count_1s = cnts[i];
        uint next_count_0s = (i + 1 < m) ? cnts[i + 1] : 0;

        if (root != max_node) {
            R_in->cnts[i_out - 1] += current_count_1s + next_count_0s;
        } else if (i_out > 1 && R_in->cnts[i_out - 1] == 0) {
            R_in->cnts[i_out - 2] += current_count_1s;
            R_in->cnts[i_out - 1] += next_count_0s;
        } else {
            R_in->cnts[i_out++] = current_count_1s;
            if (i + 1 < m) {
                R_in->cnts[i_out++] = next_count_0s;
            }
        }
    }

    R_in->m = i_out;
    rleEliminateZeroRuns(R_in);
    _rleCleanupUF(uf, R_split);
}

// =============================================================================
// Helper: Build union-find structure
// =============================================================================

static struct UnionFindNode *_rleBuildComponentsUF(
    const RLE *R_in,
    int connectivity,
    RLE **R_split_out,
    const RLE **R_out
) {
    *R_split_out = NULL;
    *R_out = R_in;

    if (R_in->m <= 1) {
        return NULL;
    }

    bool diagonal = (connectivity == 8);
    RLE *R_split = _rleSplitRunsThatMayBelongToDifferentComponents(R_in, connectivity);
    const RLE *R = R_split ? R_split : R_in;

    *R_split_out = R_split;
    *R_out = R;

    siz m = R->m;
    siz h = R->h;
    uint *cnts = R->cnts;

    struct UnionFindNode *uf = calloc(m / 2, sizeof(struct UnionFindNode));
    for (siz i = 1; i < m; i += 2) {
        uf[i / 2].size = cnts[i];
    }

    siz i1 = 1, i2 = 1, r1 = cnts[0], r2 = cnts[0];
    while (i1 < m && i2 < m) {
        siz overlap_start = uintMax(r1 + h, r2);
        siz overlap_end = uintMin(r1 + cnts[i1] + h, r2 + cnts[i2]);
        if (overlap_start < overlap_end ||
            (diagonal && overlap_start == overlap_end && overlap_start % h != 0)) {
            _uf_union(&uf[i1 / 2], &uf[i2 / 2]);
        }

        if (r1 + cnts[i1] + h < r2 + cnts[i2]) {
            if (i1 + 1 >= m) break;
            r1 += cnts[i1] + cnts[i1 + 1];
            i1 += 2;
        } else {
            if (i2 + 1 >= m) break;
            r2 += cnts[i2] + cnts[i2 + 1];
            i2 += 2;
        }
    }

    return uf;
}

static void _rleCleanupUF(struct UnionFindNode *uf, RLE *R_split) {
    if (R_split) {
        rleFree(R_split);
        free(R_split);
    }
    if (uf) free(uf);
}

// =============================================================================
// Helper: Split runs that may belong to different components
// =============================================================================

static RLE *_rleSplitRunsThatMayBelongToDifferentComponents(const RLE *R, int connectivity) {
    uint h = R->h;
    uint w = R->w;
    siz m = R->m;

    if (m <= 1 || h == 0 || w == 0) {
        return NULL;
    }
    bool diagonal = (connectivity == 8);
    siz r = 0;
    siz n_splits = 0;
    for (siz j = 0; j < m; j++) {
        uint cnt = R->cnts[j];
        if (j % 2 == 1 && cnt <= h + (diagonal ? 1 : 0) && (r + cnt - 1) / h > r / h) {
            n_splits++;
        }
        r += cnt;
    }

    if (n_splits == 0) {
        return NULL;
    }

    RLE *M = malloc(sizeof(RLE));
    uint *cnts_out = rleInit(M, h, w, m + 2 * n_splits);
    r = 0;
    siz i_out = 0;
    for (siz j = 0; j < m; j++) {
        uint cnt = R->cnts[j];
        if (j % 2 == 1 && cnt <= h + (diagonal ? 1 : 0) && (r + cnt - 1) / h > r / h) {
            cnts_out[i_out] = h - r % h;
            cnts_out[i_out + 1] = 0;
            cnts_out[i_out + 2] = cnt - cnts_out[i_out];
            i_out += 3;
        } else {
            cnts_out[i_out++] = cnt;
        }
        r += cnt;
    }
    return M;
}

// =============================================================================
// Helper: Assign labels and count runs per component
// =============================================================================

static siz _rleAssignComponentLabels(
    const RLE *R,
    struct UnionFindNode *uf,
    siz min_size,
    uint *new_labels,
    uint *component_run_counts
) {
    siz m = R->m;
    uint *cnts = R->cnts;

    uint *root_to_component = malloc(sizeof(uint) * (m / 2));
    for (siz i = 0; i < m / 2; i++) {
        root_to_component[i] = m;
    }

    siz n_components = 0;

    for (siz i = 1; i < m; i += 2) {
        struct UnionFindNode *root = _uf_find(&uf[i / 2]);

        if (root->size < min_size) {
            new_labels[i / 2] = m;
            continue;
        }

        uint root_idx = (uint)(root - uf);
        uint component;

        if (root_to_component[root_idx] == m) {
            component = n_components++;
            root_to_component[root_idx] = component;
            component_run_counts[component] = 1;
        } else {
            component = root_to_component[root_idx];
        }

        new_labels[i / 2] = component;

        if (i == 1 || component != new_labels[i / 2 - 1] || cnts[i - 1] > 0) {
            component_run_counts[component] += 2;
        }
    }

    free(root_to_component);
    return n_components;
}

// =============================================================================
// Helper: Compute stats for each component
// =============================================================================

static void _rleComputeComponentStats(
    const RLE *R,
    uint *new_labels,
    siz n_components,
    struct RLEComponentStats *stats
) {
    siz m = R->m;
    siz h = R->h;
    uint *cnts = R->cnts;

    // Initialize stats
    for (siz i = 0; i < n_components; i++) {
        stats[i].area = 0;
        stats[i].bbox_x = UINT32_MAX;
        stats[i].bbox_y = UINT32_MAX;
        stats[i].bbox_w = 0;  // will compute from max
        stats[i].bbox_h = 0;
        stats[i].centroid_x = 0;
        stats[i].centroid_y = 0;
    }

    // Temp storage for max coordinates
    uint *max_x = calloc(n_components, sizeof(uint));
    uint *max_y = calloc(n_components, sizeof(uint));

    // Accumulate stats from runs
    siz r = cnts[0];
    for (siz i = 1; i < m; i += 2) {
        uint component = new_labels[i / 2];
        uint run_len = cnts[i];

        if (component != m) {
            uint col_start = r / h;
            uint row_start = r % h;
            uint col_end = (r + run_len - 1) / h;
            uint row_end = (r + run_len - 1) % h;

            stats[component].area += run_len;

            // Update bbox x (column) range
            if (col_start < stats[component].bbox_x) stats[component].bbox_x = col_start;
            if (col_end > max_x[component]) max_x[component] = col_end;

            // Update bbox y (row) range
            // If run spans multiple columns, it touches all rows in between
            if (col_start < col_end) {
                stats[component].bbox_y = 0;
                max_y[component] = h - 1;
            } else {
                if (row_start < stats[component].bbox_y) stats[component].bbox_y = row_start;
                if (row_end > max_y[component]) max_y[component] = row_end;
            }

            // Centroid accumulation - handle multi-column runs in O(1)
            // First partial column (may be full or partial)
            uint cnt1 = uintMin(run_len, h - row_start);
            stats[component].centroid_x += (double)col_start * cnt1;
            stats[component].centroid_y += (row_start + (cnt1 - 1) * 0.5) * cnt1;

            if (cnt1 < run_len) {
                // Full middle columns
                uint num_full_cols = (run_len - cnt1) / h;
                uint cnt2 = num_full_cols * h;
                if (cnt2) {
                    stats[component].centroid_x += (col_start + 1 + (num_full_cols - 1) * 0.5) * cnt2;
                    stats[component].centroid_y += ((h - 1) * 0.5) * cnt2;
                }

                // Last partial column
                uint cnt3 = run_len - cnt1 - cnt2;
                if (cnt3) {
                    stats[component].centroid_x += (double)(col_start + num_full_cols + 1) * cnt3;
                    stats[component].centroid_y += ((cnt3 - 1) * 0.5) * cnt3;
                }
            }
        }

        r += run_len;
        if (i + 1 < m) {
            r += cnts[i + 1];
        }
    }

    // Finalize: compute width/height from max, divide centroid sums by area
    for (siz i = 0; i < n_components; i++) {
        if (stats[i].area > 0) {
            stats[i].bbox_w = max_x[i] - stats[i].bbox_x + 1;
            stats[i].bbox_h = max_y[i] - stats[i].bbox_y + 1;
            stats[i].centroid_x /= stats[i].area;
            stats[i].centroid_y /= stats[i].area;
        }
    }

    free(max_x);
    free(max_y);
}

// =============================================================================
// Helper: Allocate component RLEs
// =============================================================================

static void _rleAllocateComponentRLEs(
    RLE *rles_out,
    siz n_components,
    siz h, siz w,
    uint first_zeros,
    uint *component_run_counts
) {
    for (siz i = 0; i < n_components; i++) {
        rleInit(&rles_out[i], h, w, component_run_counts[i]);
        rles_out[i].cnts[0] = first_zeros;
    }
}

// =============================================================================
// Helper: Fill component RLEs
// =============================================================================

static void _rleFillComponentRLEs(
    RLE *rles_out,
    siz n_components,
    const RLE *R,
    uint *new_labels
) {
    siz m = R->m;
    uint *cnts = R->cnts;

    uint **write_ptrs = malloc(sizeof(uint *) * n_components);
    for (siz i = 0; i < n_components; i++) {
        write_ptrs[i] = rles_out[i].cnts + 1;
    }

    for (siz i = 1; i < m; i += 2) {
        uint current_1s = cnts[i];
        uint next_0s = (i + 1 < m) ? cnts[i + 1] : 0;
        uint component = new_labels[i / 2];

        if (component == m) {
            for (siz c = 0; c < n_components; c++) {
                write_ptrs[c][-1] += current_1s + next_0s;
            }
        } else {
            if (write_ptrs[component] - rles_out[component].cnts > 1 &&
                write_ptrs[component][-1] == 0) {
                write_ptrs[component][-2] += current_1s;
                write_ptrs[component][-1] += next_0s;
            } else {
                write_ptrs[component][0] = current_1s;
                write_ptrs[component][1] = next_0s;
                write_ptrs[component] += 2;
            }

            for (siz c = 0; c < n_components; c++) {
                if (c != component) {
                    write_ptrs[c][-1] += current_1s + next_0s;
                }
            }
        }
    }

    uint last_component = new_labels[m / 2 - 1];
    if (last_component != m) {
        RLE *last = &rles_out[last_component];
        if (last->cnts[last->m - 1] == 0) {
            rleRealloc(last, last->m - 1);
        }
    }

    free(write_ptrs);
}

// =============================================================================
// Union-find implementation
// =============================================================================

static struct UnionFindNode *_uf_find(struct UnionFindNode *x) {
    struct UnionFindNode *z = x;
    while (z->parent) {
        z = z->parent;
    }
    while (x->parent) {
        struct UnionFindNode *tmp = x->parent;
        x->parent = z;
        x = tmp;
    }
    return z;
}

static void _uf_union(struct UnionFindNode *x, struct UnionFindNode *y) {
    x = _uf_find(x);
    y = _uf_find(y);
    if (x == y) {
        return;
    }
    if (x->size < y->size) {
        x->parent = y;
        y->size += x->size;
    } else {
        y->parent = x;
        x->size += y->size;
    }
}
