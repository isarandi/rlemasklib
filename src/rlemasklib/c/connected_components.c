#include <stdbool.h> // for bool
#include <stddef.h> // for NULL
#include <stdlib.h> // for malloc, free
#include "basics.h"
#include "minmax.h"
#include "connected_components.h"


static RLE *_rleSplitRunsThatMayBelongToDifferentComponents(const RLE *R, int connectivity);

// Union-find data structure for tracking connected components
struct UnionFindNode {
    struct UnionFindNode *parent;
    siz size;
};
static struct UnionFindNode *_uf_find(struct UnionFindNode *x);
static void _uf_union(struct UnionFindNode *x, struct UnionFindNode *y);


// Implementation:
void rleConnectedComponents(const RLE *R_in, int connectivity, siz min_size, RLE **components, siz *n_components_out) {
    bool diagonal = (connectivity == 8);
    if (R_in->m <= 1) {
        *n_components_out = 0;
        *components = NULL;
        return;
    }
    RLE *R_split = _rleSplitRunsThatMayBelongToDifferentComponents(R_in, connectivity);
    const RLE *R = R_split ? R_split : R_in;

    siz m = R->m;
    siz h = R->h;
    siz w = R->w;
    uint *cnts = R->cnts;

    // initially each run of 1s is a separate component
    struct UnionFindNode *uf = calloc(m / 2, sizeof(struct UnionFindNode));
    for (siz i = 1; i < m; i += 2) {
        uf[i / 2].size = cnts[i];  // initialize the size of the component to the size of the run of 1s
    }

    siz i1 = 1, i2 = 1, r1 = cnts[0], r2 = cnts[0];
    while (i1 < m && i2 < m) {
        // If the runs overlap vertically and are in neighboring columns, we need to update the labels
        siz overlap_start = uintMax(r1 + h, r2);
        siz overlap_end = uintMin(r1 + cnts[i1] + h, r2 + cnts[i2]);
        if (overlap_start < overlap_end || (diagonal && overlap_start == overlap_end && overlap_start % h != 0)) {
            _uf_union(&uf[i1 / 2], &uf[i2 / 2]);
        }

        // Step to the next run. Advance either the first or the second run, depending on which one ends first
        // Taking into account that we need a lag of h between the runs to be in neighboring columns.
        if (r1 + cnts[i1] + h < r2 + cnts[i2]) {
            if (i1 + 1 >= m) {
                break;
            }
            r1 += cnts[i1] + cnts[i1 + 1];
            i1 += 2;
        } else {
            if (i2 + 1 >= m) {
                break;
            }
            r2 += cnts[i2] + cnts[i2 + 1];
            i2 += 2;
        }
    }

    // Now, create new components based on the labels
    // we can potentially have as many components as runs of 1s, but we will likely have fewer
    // because some runs of 1s may be connected. Hence, we first count the number of unique labels
    // This will be the rle.m of the new components
    uint *new_label_to_component_m = malloc(sizeof(uint) * (m / 2));
    // This is the mapping from the old label index to the new compacted label index
    uint *label_to_new_label = malloc(sizeof(uint) * (m / 2));
    // fill with m to indicate that the label has not been used yet
    for (siz i = 0; i < m / 2; i++) {
        label_to_new_label[i] = m; // m indicates that the label has not been used yet
    }

    // This is the number of labels we have found so far
    siz n_components = 0;
    uint *new_labels = malloc(sizeof(uint) * (m / 2));

    for (siz i = 1; i < m; i += 2) {
        struct UnionFindNode *root = _uf_find(&uf[i / 2]);
        if (root->size < min_size) {
            new_labels[i / 2] = m; // this run is too small to be kept
            continue;
        }
        uint label = (uint) (root - uf);
        uint new_label;
        if (label_to_new_label[label] == m) {
            new_label = n_components++;
            label_to_new_label[label] = new_label;
            new_label_to_component_m[new_label] = 1;
        } else {
            new_label = label_to_new_label[label];
        }
        new_labels[i / 2] = new_label;
        // A new pair of runs is not added if the previous label was the same and the run of 0s was empty
        // This happens if a run of 1s got split earlier but then turned out to belong to the same component.
        if (i == 1 || new_label != new_labels[i / 2 - 1] || cnts[i - 1] > 0) {
            new_label_to_component_m[new_label] += 2;
        }
    }
    *n_components_out = n_components;
    if (n_components == 0) {
        *components = NULL;
        return;
    }

    // Now that we know how many runs each component will have, we can initialize the components
    rlesInit(components, n_components);
    RLE *rles_out = *components;
    for (siz i = 0; i < n_components; i++) {
        rleInit(&rles_out[i], h, w, new_label_to_component_m[i]);
        rles_out[i].cnts[0] = cnts[0]; // the first run of zeroes will be part of all components
    }

    // Now we can fill in the components. The next pointer will point to the next run of each component.
    uint **new_label_to_cnts = malloc(sizeof(uint *) * n_components);
    for (siz i = 0; i < n_components; i++) {
        new_label_to_cnts[i] = rles_out[i].cnts + 1;
    }

    // Now we scan over the original RLE runs and depending on its label, we update the components.
    // The component for that label will be updated with the run of 1s and the following run of 0s.
    // All other components will get their last run of 0s extended.
    siz r = cnts[0];
    for (siz i = 1; i < m; i += 2) {
        uint current_count_1s = cnts[i];
        uint next_count_0s = (i + 1 < R->m) ? cnts[i + 1] : 0;
        uint new_label = new_labels[i / 2];

        if (new_label == m) {
            // This run is too small to be kept
        } else if (
                new_label_to_cnts[new_label] - rles_out[new_label].cnts > 1 &&
                new_label_to_cnts[new_label][-1] == 0) {
            // The new run belongs to the same label as the previous and the run of 0s in between has size 0,
            // so we just extend the previous run of 1s and 0s.
            new_label_to_cnts[new_label][-2] += current_count_1s;
            new_label_to_cnts[new_label][-1] += next_count_0s;
        } else {
            // We must add a new pair of runs
            new_label_to_cnts[new_label][0] = current_count_1s;
            new_label_to_cnts[new_label][1] = next_count_0s;
            new_label_to_cnts[new_label] += 2;
        }

        for (siz j = 0; j < n_components; j++) {
            if (j != new_label) {
                // We extend the last run of 0s of all other components
                new_label_to_cnts[j][-1] += current_count_1s + next_count_0s;
            }
        }
        r += current_count_1s + next_count_0s;
    }

    // Check if the last run is empty, and if so, remove it
    uint last_label = new_labels[m / 2 - 1];
    if (last_label != m) {
        RLE *last_component = &rles_out[last_label];
        if (last_component->cnts[last_component->m - 1] == 0) {
            rleRealloc(last_component, last_component->m - 1);
        }
    }

    // Clean up
    free(new_label_to_component_m);
    free(label_to_new_label);
    free(new_label_to_cnts);
    free(new_labels);
    if (R_split) {
        rleFree(R_split);
        free(R_split);
    }
    free(uf);
}

void rleRemoveSmallConnectedComponentsInplace(RLE *R_in, siz min_size, int connectivity) {
    bool diagonal = (connectivity == 8);
    if (R_in->m <= 1) {
        return;
    }
    RLE *R_split = _rleSplitRunsThatMayBelongToDifferentComponents(R_in, connectivity);
    const RLE *R = R_split ? R_split : R_in;

    siz m = R->m;
    siz h = R->h;
    siz w = R->w;
    uint *cnts = R->cnts;

    // initially each run of 1s is a separate component
    struct UnionFindNode *uf = calloc(m / 2, sizeof(struct UnionFindNode));
    for (siz i = 1; i < m; i += 2) {
        uf[i / 2].size = cnts[i];  // initialize the size of the component to the size of the run of 1s
    }

    siz i1 = 1, i2 = 1, r1 = cnts[0], r2 = cnts[0];
    while (i1 < m && i2 < m) {
        // If the runs overlap vertically and are in neighboring columns, we need to update the labels
        siz overlap_start = uintMax(r1 + h, r2);
        siz overlap_end = uintMin(r1 + cnts[i1] + h, r2 + cnts[i2]);
        if (overlap_start < overlap_end || (diagonal && overlap_start == overlap_end && overlap_start % h != 0)) {
            _uf_union(&uf[i1 / 2], &uf[i2 / 2]);
        }

        // Step to the next run. Advance either the first or the second run, depending on which one ends first
        // Taking into account that we need a lag of h between the runs to be in neighboring columns.
        if (r1 + cnts[i1] + h < r2 + cnts[i2]) {
            if (i1 + 1 >= m) {
                break;
            }
            r1 += cnts[i1] + cnts[i1 + 1];
            i1 += 2;
        } else {
            if (i2 + 1 >= m) {
                break;
            }
            r2 += cnts[i2] + cnts[i2 + 1];
            i2 += 2;
        }
    }

    siz i_out = 1;
    for (siz i = 1; i < m; i += 2) {
        struct UnionFindNode *root = _uf_find(&uf[i / 2]);

        uint current_count_1s = cnts[i];
        uint next_count_0s = (i + 1 < R->m) ? cnts[i + 1] : 0;

        if (root->size < min_size) {
            // skip this one, the previous 0s are extended
            R_in->cnts[i_out - 1] += current_count_1s + next_count_0s;
        } else if (i_out > 1 && R_in->cnts[i_out - 1] == 0) {
            // The run of 0s in between has size 0 and there exists a prev run of 1s
            // so we just extend the previous run of 1s and 0s.
            R_in->cnts[i_out - 2] += current_count_1s;
            R_in->cnts[i_out - 1] += next_count_0s;
        } else {
            // We must add a new pair of runs
            R_in->cnts[i_out++] = current_count_1s;
            if (i + 1 < R->m) {
                R_in->cnts[i_out++] = next_count_0s;
            }
        }
    }
    R_in->m = i_out;
    rleEliminateZeroRuns(R_in);
    // Clean up
    if (R_split) {
        rleFree(R_split);
        free(R_split);
    }
    free(uf);
}

void rleLargestConnectedComponentInplace(RLE *R_in, int connectivity) {
    bool diagonal = (connectivity == 8);
    if (R_in->m <= 2) {
        return;
    }
    RLE *R_split = _rleSplitRunsThatMayBelongToDifferentComponents(R_in, connectivity);
    const RLE *R = R_split ? R_split : R_in;

    siz m = R->m;
    siz h = R->h;
    siz w = R->w;
    uint *cnts = R->cnts;

    // initially each run of 1s is a separate component
    struct UnionFindNode *uf = calloc(m / 2, sizeof(struct UnionFindNode));
    for (siz i = 1; i < m; i += 2) {
        uf[i / 2].size = cnts[i];  // initialize the size of the component to the size of the run of 1s
    }

    siz i1 = 1, i2 = 1, r1 = cnts[0], r2 = cnts[0];
    while (i1 < m && i2 < m) {
        // If the runs overlap vertically and are in neighboring columns, we need to update the labels
        siz overlap_start = uintMax(r1 + h, r2);
        siz overlap_end = uintMin(r1 + cnts[i1] + h, r2 + cnts[i2]);
        if (overlap_start < overlap_end || (diagonal && overlap_start == overlap_end && overlap_start % h != 0)) {
            _uf_union(&uf[i1 / 2], &uf[i2 / 2]);
        }

        // Step to the next run. Advance either the first or the second run, depending on which one ends first
        // Taking into account that we need a lag of h between the runs to be in neighboring columns.
        if (r1 + cnts[i1] + h < r2 + cnts[i2]) {
            if (i1 + 1 >= m) {
                break;
            }
            r1 += cnts[i1] + cnts[i1 + 1];
            i1 += 2;
        } else {
            if (i2 + 1 >= m) {
                break;
            }
            r2 += cnts[i2] + cnts[i2 + 1];
            i2 += 2;
        }
    }

    siz max_size = 0;
    struct UnionFindNode *max_node = NULL;

    // find the root node with the largest size
    for (struct UnionFindNode* n = uf; n < uf + m / 2; n++) {
        if (n->parent == NULL && n->size > max_size) {
            max_size = n->size;
            max_node = n;
        }
    }

    siz i_out = 1;
    for (siz i = 1; i < m; i += 2) {
        struct UnionFindNode *root = _uf_find(&uf[i / 2]);

        uint current_count_1s = cnts[i];
        uint next_count_0s = (i + 1 < R->m) ? cnts[i + 1] : 0;

        if (root != max_node) {
            // skip this one, the previous 0s are extended
            R_in->cnts[i_out - 1] += current_count_1s + next_count_0s;
        } else if (i_out > 1 && R_in->cnts[i_out - 1] == 0) {
            // The run of 0s in between has size 0 and there exists a prev run of 1s
            // so we just extend the previous run of 1s and 0s.
            R_in->cnts[i_out - 2] += current_count_1s;
            R_in->cnts[i_out - 1] += next_count_0s;
        } else {
            // We must add a new pair of runs
            R_in->cnts[i_out++] = current_count_1s;
            if (i + 1 < R->m) {
                R_in->cnts[i_out++] = next_count_0s;
            }
        }
    }
    R_in->m = i_out;
    rleEliminateZeroRuns(R_in);
    // Clean up
    if (R_split) {
        rleFree(R_split);
        free(R_split);
    }
    free(uf);
}


static RLE *_rleSplitRunsThatMayBelongToDifferentComponents(const RLE *R, int connectivity) {
    // A run of 1s may belong to different connected components if it spans more than one column
    // and its length is less than the height of the image. If it's exactly the height of the image it does not
    // split if connectivity is 8, but it might if connectivity is 4. If it has larger length, it will never split.

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

    RLE *M = malloc(sizeof(RLE));  // This memory must be freed by the caller
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