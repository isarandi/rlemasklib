/**************************************************************************
* Based on code from the Microsoft COCO Toolbox.      version 2.0
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015
* Modifications by Istvan Sarandi, 2023
* Licensed under the Simplified BSD License [see license.txt]
**************************************************************************/
#include "rlemasklib.h"
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>

uint umin(uint a, uint b) {
    return (a < b) ? a : b;
}

uint umax(uint a, uint b) {
    return (a > b) ? a : b;
}

siz smin(siz a, siz b) {
    return (a < b) ? a : b;
}

siz smax(siz a, siz b) {
    return (a > b) ? a : b;
}

void rleInit(RLE *R, siz h, siz w, siz m, uint *cnts, bool transfer_ownership) {
    R->h = h;
    R->w = w;
    R->m = m;

    if (m == 0) {
        R->cnts = NULL;
        if (transfer_ownership) {
            free(cnts);
        }
    } else if (transfer_ownership) {
        R->cnts = realloc(cnts, sizeof(uint) * m);
    } else {
        R->cnts = malloc(sizeof(uint) * m);
        if (cnts) {
            memcpy(R->cnts, cnts, sizeof(uint) * m);
        }
    }
}

void rleFree(RLE *R) {
    free(R->cnts);
    R->cnts = NULL;
}

void rlesInit(RLE **R, siz n) {
    *R = calloc(n, sizeof(RLE));
}

void rlesFree(RLE **R, siz n) {
    for (siz i = 0; i < n; i++) {
        rleFree((*R) + i);
    }
    free(*R);
    *R = NULL;
}

void rleEncode(RLE *R, const byte *M, siz h, siz w, siz n) {
    siz a = w * h;

    for (siz i = 0; i < n; i++) {
        const byte *T = M + a * i;
        uint *cnts = malloc(sizeof(uint) * (a + 1));

        siz k = 0;
        byte prev = 0;
        uint c = 0;

        for (siz j = 0; j < a; j++) {
            byte current = T[j];
            if (current != prev) {
                cnts[k++] = c;
                c = 0;
                prev = current;
            }
            c++;
        }
        cnts[k++] = c;
        rleInit(R + i, h, w, k, cnts, true);
    }
}


void rleDecode(const RLE *R, byte *M, siz n) {
    for (siz i = 0; i < n; i++) {
        byte v = 0;
        for (siz j = 0; j < R[i].m; j++) {
            uint cnt = R[i].cnts[j];
            memset(M, v, cnt);
            M += cnt;
            v = !v;
        }
    }
}

void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc) {
    uint c, ca, cb, cc, ct;
    bool v, va, vb, v_prev;
    siz h = R[0].h, w = R[0].w;

    if (n == 0) {
        rleInit(M, 0, 0, 0, NULL, false);
        return;
    }
    if (n == 1) {
        rleInit(M, h, w, R[0].m, R[0].cnts, false);
        return;
    }
    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    siz m_total = 0;
    for (siz i = 0; i < n; i++) {
        m_total += R[i].m;
    }
    uint *cnts = malloc(sizeof(uint) * smin(h * w + 1, m_total)); // cnts is the output

    RLE *A = NULL;
    siz m;

    for (siz i = 1; i < n; i++) {
        const RLE *B = &R[i];
        if (B->h != h || B->w != w) {
            h = w = m = 0;
            break;
        }

        uint *A_cnts;
        siz A_m;

        if (i == 1) {
            A_m = R[0].m;
            A_cnts = R[0].cnts;
        } else {
            if (!A) {
                A = malloc(sizeof(RLE));
            }
            rleInit(A, h, w, m, cnts, false); // A is the merged RLE so far
            A_m = m;
            A_cnts = A->cnts;
        }

        ca = A_cnts[0];
        cb = B->cnts[0];
        v = va = vb = false;
        m = 0;
        siz a = 1;
        siz b = 1;
        cc = 0;
        do {
            c = umin(ca, cb);
            cc += c; // add the current consumed amount to the output run

            // consume from the current run of A
            ca -= c;
            if (!ca && a < A_m) { // consumed a whole run from A and there are more
                ca = A_cnts[a++]; // get next run from A
                va = !va; // toggle the value of A
            }
            ct = ca; // ct is used to check if there are more runs to consume in either A or B

            // consume from the current run of B
            cb -= c;
            if (!cb && b < B->m) {
                cb = B->cnts[b++];
                vb = !vb;
            }
            ct += cb;

            v_prev = v;
            v = applyBoolFunc(va, vb, boolfunc);

            if (v != v_prev || ct == 0) {
                // if the value changed or we consumed all runs, we need to save the current run to the output
                cnts[m++] = cc;
                cc = 0;
            }
        } while (ct > 0); // continue until we consumed all runs from both A and B

        if (i > 1) {
            rleFree(A); // free the memory of previous intermediate result
        }
    }
    rleInit(M, h, w, m, cnts, true);
    free(A);
}

bool applyBoolFunc(bool x, bool y, uint boolfunc) {
    // boolfunc contains in its lowest 4 bits the truth table of the boolean function
    // (x << 1 | y) is the row index of the truth table (same as x*2 + y)
    // the value of the boolean function is the bit at that index, so we shift the selected bit to the lowest bit
    // and mask it with 1 to get this last bit.
    return (boolfunc >> ((int) x << 1 | (int) y)) & 1;
}

void rleArea(const RLE *R, siz n, uint *a) {
    for (siz i = 0; i < n; i++) {
        a[i] = 0;
        for (siz j = 1; j < R[i].m; j += 2) {
            a[i] += R[i].cnts[j];
        }
    }
}

void rleComplement(const RLE *R, RLE *M, siz n) {
    for (siz i = 0; i < n; i++) {
        siz h = R[i].h;
        siz w = R[i].w;
        if (R[i].m == 0 || h == 0 || w == 0) {
            rleInit(&M[i], h, w, 0, NULL, false);
        } else if (R[i].m > 0 && R[i].cnts[0] == 0) {
            // if the first run has size 0, we can just remove it
            rleInit(&M[i], h, w, R[i].m - 1, R[i].cnts + 1, false);
        } else {
            // if the first run has size > 0, we need to add a new run of 0s at the beginning
            rleInit(&M[i], h, w, R[i].m + 1, NULL, false);
            M[i].cnts[0] = 0;
            memcpy(M[i].cnts + 1, R[i].cnts, sizeof(uint) * R[i].m);
        }
    }
}

void rleComplementInplace(RLE *R, siz n) {
    for (siz i = 0; i < n; i++) {
        siz h = R[i].h;
        siz w = R[i].w;
        if (R[i].m == 0 || h == 0 || w == 0) {
            continue;
        } else if (R[i].m > 0 && R[i].cnts[0] == 0) {
            // if the first run has size 0, we can just remove it
            R[i].m -= 1;
            memmove(R[i].cnts, R[i].cnts + 1, sizeof(uint) * R[i].m);
            rleShrink(&R[i]);
        } else {
            // if the first run has size > 0, we need to add a new run of 0s at the beginning
            R[i].m += 1;
            R[i].cnts = realloc(R[i].cnts, sizeof(uint) * R[i].m);
            memmove(R[i].cnts + 1, R[i].cnts, sizeof(uint) * (R[i].m - 1));
            R[i].cnts[0] = 0;
        }
    }
}

void rleCrop(const RLE *R, RLE *M, siz n, const uint *bbox) {
    /* Crop RLEs to a specified bounding box.*/
    for (siz i = 0; i < n; i++) {
        uint h = R[i].h;
        uint w = R[i].w;
        siz m = R[i].m;

        // Clip bbox to image boundary.
        uint box_x_start = umin(bbox[i * 4 + 0], w);
        uint box_y_start = umin(bbox[i * 4 + 1], h);
        uint box_w = umin(bbox[i * 4 + 2], w - box_x_start);
        uint box_h = umin(bbox[i * 4 + 3], h - box_y_start);

        if (m == 0 || box_w == 0 || box_h == 0) {
            // RLE is empty, so just fill in zeros.
            rleInit(&M[i], box_h, box_w, 0, NULL, false);
            continue;
        }
        rleInit(&M[i], box_h, box_w, R[i].m, R[i].cnts, false);

        M[i].h = box_h;
        M[i].w = box_w;
        uint *cnts = M[i].cnts;
        uint box_start = box_x_start * h + box_y_start;
        uint box_end = (box_x_start + box_w - 1) * h + box_h + box_y_start;

        uint start = 0;
        for (siz j = 0; j < m; ++j) {
            uint end = start + cnts[j];
            if (end <= box_start || start >= box_end) {
                // The run is fully before or after the box, remove it.
                cnts[j] = 0;
            } else {
                // The run intersects the box, so we need to adjust it to the size of intersection.
                uint rel_start = umax(start, box_start) - box_start;
                uint rel_end = umin(end, box_end) - box_start;
                cnts[j] = umin(rel_end % h, box_h) - umin(rel_start % h, box_h) + box_h * (rel_end / h);
            }
            start = end;
        }
    }
    // Remove non-initial empty runs, in order to make the RLE encoding valid.
    //rleEliminateZeroRuns(M, n);
}

void rleCropInplace(RLE *R, siz n, const uint *bbox) {
    /* Crop RLEs to a specified bounding box in place.*/
    for (siz i = 0; i < n; i++) {
        uint h = R[i].h;
        uint w = R[i].w;
        siz m = R[i].m;

        // Clip bbox to image boundary.
        uint box_x_start = umin(bbox[i * 4 + 0], w);
        uint box_y_start = umin(bbox[i * 4 + 1], h);
        uint box_w = umin(bbox[i * 4 + 2], w - box_x_start);
        uint box_h = umin(bbox[i * 4 + 3], h - box_y_start);

        R[i].h = box_h;
        R[i].w = box_w;
        if (m == 0 || box_w == 0 || box_h == 0) {
            rleFree(&R[i]);
            R[i].m = 0;
            continue;
        }

        uint *cnts = R[i].cnts;
        uint box_start = box_x_start * h + box_y_start;
        uint box_end = (box_x_start + box_w - 1) * h + box_h + box_y_start;

        uint start = 0;
        for (siz j = 0; j < m; ++j) {
            uint end = start + cnts[j];
            if (end <= box_start || start >= box_end) {
                // The run is fully before or after the box, remove it.
                cnts[j] = 0;
            } else {
                // The run intersects the box, so we need to adjust it to the size of intersection.
                uint rel_start = umax(start, box_start) - box_start;
                uint rel_end = umin(end, box_end) - box_start;
                cnts[j] = (rel_end / h - rel_start / h) * box_h + umin(rel_end % h, box_h) - umin(rel_start % h, box_h);
            }
            start = end;
        }
    }
    // Remove non-initial empty runs, in order to make the RLE encoding valid.
    rleEliminateZeroRuns(R, n);
}


void rlePad(const RLE *R, RLE *M, siz n, const uint *pad_amounts) {
    // pad_amounts is four values: left, right, top, bottom
    for (siz i = 0; i < n; i++) {
        uint h = R[i].h;
        uint w = R[i].w;
        siz m = R[i].m;

        uint h_out = h + pad_amounts[2] + pad_amounts[3];
        uint w_out = w + pad_amounts[0] + pad_amounts[1];

        if (w_out == 0 || h_out == 0) {
            // RLE is empty, so just fill in zeros.
            rleInit(&M[i], h_out, w_out, 0, NULL, false);
            continue;
        }

        if (m == 0 || w == 0 || h == 0) {
            // RLE is empty, so just fill in zeros.
            rleInit(&M[i], h_out, w_out, 1, NULL, false);
            M[i].cnts[0] = h_out * w_out;
            continue;
        }

        uint start_px_addition = pad_amounts[0] * h_out + pad_amounts[2];
        uint end_px_addition = pad_amounts[1] * h_out + pad_amounts[3];

        if (pad_amounts[2] + pad_amounts[3] == 0) {
            // If there's no vertical padding, we can just copy the RLE and add the horizontal padding.
            if (m % 2 == 0 && end_px_addition > 0) {
                // if num of runs is even, it ends with a run of 1s. If we add padding at the end
                // we need to add a new run.
                rleInit(&M[i], h_out, w_out, m + 1, NULL, false);
                memcpy(M[i].cnts, R[i].cnts, sizeof(uint) * m);
                M[i].cnts[0] += start_px_addition;
                M[i].cnts[m] = end_px_addition;
            } else {
                rleInit(&M[i], h_out, w_out, m, R[i].cnts, false);
                M[i].cnts[0] += +start_px_addition;
                M[i].cnts[m - 1] += end_px_addition;
            }
            continue;
        }

        // We don't know how many runs will be added, as some runs of 1s may be split into multiple, when a padding is
        // added and the run was spanning multiple columns.
        // Therefore, we first do a pass just to calculate the number of runs that will be in the result.
        siz m_out = m;
        uint y = 0;
        for (siz j = 0; j < m; ++j) {
            // if this is a run of 1s, it may need to be split. Runs of 0s will just be expanded, never split
            if (j % 2 == 1) {
                // how many columns this run spans:
                // ie the last pixel of this run is how many columns to the right of the first pixel of this run
                m_out += ((y + R[i].cnts[j] - 1) / h) * 2;
            }
            y = (y + R[i].cnts[j]) % h;
        }

        // if num of runs is even, it ends with a run of 1s. If we add padding at the end, we need to add a new run.
        if (m % 2 == 0 && end_px_addition > 0) {
            ++m_out;
        }

        rleInit(&M[i], h_out, w_out, m_out, NULL, false);
        uint *cnts = M[i].cnts;
        y = 0;
        siz j_out = 0;
        bool carry_over = false;
        uint pad_vertical = pad_amounts[2] + pad_amounts[3];
        for (siz j = 0; j < m; ++j) {
            uint n_cols_spanned = (y + R[i].cnts[j] - 1) / h;
            uint n_cols_completed = (y + R[i].cnts[j]) / h;
            if (j % 2 == 0) {
                // run of 0s, make it longer
                cnts[j_out++] = R[i].cnts[j] + n_cols_completed * pad_vertical + (carry_over ? pad_vertical : 0);
            } else {
                // run of 1s
                if (n_cols_spanned > 0) {
                    cnts[j_out++] = h - y; // 1s
                    for (siz k = 0; k < n_cols_spanned - 1; ++k) {
                        cnts[j_out++] = pad_vertical; // 0s
                        cnts[j_out++] = h; // 1s
                    }
                    cnts[j_out++] = pad_vertical; // 0s
                    cnts[j_out++] = y + R[i].cnts[j] - n_cols_spanned * h; // 1s
                } else {
                    cnts[j_out++] = R[i].cnts[j];
                }
                carry_over = n_cols_completed > n_cols_spanned;
            }
            y = (y + R[i].cnts[j]) % h;
        }

        cnts[0] += start_px_addition;

        if (m % 2 == 0) {
            // original ends with 1s run
            if (end_px_addition > 0) {
                // new run of 0s is the last one
                cnts[m_out - 1] = end_px_addition;
            }
            // else, the last run of 1s is already good
        } else {
            // original ends with 0s run, so we extend it
            // it was already extended by pad_vertical in the for loop above
            cnts[m_out - 1] += end_px_addition - pad_vertical;
        }

    }
}


void rleConnectedComponents(const RLE *R_in, int connectivity, siz min_size, RLE **components, siz *n_components_out) {
    bool diagonal = (connectivity == 8);
    if (R_in->m <= 1) {
        *n_components_out = 0;
        *components = NULL;
        return;
    }
    RLE *R_split = rleSplitRunsThatMayBelongToDifferentComponents(R_in, connectivity);
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
        siz overlap_start = umax(r1 + h, r2);
        siz overlap_end = umin(r1 + cnts[i1] + h, r2 + cnts[i2]);
        if (overlap_start < overlap_end || (diagonal && overlap_start == overlap_end && overlap_start % h != 0)) {
            uf_union(&uf[i1 / 2], &uf[i2 / 2]);
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
    // because some runs of 1s may be connected. Hence, we first count the number of unique labels that remained
    // after the forward and backward passes.
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
        struct UnionFindNode *root = uf_find(&uf[i / 2]);
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
        rleInit(&rles_out[i], h, w, new_label_to_component_m[i], NULL, false);
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
            last_component->m -= 1;
            rleShrink(last_component);
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

void rleRotate180Inplace(RLE *R, siz n) {
    for (siz i = 0; i < n; i++) {
        siz h = R[i].h;
        siz w = R[i].w;
        siz m = R[i].m;
        if (m <= 1 || h == 0 || w == 0) {
            continue;
        }

        if (m % 2 == 0 && R[i].cnts[0] > 0) {
            // if the number of runs is even, the last run is a run of 1s, so we need to add a new run of 0s
            R[i].m += 1;
            R[i].cnts = realloc(R[i].cnts, sizeof(uint) * R[i].m);
            R[i].cnts[R[i].m - 1] = 0;
        }

        uint *cnts = R[i].cnts;
        uint *p1 = cnts + (m % 2 == 0 && R[i].cnts[0] == 0 ? 1 : 0);
        uint *p2 = cnts + R[i].m - 1;
        while (p1 < p2) {
            uint tmp = *p1;
            *p1 = *p2;
            *p2 = tmp;
            p1++;
            p2--;
        }
    }
}
//
//void rleHorizontalFlipInplace(RLE *R, siz n) {
//    rleChunkupInplace(R, n);
//
//    for (siz i = 0; i < n; i++) {
//        siz h = R[i].h;
//        siz w = R[i].w;
//        siz m = R[i].m;
//        if (m == 0 || h == 0 || w == 0) {
//            continue;
//        }
//
//
//        uint *cnts_out = malloc(sizeof(uint) * m);
//        uint *cnts_in = R[i].cnts;
//        memcpy(cnts_out, cnts_in, sizeof(uint) * m);
//        siz j_out = 0;
//        siz col_end = h * w;
//        siz col_start = r - h;
//        siz j_col_end = m;
//        siz r = col_end;
//        siz j;
//
//        while() {
//            j = j_col_end;
//            while (r > col_start) { // find a run that starts exactly at the column start
//                j--;
//                r -= cnts_in[j];
//            }
//
//            siz j2 = j;
//            for (siz j2=j; j2<j_col_end; ++j2) {
//                uint cnt = cnts_in[j2];
//                if (!cnt) {
//                    break;
//                }
//                cnts_out[j_out++] = cnt;
//            }
//            j_col_end = j;
//            col_start -= h;
//
//        }
//
//
//
//
//    }
//}

void rleChunkupInplace(RLE *R, siz n) {
    // Helper for flipping functions. Splits runs so that they only need to be reordered for flipping.
    for (siz i = 0; i < n; i++) {
        siz h = R[i].h;
        siz w = R[i].w;
        siz m = R[i].m;
        if (m == 0 || h == 0 || w == 0) {
            continue;
        }

        siz m_out = 0;
        siz r = 0;
        for (siz j = 0; j < m; j++) {
            // each run that spans multiple columns gets split
            // how many bottom borders does it pass through?
            uint cnt = R[i].cnts[j];
            uint n_cols_spanned = (r % h + cnt - 1) / h;
            m_out += umin(n_cols_spanned, 2) * 2 + 1;
            r += cnt;
        }

        if (m_out == m) {
            // no need to split
            continue;
        }

        uint *cnts = malloc(sizeof(uint) * m_out);
        siz j_out = 0;
        r = 0;
        for (siz j = 0; j < m; j++) {
            uint cnt = R[i].cnts[j];
            uint y = r % h;
            cnts[j_out++] = umin(cnt, h - y); // until the bottom

            uint n_full_cols = (y + cnt) / h - 1;
            if (n_full_cols > 0) {
                cnts[j_out++] = 0; // divider
                cnts[j_out++] = n_full_cols * h; // the full columns
            }
            uint rest = (y + cnt) % h;
            if (rest > 0) {
                cnts[j_out++] = 0; // divider
                cnts[j_out++] = rest; // the rest
            }
            r += R[i].cnts[j];
        }
        free(R[i].cnts);
        R[i].cnts = cnts;
    }
}

void rleEliminateZeroRuns(RLE *R, siz n) {
    for (siz i = 0; i < n; ++i) {
        if (R[i].m == 0) {
            // Already empty.
            continue;
        }
        siz k = 0;
        siz j = 1;
        while (j < R[i].m) {
            // opposite parity
            if (R[i].cnts[j] > 0) {
                ++k;
                R[i].cnts[k] = R[i].cnts[j];
                ++j;
            } else {
                ++j;
                if (j < R[i].m) {
                    R[i].cnts[k] += R[i].cnts[j];
                    ++j;
                }
            }
        }
        R[i].m = k + 1;
        rleShrink(&R[i]);
    }
}

void rleShrink(RLE *R) {
    R->cnts = realloc(R->cnts, sizeof(uint) * R->m);
}

RLE *rleSplitRunsThatMayBelongToDifferentComponents(const RLE *R, int connectivity) {
    // A run of 1s may belong to different connected components if it spans more than one column
    // and its length is less than the height of the image. If it's exactly the height of the image it does not
    // split if connectivity is 8, but it might if connectivity is 4. If it has larger length, it will never split.
    siz j;
    bool diagonal = (connectivity == 8);
    uint h = R->h;
    uint w = R->w;
    siz m = R->m;
    siz r = 0;
    siz n_splits = 0;
    for (j = 0; j < m; ++j) {
        uint cnt = R->cnts[j];
        if (j % 2 == 1 && cnt <= h + (diagonal ? 1 : 0) && (r + cnt - 1) / h > r / h) {
            ++n_splits;
        }
        r += cnt;
    }

    if (n_splits == 0) {
        return NULL;
    }

    RLE *M = malloc(sizeof(RLE));  // This memory must be freed by the caller
    rleInit(M, h, w, m + 2 * n_splits, NULL, false);
    r = 0;
    siz i_out = 0;
    for (j = 0; j < m; ++j) {
        uint cnt = R->cnts[j];
        if (j % 2 == 1 && cnt <= h + (diagonal ? 1 : 0) && (r + cnt - 1) / h > r / h) {
            M->cnts[i_out] = h - r % h;
            M->cnts[i_out + 1] = 0;
            M->cnts[i_out + 2] = cnt - M->cnts[i_out];
            i_out += 3;
        } else {
            M->cnts[i_out] = cnt;
            ++i_out;
        }
        r += cnt;
    }
    return M;
}

void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o) {
    siz g, d;
    BB db, gb;
    int crowd;
    db = malloc(sizeof(double) * m * 4);
    rleToBbox(dt, db, m);
    gb = malloc(sizeof(double) * n * 4);
    rleToBbox(gt, gb, n);
    bbIou(db, gb, m, n, iscrowd, o);
    free(db);
    free(gb);
    for (g = 0; g < n; g++) {
        for (d = 0; d < m; d++) {
            if (o[g * m + d] > 0) {
                crowd = iscrowd != NULL && iscrowd[g];
                if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) {
                    o[g * m + d] = -1;
                    continue;
                }
                siz ka, kb, a, b;
                uint c, ca, cb, ct, i, u;
                int va, vb;
                ca = dt[d].cnts[0];
                ka = dt[d].m;
                va = vb = 0;
                cb = gt[g].cnts[0];
                kb = gt[g].m;
                a = b = 1;
                i = u = 0;
                ct = 1;
                while (ct > 0) {
                    c = umin(ca, cb);
                    if (va || vb) {
                        u += c;
                        if (va && vb) {
                            i += c;
                        }
                    }
                    ct = 0;
                    ca -= c;
                    if (!ca && a < ka) {
                        ca = dt[d].cnts[a++];
                        va = !va;
                    }
                    ct += ca;
                    cb -= c;
                    if (!cb && b < kb) {
                        cb = gt[g].cnts[b++];
                        vb = !vb;
                    }
                    ct += cb;
                }
                if (i == 0) {
                    u = 1;
                } else if (crowd) {
                    rleArea(dt + d, 1, &u);
                }
                o[g * m + d] = (double) i / (double) u;
            }
        }
    }
}

void rleNms(RLE *dt, siz n, uint *keep, double thr) {
    siz i, j;
    double u;
    for (i = 0; i < n; i++) {
        keep[i] = 1;
    }
    for (i = 0; i < n; i++) {
        if (keep[i]) {
            for (j = i + 1; j < n; j++) {
                if (keep[j]) {
                    rleIou(dt + i, dt + j, 1, 1, 0, &u);
                    if (u > thr) {
                        keep[j] = 0;
                    }
                }
            }
        }
    }
}

void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o) {
    double h, w, i, u, ga, da;
    siz g, d;
    int crowd;
    for (g = 0; g < n; g++) {
        BB G = gt + g * 4;
        ga = G[2] * G[3];
        crowd = iscrowd != NULL && iscrowd[g];
        for (d = 0; d < m; d++) {
            BB D = dt + d * 4;
            da = D[2] * D[3];
            o[g * m + d] = 0;
            w = fmin(D[2] + D[0], G[2] + G[0]) - fmax(D[0], G[0]);
            if (w <= 0) {
                continue;
            }
            h = fmin(D[3] + D[1], G[3] + G[1]) - fmax(D[1], G[1]);
            if (h <= 0) {
                continue;
            }
            i = w * h;
            u = crowd ? da : da + ga - i;
            o[g * m + d] = i / u;
        }
    }
}

void bbNms(BB dt, siz n, uint *keep, double thr) {
    siz i, j;
    double u;
    for (i = 0; i < n; i++) { keep[i] = 1; }
    for (i = 0; i < n; i++) {
        if (keep[i]) {
            for (j = i + 1; j < n; j++) {
                if (keep[j]) {
                    bbIou(dt + i * 4, dt + j * 4, 1, 1, 0, &u);
                    if (u > thr) { keep[j] = 0; }
                }
            }
        }
    }
}

void rleToBbox(const RLE *R, BB bb, siz n) {
    siz i;
    for (i = 0; i < n; i++) {
        uint h, w, x, y, xs, ys, xe, ye, xp, cc, t;
        siz j, m;
        h = (uint) R[i].h;
        w = (uint) R[i].w;
        m = R[i].m;
        m = ((siz) (m / 2)) * 2;
        xs = w;
        ys = h;
        xe = ye = 0;
        cc = 0;
        if (m == 0) {
            bb[4 * i + 0] = bb[4 * i + 1] = bb[4 * i + 2] = bb[4 * i + 3] = 0;
            continue;
        }
        for (j = 0; j < m; j++) {
            cc += R[i].cnts[j];
            t = cc - j % 2;
            y = t % h;
            x = (t - y) / h;
            if (j % 2 == 0) {
                xp = x;
            } else if (xp < x) {
                ys = 0;
                ye = h - 1;
            }
            xs = umin(xs, x);
            xe = umax(xe, x);
            ys = umin(ys, y);
            ye = umax(ye, y);
        }
        bb[4 * i + 0] = xs;
        bb[4 * i + 2] = xe - xs + 1;
        bb[4 * i + 1] = ys;
        bb[4 * i + 3] = ye - ys + 1;
    }
}

//void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n) {
//    for (siz i = 0; i < n; i++) {
//        double xs = bb[4 * i + 0], xe = xs + bb[4 * i + 2];
//        double ys = bb[4 * i + 1], ye = ys + bb[4 * i + 3];
//        double xy[8] = {xs, ys, xs, ye, xe, ye, xe, ys};
//        rleFrPoly(R + i, xy, 4, h, w);
//    }
//}
//
void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n) {
    for (siz i = 0; i < n; i++) {
        double xs_ = fmax(0, fmin(w, bb[4 * i + 0]));
        double ys_ = fmax(0, fmin(h, bb[4 * i + 1]));
        double xe_ = fmax(0, fmin(w, bb[4 * i + 0] + bb[4 * i + 2]));
        double ye_ = fmax(0, fmin(h, bb[4 * i + 1] + bb[4 * i + 3]));
        siz xs = round(xs_);
        siz ys = round(ys_);
        siz xe = round(xe_);
        siz ye = round(ye_);
        siz bw = (xs <= xe ? xe - xs : 0);
        siz bh = (ys <= ye ? ye - ys : 0);
        siz m;
        if (bw == 0 || bh == 0) {
            rleInit(R + i, h, w, 1, NULL, false);
            R[i].cnts[0] = h * w;
            continue;
        }
        if (bh == h) {
            // if the bounding box spans the entire height, it will have a single run of 1s.
            m = (xe == w ? 2 : 3);
            rleInit(R + i, h, w, m, NULL, false);
            R[i].cnts[0] = h * xs; // run of 0s
            // if it spans the entire width, it will have no final run of 0s.
            R[i].cnts[1] = h * bw; // run of 1s
        } else {
            // runs of 1 are the columns of the box (as many as the width of the box)
            // and each will have a preceding run of 0s. If the box does not end at the end of the image,
            // there will be a final run of 0s too.
            m = bw * 2 + (xe == w && ye == h ? 0 : 1);
            rleInit(R + i, h, w, m, NULL, false);
            R[i].cnts[0] = h * xs + ys; // run of 0s
            R[i].cnts[1] = bh; // run of 1s
            for (siz j = 1; j < bw; j++) {
                R[i].cnts[j * 2] = h - bh; // run of 0s
                R[i].cnts[j * 2 + 1] = bh; // run of 1s
            }
        }

        if (!(xe == w && ye == h)) {
            R[i].cnts[m - 1] = h * (w - xe) + (h - ye); // run of 0s
        }
    }
}

int uintCompare(const void *a, const void *b) {
    uint c = *((uint *) a), d = *((uint *) b);
    return c > d ? 1 : c < d ? -1 : 0;
}

void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w) {
    /* upsample and get discrete points densely along entire boundary */
    siz j, m = 0;
    double scale = 5;
    int *x, *y, *u, *v;
    uint *a, *b;
    x = malloc(sizeof(int) * (k + 1));
    y = malloc(sizeof(int) * (k + 1));
    for (j = 0; j < k; j++) {
        x[j] = (int) (scale * xy[j * 2 + 0] + .5);
    }
    x[k] = x[0];
    for (j = 0; j < k; j++) {
        y[j] = (int) (scale * xy[j * 2 + 1] + .5);
    }
    y[k] = y[0];
    for (j = 0; j < k; j++) {
        m += umax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
    }
    u = malloc(sizeof(int) * m);
    v = malloc(sizeof(int) * m);
    m = 0;
    for (j = 0; j < k; j++) {
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1], dx, dy, t, d;
        int flip;
        double s;
        dx = abs(xe - xs);
        dy = abs(ys - ye);
        flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {
            t = xs;
            xs = xe;
            xe = t;
            t = ys;
            ys = ye;
            ye = t;
        }
        s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        if (dx >= dy) {
            for (d = 0; d <= dx; d++) {
                t = flip ? dx - d : d;
                u[m] = t + xs;
                v[m] = (int) (ys + s * t + .5);
                m++;
            }
        } else {
            for (d = 0; d <= dy; d++) {
                t = flip ? dy - d : d;
                v[m] = t + ys;
                u[m] = (int) (xs + s * t + .5);
                m++;
            }
        }
    }
    /* get points along y-boundary and downsample */
    free(x);
    free(y);
    k = m;
    m = 0;
    double xd, yd;
    x = malloc(sizeof(int) * k);
    y = malloc(sizeof(int) * k);
    for (j = 1; j < k; j++) {
        if (u[j] != u[j - 1]) {
            xd = (double) (u[j] < u[j - 1] ? u[j] : u[j] - 1);
            xd = (xd + .5) / scale - .5;
            if (floor(xd) != xd || xd < 0 || xd > w - 1) { continue; }
            yd = (double) (v[j] < v[j - 1] ? v[j] : v[j - 1]);
            yd = (yd + .5) / scale - .5;
            if (yd < 0) {
                yd = 0;
            } else if (yd > h) {
                yd = h;
            }
            yd = ceil(yd);
            x[m] = (int) xd;
            y[m] = (int) yd;
            m++;
        }
    }
    /* compute rle encoding given y-boundary points */
    k = m;
    a = malloc(sizeof(uint) * (k + 1));
    for (j = 0; j < k; j++) {
        a[j] = (uint) (x[j] * (int) (h) + y[j]);
    }
    a[k++] = (uint) (h * w);
    free(u);
    free(v);
    free(x);
    free(y);
    qsort(a, k, sizeof(uint), uintCompare);
    uint p = 0;
    for (j = 0; j < k; j++) {
        uint t = a[j];
        a[j] -= p;
        p = t;
    }
    b = malloc(sizeof(uint) * k);
    j = m = 0;
    b[m++] = a[j++];
    while (j < k) {
        if (a[j] > 0) {
            b[m++] = a[j++];
        } else {
            j++;
            if (j < k) {
                b[m - 1] += a[j++];
            }
        }
    }
    rleInit(R, h, w, m, b, true);
    free(a);
}

void rleCentroid(const RLE *R, double *xys, siz n) {
    for (siz i = 0; i < n; i++) {
        siz m = R[i].m;
        uint h = R[i].h;
        uint pos = 0;
        uint area = 0;
        double x = 0, y = 0;

        for (siz j = 0; j < m; j++) {
            uint cnt = R[i].cnts[j];

            if (j % 2 == 1) { // run of 1s
                area += cnt;

                uint start_col = pos / h;
                uint cnt1 = umin(cnt, h - (pos % h));
                x += start_col * cnt1;
                y += ((pos % h) + (cnt1 - 1) * 0.5) * cnt1;

                // second part is multiple full columns
                uint num_full_cols = (cnt - cnt1) / h;
                uint cnt2 = num_full_cols * h;
                if (cnt2) {
                    x += (start_col + 1 + (num_full_cols - 1) * 0.5) * cnt2;
                    y += ((h - 1) * 0.5) * cnt2;
                }

                // third part is a partial column
                uint cnt3 = cnt - cnt1 - cnt2;
                if (cnt3) {
                    x += (start_col + num_full_cols + 1) * cnt3;
                    y += ((cnt3 - 1) * 0.5) * cnt3;
                }
            }
            pos += cnt;
        }

        xys[i * 2 + 0] = x / area;
        xys[i * 2 + 1] = y / area;
    }
}

char *rleToString(const RLE *R) {
    /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
    siz m = R->m;
    // optimistic allocation (realistic masks tend to take up less than 2x the space of the rle)
    // in a single byte we can store 5 bits of the run length (difference), allowing for a range of -16 to 15
    // in two bytes we can store 10 bits, allowing for a range of -512 to 511
    // experimentally, in a realistic human mask dataset (mixed sources) the factor for the 99th percentile is
    // still below 2.0, so we use that as a heuristic. The average is around 1.5.
    siz alloc_size = 2 * m * sizeof(char);
    char *s = malloc(alloc_size); // optimistic allocation
    siz p = 0;
    bool more;
    for (siz i = 0; i < m; i++) {
        long x = R->cnts[i];
        if (i > 2) {
            x -= R->cnts[i - 2];  // take the difference from the last run of the same value
        }

        do {
            char c = x & 0x1f;  // last 5 bits of the run length difference
            x >>= 5;
            // if all the remaining bits are the same as the highest bit of the current 5 bits,
            // then this is the last 5-bit chunk we need
            more = (c & 0x10) ? x != -1 : x != 0;
            if (more) {
                c |= 0x20; // set continuation bit at the 3rd position
            }

            if (p >= alloc_size) {
                alloc_size *= 2;
                s = realloc(s, alloc_size);
            }
            s[p++] = c + 48;  // ascii 48 is '0'. 48-111 is the range of ascii chars we use
        } while (more);
    }
    s = realloc(s, sizeof(char) * (p + 1));
    s[p] = 0; // null-terminate the string
    return s;
}

void rleFrString(RLE *R, char *s, siz h, siz w) {
    uint *cnts = malloc(sizeof(uint) * strlen(s));
    siz m = 0;
    for (siz p=0; s[p];) {
        long x = 0; // the run length (difference)
        siz k = 0; // the number of bytes (of which 5 bits and a continuation bit are used) in the run length
        char more;
        do {
            char c = s[p];
            if (!c) {
                return; // unexpected end of string, last char promised more to come, but lied. Malformed input!
            }

            c -= 48; // subtract the offset, so the range is 0-63
            x |= (long)(c & 0x1f) << k; // take the last 5 bits of the char and shift them to the right position
            more = c & 0x20; // check the continuation bit
            p++;
            k += 5;
            if (!more && (c & 0x10)) {
                x |= -1 << k; // if the highest bit of the last 5 bits is set, set all the remaining bits to 1
            }
        } while (more);

        if (m > 2) {
            x += cnts[m - 2]; // add the difference to the last run of the same value
        }

        cnts[m++] = x;
    }
    rleInit(R, h, w, m, cnts, true);

}

// Union-find data structure for tracking connected components
struct UnionFindNode *uf_find(struct UnionFindNode *x) {
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

void uf_union(struct UnionFindNode *x, struct UnionFindNode *y) {
    x = uf_find(x);
    y = uf_find(y);
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

