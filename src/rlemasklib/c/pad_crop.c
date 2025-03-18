#include <stdbool.h> // for bool
#include <stddef.h> // for NULL
#include <stdlib.h> // for malloc, free
#include <string.h> // for memcpy
#include "basics.h"
#include "minmax.h"
#include "pad_crop.h"

// Count the number of pixels in a run starting at height y with length cnt (for image height h)
// which are at the very bottom of the image. I.e. it may have the bottom pixel of column x, the
// bottom pixel of column x+1, etc. and we count how many it has.
static uint nBottomPixels(uint y, uint cnt, siz h);
// Same for counting how many top pixels a run has.
static uint nTopPixels(uint y, uint cnt, siz h);
// Like nTopPixels, but cnt is guaranteed to be nonzero.
static uint nTopPixelsNonzero(uint y, uint cnt, siz h);

//------------------------------------------------------------------------

void rleCrop(const RLE *R, RLE *M, siz n, const uint *bbox) {
    /* Crop RLEs to a specified bounding box.*/
    for (siz i = 0; i < n; i++) {
        uint h = R[i].h;
        uint w = R[i].w;
        siz m = R[i].m;

        // Clip bbox to image boundary.
        uint box_x_start = uintMin(bbox[i * 4 + 0], w);
        uint box_y_start = uintMin(bbox[i * 4 + 1], h);
        uint box_w = uintMin(bbox[i * 4 + 2], w - box_x_start);
        uint box_h = uintMin(bbox[i * 4 + 3], h - box_y_start);

        if (m == 0 || box_w == 0 || box_h == 0) {
            // RLE is empty, so just fill in zeros.
            rleInit(&M[i], box_h, box_w, 0);
            continue;
        }

        uint *cnts_in = R[i].cnts;
        uint *cnts_out = rleInit(&M[i], box_h, box_w, m);

        if (box_h == h) {
            // The box is full-height, so we only need to remove things from the sides.
            // find the run that has the topleft pixel of the box
            uint box_start = box_x_start * h;
            uint box_end = (box_x_start + box_w) * h;

            siz j_first;
            uint cnt_first;
            siz r = 0;
            for (siz j = 0; j < m; j++) {
                r += cnts_in[j];
                if (r > box_start) {
                    j_first = j;
                    cnt_first = r - box_start;
                    break;
                }
            }

            // find the run that has the bottomright pixel of the box
            siz j_last;
            uint cnt_last;
            r = h * w;
            for (siz j = m - 1; true; j--) {
                r -= cnts_in[j];
                if (r < box_end) {
                    j_last = j;
                    cnt_last = box_end - r;
                    break;
                }
            }

            siz m_out;
            if (j_first == j_last) {
                // The box is fully contained in a single run.
                if (j_first % 2 == 0) {
                    // The run is 0s
                    cnts_out[0] = box_h * box_w;
                    m_out = 1;
                } else {
                    // The run is 1s
                    cnts_out[0] = 0;
                    cnts_out[1] = box_h * box_w;
                    m_out = 2;
                }
            } else {
                siz num_runs = j_last - j_first + 1;
                if (j_first % 2 == 0) {
                    // The first run is 0s
                    cnts_out[0] = cnt_first;
                    memcpy(cnts_out + 1, cnts_in + j_first + 1, sizeof(uint) * (num_runs - 2));
                    cnts_out[num_runs - 1] = cnt_last;
                    m_out = num_runs;
                } else {
                    // The first run is 1s
                    cnts_out[0] = 0;
                    cnts_out[1] = cnt_first;
                    memcpy(cnts_out + 2, cnts_in + j_first + 1, sizeof(uint) * (num_runs - 2));
                    cnts_out[num_runs] = cnt_last;
                    m_out = num_runs + 1;
                }
            }
            rleRealloc(&M[i], m_out);
        } else {
            uint box_start = box_x_start * h + box_y_start;
            uint box_end = (box_x_start + box_w - 1) * h + box_h + box_y_start;

            uint start = 0;
            for (siz j = 0; j < m; j++) {
                uint end = start + cnts_in[j];
                if (end <= box_start || start >= box_end) {
                    // The run is fully before or after the box, remove it.
                    cnts_out[j] = 0;
                } else {
                    // The run intersects the box, so we need to adjust it to the size of intersection.
                    uint rel_start = uintMax(start, box_start) - box_start;
                    uint rel_end = uintMin(end, box_end) - box_start;
                    cnts_out[j] = (
                        uintMin(rel_end % h, box_h) + box_h * (rel_end / h)
                        - uintMin(rel_start % h, box_h) - box_h * (rel_start / h));
                }
                start = end;
            }
            // Remove non-initial empty runs, in order to make the RLE encoding valid.
            rleEliminateZeroRuns(&M[i]);
        }
    }
}

void rleCropInplace(RLE *R, siz n, const uint *bbox) {
    /* Crop RLEs to a specified bounding box in place.*/
    for (siz i = 0; i < n; i++) {
        uint h = R[i].h;
        uint w = R[i].w;
        siz m = R[i].m;

        // Clip bbox to image boundary.
        uint box_x_start = uintMin(bbox[i * 4 + 0], w);
        uint box_y_start = uintMin(bbox[i * 4 + 1], h);
        uint box_w = uintMin(bbox[i * 4 + 2], w - box_x_start);
        uint box_h = uintMin(bbox[i * 4 + 3], h - box_y_start);

        R[i].h = box_h;
        R[i].w = box_w;
        if (m == 0 || box_w == 0 || box_h == 0) {
            rleFree(&R[i]);
            R[i].m = 0;
            continue;
        }

        uint *cnts = R[i].cnts;

        if (box_h == h) {
            // The box is full-height, so we only need to remove things from the sides.
            // find the run that has the topleft pixel of the box
            uint box_start = box_x_start * h;
            uint box_end = (box_x_start + box_w) * h;

            siz j_first;
            uint cnt_first;
            siz r = 0;
            for (siz j = 0; j < m; j++) {
                r += cnts[j];
                if (r > box_start) {
                    j_first = j;
                    cnt_first = r - box_start;
                    break;
                }
            }

            // find the run that has the bottomright pixel of the box
            siz j_last;
            uint cnt_last;
            r = h * w;
            for (siz j = m - 1; true; j--) {
                r -= cnts[j];
                if (r < box_end) {
                    j_last = j;
                    cnt_last = box_end - r;
                    break;
                }
            }

            siz m_out;
            if (j_first == j_last) {
                // The box is fully contained in a single run.
                if (j_first % 2 == 0) {
                    // The run is 0s
                    cnts[0] = box_h * box_w;
                    m_out = 1;
                } else {
                    // The run is 1s
                    cnts[0] = 0;
                    cnts[1] = box_h * box_w;
                    m_out = 2;
                }
            } else {
                siz num_runs = j_last - j_first + 1;
                if (j_first % 2 == 0) {
                    // The first run is 0s
                    cnts[0] = cnt_first;
                    memmove(cnts + 1, cnts + j_first + 1, sizeof(uint) * (num_runs - 2));
                    cnts[num_runs - 1] = cnt_last;
                    m_out = num_runs;
                } else {
                    // The first run is 1s
                    cnts[0] = 0;
                    cnts[1] = cnt_first;
                    memmove(cnts + 2, cnts + j_first + 1, sizeof(uint) * (num_runs - 2));
                    cnts[num_runs] = cnt_last;
                    m_out = num_runs + 1;
                }
            }
            rleRealloc(&R[i], m_out);
        } else {
            uint box_start = box_x_start * h + box_y_start;
            uint box_end = (box_x_start + box_w - 1) * h + box_h + box_y_start;

            uint start = 0;
            for (siz j = 0; j < m; j++) {
                uint end = start + cnts[j];
                if (end <= box_start || start >= box_end) {
                    // The run is fully before or after the box, remove it.
                    cnts[j] = 0;
                } else {
                    // The run intersects the box, so we need to adjust it to the size of intersection.
                    uint rel_start = uintMax(start, box_start) - box_start;
                    uint rel_end = uintMin(end, box_end) - box_start;
                    cnts[j] = (rel_end / h - rel_start / h) * box_h + uintMin(rel_end % h, box_h) - uintMin(rel_start % h, box_h);
                }
                start = end;
            }
            // Remove non-initial empty runs, in order to make the RLE encoding valid.
            rleEliminateZeroRuns(&R[i]);
        }
    }
}

void rleZeroPad(const RLE *R, RLE *M, siz n, const uint *pad_amounts) {
    // pad_amounts is four values: left, right, top, bottom
    for (siz i = 0; i < n; i++) {
        uint h = R[i].h;
        uint w = R[i].w;
        siz m = R[i].m;

        uint h_out = h + pad_amounts[2] + pad_amounts[3];
        uint w_out = w + pad_amounts[0] + pad_amounts[1];

        if (w_out == 0 || h_out == 0) {
            // RLE is empty, so just fill in zeros.
            rleInit(&M[i], h_out, w_out, 0);
            continue;
        }

        if (m == 0 || w == 0 || h == 0) {
            // RLE is empty, so just fill in zeros.
            rleZeros(&M[i], h_out, w_out);
            continue;
        }

        uint start_px_addition = pad_amounts[0] * h_out + pad_amounts[2];
        uint end_px_addition = pad_amounts[1] * h_out + pad_amounts[3];

        if (pad_amounts[2] == 0 && pad_amounts[3] == 0) {
            // If there's no vertical padding, we can just copy the RLE and add the horizontal padding.
            if (m % 2 == 0 && end_px_addition > 0) {
                // if num of runs is even, it ends with a run of 1s. If we add padding at the end
                // we need to add a new run.
                rleInit(&M[i], h_out, w_out, m + 1);
                memcpy(M[i].cnts, R[i].cnts, sizeof(uint) * m);
                M[i].cnts[0] += start_px_addition;
                M[i].cnts[m] = end_px_addition;
            } else {
                rleFrCnts(&M[i], h_out, w_out, m, R[i].cnts);
                M[i].cnts[0] += start_px_addition;
                M[i].cnts[m - 1] += end_px_addition;
            }
            continue;
        }

        // We don't know how many runs will be added, as some runs of 1s may be split into multiple, when a padding is
        // added and the run was spanning multiple columns.
        // Therefore, we first do a pass just to calculate the number of runs that will be in the result.
        siz m_out = m;
        uint y = 0;
        for (siz j = 0; j < m; j++) {
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
            m_out++;
        }

        uint *cnts = rleInit(&M[i], h_out, w_out, m_out);
        y = 0;
        siz j_out = 0;
        bool carry_over = false;
        uint pad_vertical = pad_amounts[2] + pad_amounts[3];
        for (siz j = 0; j < m; j++) {
            uint cnt = R[i].cnts[j];
            if (y + cnt < h) {
                // this run is fully contained in one column
                if (j % 2 == 0 && carry_over) {
                    cnts[j_out++] = cnt + pad_vertical;
                } else {
                    cnts[j_out++] = cnt;
                }
                y = (y + cnt) % h;
                carry_over = false;
                continue;
            }
            
            uint n_cols_completed = (y + cnt) / h;
            uint n_cols_spanned = (y + cnt - 1) / h;

            if (j % 2 == 0) {
                // run of 0s, make it longer
                cnts[j_out++] = cnt + n_cols_completed * pad_vertical + (carry_over ? pad_vertical : 0);
            } else {
                // run of 1s
                if (n_cols_spanned > 0) {
                    cnts[j_out++] = h - y; // 1s
                    for (siz k = 0; k < n_cols_spanned - 1; k++) {
                        cnts[j_out++] = pad_vertical; // 0s
                        cnts[j_out++] = h; // 1s
                    }
                    cnts[j_out++] = pad_vertical; // 0s
                    cnts[j_out++] = y + cnt - n_cols_spanned * h; // 1s
                } else {
                    cnts[j_out++] = cnt;
                }
                carry_over = n_cols_completed > n_cols_spanned;
            }
            y = (y + cnt) % h;
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


void rleZeroPadInplace(RLE *R, siz n, const uint *pad_amounts) {
    // pad_amounts is four values: left, right, top, bottom
    for (siz i = 0; i < n; i++) {
        uint h = R[i].h;
        uint w = R[i].w;
        siz m = R[i].m;

        uint h_out = h + pad_amounts[2] + pad_amounts[3];
        uint w_out = w + pad_amounts[0] + pad_amounts[1];
        R[i].h = h_out;
        R[i].w = w_out;

        if (w_out == 0 || h_out == 0) {
            free(R[i].cnts);
            R[i].m = 0;
            R[i].cnts = NULL;
            continue;
        }

        if (m == 0 || w == 0 || h == 0) {
            // RLE is empty, so just fill in zeros.
            R[i].m = 1;
            R[i].cnts = realloc(R[i].cnts, sizeof(uint) * 1);
            R[i].cnts[0] = h_out * w_out;
            continue;
        }

        uint start_px_addition = pad_amounts[0] * h_out + pad_amounts[2];
        uint end_px_addition = pad_amounts[1] * h_out + pad_amounts[3];

        if (pad_amounts[2] == 0 && pad_amounts[3] == 0) {
            // If there's no vertical padding, we can just copy the RLE and add the horizontal padding.
            if (m % 2 == 0 && end_px_addition > 0) {
                // if num of runs is even, it ends with a run of 1s. If we add padding at the end
                // we need to add a new run.
                rleRealloc(&R[i], R[i].m + 1);
                R[i].cnts[0] += start_px_addition;
                R[i].cnts[m] = end_px_addition;
            } else {
                R[i].cnts[0] += start_px_addition;
                R[i].cnts[m - 1] += end_px_addition;
            }
            continue;
        }

        // We don't know how many runs will be added, as some runs of 1s may be split into multiple, when a padding is
        // added and the run was spanning multiple columns.
        // Therefore, we first do a pass just to calculate the number of runs that will be in the result.
        siz m_out = m;
        uint y = 0;
        for (siz j = 0; j < m; j++) {
            // if this is a run of 1s, it may need to be split. Runs of 0s will just be expanded, never split
            y += R[i].cnts[j];
            if (j % 2 == 1) {
                // how many columns this run spans:
                // ie the last pixel of this run is how many columns to the right of the first pixel of this run
                m_out += ((y - 1) / h) * 2;
            }
            y %= h;
        }

        // if num of runs is even, it ends with a run of 1s. If we add padding at the end, we need to add a new run.
        if (m % 2 == 0 && end_px_addition > 0) {
            m_out++;
        }

        RLE* tmp;
        uint *cnts;
        if (m_out == m) {
            // We can overwrite the existing RLE
            cnts = R[i].cnts;
            R[i].h = h_out;
            R[i].w = w_out;
            tmp = NULL;
        } else {
            // We need to allocate a new RLE
            tmp = malloc(sizeof(RLE));
            cnts = rleInit(tmp, h_out, w_out, m_out);
        }

        y = 0;
        siz j_out = 0;
        bool carry_over = false;
        uint pad_vertical = pad_amounts[2] + pad_amounts[3];
        for (siz j = 0; j < m; j++) {
            uint cnt = R[i].cnts[j];
            if (y + cnt < h) {
                if (j % 2 == 0 && carry_over) {
                    cnts[j_out++] = cnt + pad_vertical;
                } else {
                    cnts[j_out++] = cnt;
                }
                y = (y + cnt) % h;
                carry_over = false;
                continue;
            }
        
            uint n_cols_spanned = (y + cnt - 1) / h;
            uint n_cols_completed = (y + cnt) / h;
            if (j % 2 == 0) {
                // run of 0s, make it longer
                cnts[j_out++] = cnt + n_cols_completed * pad_vertical + (carry_over ? pad_vertical : 0);
            } else {
                // run of 1s
                if (n_cols_spanned > 0) {
                    cnts[j_out++] = h - y; // 1s
                    for (siz k = 0; k < n_cols_spanned - 1; ++k) {
                        cnts[j_out++] = pad_vertical; // 0s
                        cnts[j_out++] = h; // 1s
                    }
                    cnts[j_out++] = pad_vertical; // 0s
                    cnts[j_out++] = y + cnt - n_cols_spanned * h; // 1s
                } else {
                    cnts[j_out++] = cnt;
                }
                carry_over = n_cols_completed > n_cols_spanned;
            }
            y = (y + cnt) % h;
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

        if (tmp != NULL) {
            rleMoveTo(tmp, &R[i]);
            free(tmp);
        }
    }
}

static uint nBottomPixels(uint y, uint cnt, siz h) {
    return (y + cnt) / h;
}

static uint nTopPixels(uint y, uint cnt, siz h) {
    if (cnt == 0) {
        return 0;
    }
    return (y + cnt - 1) / h + (y == 0 ? 1 : 0);
}

static uint nTopPixelsNonzero(uint y, uint cnt, siz h) {
    return (y + cnt - 1) / h + (y == 0 ? 1 : 0);
}

void rlePadReplicate(const RLE *R, RLE *M, const uint *pad_amounts) {
    // pad_amounts is four values: left, right, top, bottom
    uint h = R->h;
    uint w = R->w;
    siz m = R->m;

    uint h_out = h + pad_amounts[2] + pad_amounts[3];
    uint w_out = w + pad_amounts[0] + pad_amounts[1];

    if (w_out == 0 || h_out == 0) {
        // RLE is empty, so just fill in zeros.
        rleInit(M, h_out, w_out, 0);
        return;
    }

    if (m == 0 || w == 0 || h == 0) {
        // RLE is empty, so just fill in zeros.
        rleZeros(M, h_out, w_out);
        return;
    }
    uint *cnts = R->cnts;

    uint plef = pad_amounts[0];
    uint prig = pad_amounts[1];
    uint ptop = pad_amounts[2];
    uint pbot = pad_amounts[3];

    // check if the entire first col has same color

    uint j_toplef = (cnts[0] == 0) ? 1 : 0;
    uint j_botlef;  // the run that has the bottomleft pixel
    uint cnt_botlef; // the number of bottomleft pixels in the run
    uint r_end_botlef; // the end of the run that has the bottomleft pixel
    siz r = 0;
    for (siz j = 0; j < m; j++) {
        uint r_prev = r;
        r += cnts[j];
        if (r >= h) {
            j_botlef = j;
            cnt_botlef = h - r_prev;
            r_end_botlef = r;
            break;
        }
    }


    uint j_toprig; // the run that has the topright pixel
    uint cnt_toprig; // the number of topright pixels in the run
    r = 0;
    for (siz j = m - 1; j >= 0; j--) {
        uint r_prev = r;
        r += cnts[j];
        if (r >= h) {
            j_toprig = j;
            cnt_toprig = h - r_prev;
            break;
        }
    }

    uint j_botrig = m-1;
    bool v_toplef = j_toplef % 2;
    bool v_botlef = j_botlef % 2;
    bool v_botrig = j_botrig % 2;
    bool v_toprig = j_toprig % 2;

    bool left_col_same = (j_botlef == 0 || (v_toplef && j_botlef == 1));
    bool right_col_same = j_toprig == j_botrig;


    uint cnt_toplef = (j_toplef == j_botlef) ? cnt_botlef : cnts[j_toplef];
    uint cnt_botrig = (j_botrig == j_toprig) ? cnt_toprig : cnts[j_botrig];

    // Compute how many runs we will have in the output
    int total_out = v_toplef ? 1 : 0;
    if (left_col_same) {
        total_out += 1;
    } else {
        total_out += (plef + 1) * (j_botlef - j_toplef)
                   + (v_toplef != v_botlef ? plef : 0)
                   + 1;
    }
    total_out += (j_toprig > j_botlef + 1) ? (j_toprig - (j_botlef + 1)) : 0;
    if (!right_col_same) {
        total_out += (prig + (w > 1 ? 1 : 0)) * (j_botrig - j_toprig)
                   + (v_botrig != v_toprig ? prig : 0)
                   + ((j_botlef != j_toprig && w > 1) ? 1 : 0);
    } else if (j_botlef != j_toprig) {
        total_out += 1;
    }

    uint *cnts_out = rleInit(M, h_out, w_out, total_out);

    siz m_out = 0;
    if (v_toplef) {
        cnts_out[m_out++] = 0;
    }
    if (left_col_same) {
        uint cnt = cnts[j_toplef];
        uint ntop = nTopPixels(0, cnt, h);
        uint nbot = nBottomPixels(0, cnt, h);
        cnts_out[m_out++] = plef * h_out + ptop * ntop + pbot * nbot + cnt;
    } else {
        for (siz x = 0; x < plef + 1; x++) {
            bool v_out = m_out % 2 != 0;
            if (v_out != v_toplef) {
                cnts_out[m_out - 1] += cnt_toplef + ptop;
            } else {
                cnts_out[m_out++] = cnt_toplef + ptop;
            }

            for (siz j = j_toplef + 1; j < j_botlef; j++) {
                cnts_out[m_out++] = cnts[j];
            }

            cnts_out[m_out++] = cnt_botlef + pbot;
        }

        uint cnt_botlef_res = cnts[j_botlef] - cnt_botlef;
        uint ntop = nTopPixels(0, cnt_botlef_res, h);
        uint nbot = nBottomPixels(0, cnt_botlef_res, h);
        cnts_out[m_out - 1] += ptop * ntop + pbot * nbot + cnt_botlef_res;
    }

    uint y = r_end_botlef % h;
    uint delay = 0;
    for (siz j = j_botlef + 1; j < j_toprig; j++) {
        uint cnt = cnts[j];
        if (y == 0 || y + cnt >= h) {
            memcpy(cnts_out + m_out, cnts + j - delay, sizeof(uint) * delay);
            m_out += delay;
            uint ntop = nTopPixelsNonzero(y, cnt, h);
            uint nbot = nBottomPixels(y, cnt, h);
            cnts_out[m_out++] = ptop * ntop + pbot * nbot + cnt;
            delay = 0;
        } else {
            delay++;
        }
        y = (y + cnt) % h;
    }

    memcpy(cnts_out + m_out, cnts + j_toprig - delay, sizeof(uint) * delay);
    m_out += delay;

    if (right_col_same) {
        uint cnt = cnts[m - 1];
        uint ntop = nTopPixels(y, cnt, h);
        uint nbot = nBottomPixels(y, cnt, h);
        uint val = prig * h_out + ptop * ntop + pbot * nbot + cnt;

        if (j_botrig == j_botlef) {
            cnts_out[m_out - 1] += prig * h_out;
        } else if (j_botlef == j_toprig) {
            cnts_out[m_out - 1] = val;
        } else {
            cnts_out[m_out++] = val;
        }
    } else {
        if (j_botlef != j_toprig && w > 1) {
            uint cnt_toprig_rest = cnts[j_toprig] - cnt_toprig;
            uint ntop = nTopPixels(y, cnt_toprig_rest, h);
            uint nbot = nBottomPixels(y, cnt_toprig_rest, h);
            cnts_out[m_out++] = ptop * ntop + pbot * nbot + cnt_toprig_rest;
        }

        for (siz x = 0; x < prig + (w > 1 ? 1 : 0); x++) {
            bool v_out = m_out % 2 != 0;
            if (x == 0 && j_botlef == j_toprig) {
                // noop
            } else if (v_out != v_toprig) {
                cnts_out[m_out - 1] += cnt_toprig + ptop;
            } else {
                cnts_out[m_out++] = cnt_toprig + ptop;
            }

            for (siz j = j_toprig + 1; j < m - 1; j++) {
                cnts_out[m_out++] = cnts[j];
            }
            cnts_out[m_out++] = cnt_botrig + pbot;
        }
    }

}



