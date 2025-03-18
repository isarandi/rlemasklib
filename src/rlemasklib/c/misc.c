#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <limits.h>
#include "basics.h"
#include "minmax.h"
#include "boolfuncs.h"
#include "pad_crop.h"
#include "transpose_flip.h"
#include "moments.h"
#include "shapes.h"
#include "misc.h"

static uint _size_after_striding(uint start, uint size, uint stride);
void rleDilateVerticalInplace(RLE *R, uint up, uint down) {
    // Dilate by a 3x1 kernel of 1s
    uint h = R->h;
    uint w = R->w;
    siz m = R->m;

    if (m <= 1 || h == 0 || w == 0 || (up == 0 && down == 0)) {
        return;
    }

    uint *cnts = R->cnts;
    siz r = cnts[0];
    for (siz j = 1; j < m; j += 2) {
        if (r % h != 0) {
            uint amount = uintMin(cnts[j - 1], up);
            cnts[j] += amount;
            cnts[j - 1] -= amount;
            r -= amount;
        }
        if ((r + cnts[j]) % h != 0) {
            uint amount = uintMin(cnts[j + 1], down);
            cnts[j] += amount;
            cnts[j + 1] -= amount;
        }
        r += cnts[j] + cnts[j + 1];
    }
    rleEliminateZeroRuns(R);
}

void rleConcatHorizontal(const RLE **R, RLE *M, siz n) {
    siz m_out = R[0]->m;
    siz w_out = R[0]->w;
    for (siz i = 1; i < n; i++) {
        m_out += R[i]->m;
        w_out += R[i]->w;
        bool lastPixelA = R[i-1]->m % 2 == 0;
        bool firstPixelB = R[i]->cnts[0] == 0;
        if (!lastPixelA) {
            m_out -= 1;
        } else if (firstPixelB) {
            m_out -= 2;
        }
    }

    uint *cnts_out = rleInit(M, R[0]->h, w_out, m_out);
    memcpy(cnts_out, R[0]->cnts, sizeof(uint) * R[0]->m);
    siz i_out = R[0]->m;

    for (siz i = 1; i < n; i++) {
        const RLE *A = R[i-1];
        const RLE *B = R[i];

        bool lastPixelA = A->m % 2 == 0;
        bool firstPixelB = B->cnts[0] == 0;
        if (!lastPixelA) {
            cnts_out[i_out - 1] += B->cnts[0];
            memcpy(cnts_out + i_out, B->cnts + 1, sizeof(uint) * (B->m - 1));
        } else if (firstPixelB){
            cnts_out[i_out - 1] += B->cnts[1];
            memcpy(cnts_out + i_out, B->cnts + 2, sizeof(uint) * (B->m - 2));
        } else {
            memcpy(cnts_out + i_out, B->cnts, sizeof(uint) * B->m);
        }
    }
}

void rleConcatVertical(const RLE **R, RLE *M, siz n) {
    RLE *paddedR = malloc(sizeof(RLE) * n);

    siz h_out = 0;
    for (siz i = 0; i < n; i++) {
        h_out += R[i]->h;
    }
    uint paddings[4] = {0, 0, 0, h_out};

    for (siz i = 0; i < n; i++) {
        paddings[3] -= R[i]->h;
        rleZeroPad(R[i], &paddedR[i], 1, paddings);
        paddings[2] += R[i]->h;
    }

    rleMerge(paddedR, M, n, BOOLFUNC_OR);
    for (siz i = 0; i < n; i++) {
        rleFree(&paddedR[i]);
    }
    free(paddedR);
}

static uint _size_after_striding(uint start, uint size, uint stride) {
    uint mod = start % stride;
    uint upmod = (mod == 0) ? stride : mod;
    return (size + upmod - 1) / stride;
}

void rleStrideInplace(RLE *R, siz sy, siz sx){
    if (sy == 1 && sx == 1) {
        return;
    }

    siz out_h = _size_after_striding(0, R->h, sy);
    siz out_w = _size_after_striding(0, R->w, sx);

    if (out_h == 0 || out_w == 0) {
        rleFree(R);
        R->m = 0;
        R->h = out_h;
        R->w = out_w;
        return;
    }

    siz r = 0;
    for (siz i = 0; i < R->m; i++) {
        uint cnt = R->cnts[i];
        uint x_start = r / R->h;
        uint x_last = (r + cnt - 1) / R->h;
        uint y = r % R->h;

        if (cnt == 0 || x_start == x_last) {
            if (x_start % sx == 0) {
                R->cnts[i] = _size_after_striding(y, cnt, sy);
            } else {
                R->cnts[i] = 0;
            }
        } else {
            uint first_colsize = R->h - y;
            uint last_colsize = (r + cnt - 1) % R->h + 1;
            uint n_full_mid_cols = (cnt - first_colsize - last_colsize) / R->h;
            uint n_keep_full_mid_cols = _size_after_striding(x_start + 1, n_full_mid_cols, sx);

            R->cnts[i] = out_h * n_keep_full_mid_cols;

            if (x_start % sx == 0) {
                R->cnts[i] += _size_after_striding(y, first_colsize, sy);
            }
            if (x_last % sx == 0) {
                R->cnts[i] += _size_after_striding(0, last_colsize, sy);
            }
        }
        r += cnt;
    }
    R->h = out_h;
    R->w = out_w;
    rleEliminateZeroRuns(R);
}

void rleRepeatInplace(RLE *R, siz nh, siz nw) {
    if (nh == 1 && nw == 1) {
        return;
    }

    siz h = R->h;
    siz w = R->w;
    siz h_out = h * nh;
    siz w_out = w * nw;

    if (h_out == 0 || w_out == 0) {
        rleFree(R);
        rleInit(R, h_out, w_out, 0);
        return;
    }

    if (nw > 1) {
        RLE transp;
        rleTranspose(R, &transp);
        for (siz i = 0; i < transp.m; i++) {
            transp.cnts[i] *= nw;
        }
        transp.h *= nw;
        rleFree(R);
        rleTranspose(&transp, R);
        rleFree(&transp);
    }

    if (nh > 1) {
        for (siz i = 0; i < R->m; i++) {
            R->cnts[i] *= nh;
        }
        R->h *= nh;
    }
}

void rleRepeat(const RLE *R, RLE *M, siz nh, siz nw) {
    if (nh == 1 && nw == 1) {
        rleCopy(R, M);
        return;
    }

    siz h = R->h;
    siz w = R->w;
    siz h_out = h * nh;
    siz w_out = w * nw;

    if (h_out == 0 || w_out == 0) {
        rleInit(M, h_out, w_out, 0);
        return;
    }

    if (nw > 1) {
        RLE transp;
        rleTranspose(R, &transp);
        for (siz i = 0; i < transp.m; i++) {
            transp.cnts[i] *= nw;
        }
        transp.h *= nw;

        rleTranspose(&transp, M);
        rleFree(&transp);
        if (nh > 1) {
            for (siz i = 0; i < M->m; i++) {
                M->cnts[i] *= nh;
            }
        }
        M->h *= nh;
    } else {
        rleInit(M, h_out, w_out, R->m);
        if (nh > 1) {
            for (siz i = 0; i < R->m; i++) {
                M->cnts[i] = R->cnts[i] * nh;
            }
        }
    }
}

//void rleContours(const RLE *R, RLE *M) {
//    if (R->m <= 1 || R->h == 0 || R->w == 0) {
//        rleZeros(M, R->h, R->w);
//        return;
//    }
//
//    RLE padded;
//    rleZeroPad(R, &padded, 1, (uint[4]){2, 2, 2, 2});
//
//    // up
//    padded.cnts[0] -= 1;
//    padded.cnts[padded.m - 1] += 1;
//    RLE tmp;
//    rleCopy(&padded, &tmp);
//
//    // down
//    padded.cnts[0] += 2;
//    padded.cnts[padded.m - 1] -= 2;
//    RLE result;
//    rleMerge2(&tmp, &padded, &result, BOOLFUNC_AND);
//    rleFree(&tmp);
//
//    // left
//    padded.cnts[0] -= 1 + padded.h;
//    padded.cnts[padded.m - 1] += 1 + padded.h;
//    rleMerge2(&result, &padded, &tmp, BOOLFUNC_AND);
//    rleFree(&result);
//
//    // right
//    padded.cnts[0] += 2 * padded.h;
//    padded.cnts[padded.m - 1] -= 2 * padded.h;
//    rleMerge2(&tmp, &padded, &result, BOOLFUNC_AND);
//    rleFree(&tmp);
//    rleFree(&padded);
//
//    rleCropInplace(&result, 1, (uint[4]){2, 2, R->w, R->h});
//    rleMerge2(R, &result, M, BOOLFUNC_SUB);
//    rleFree(&result);
//
//}

void rleContours(const RLE *R, RLE *M) {
    if (R->m <= 1 || R->h == 0 || R->w == 0) {
        rleZeros(M, R->h, R->w);
        return;
    }

    RLE padded;
    rleZeroPad(R, &padded, 1, (uint[4]){2, 2, 2, 2});

    // left
    RLE left;
    rleCopy(&padded, &left);
    left.cnts[0] -= padded.h;
    left.cnts[padded.m - 1] += padded.h;

    // padded becomes `right`
    padded.cnts[0] += padded.h;
    padded.cnts[padded.m - 1] -= padded.h;

    RLE left_and_right;
    rleMerge2(&left, &padded, &left_and_right, BOOLFUNC_AND);
    rleFree(&left);

    // padded becomes `up_and_down`
    // undo what we did for `right`
    padded.cnts[0] -= padded.h;
    padded.cnts[padded.m - 1] += padded.h;
    for (siz i = 1; i < padded.m; i += 2) {
        // every run of 1s of size 1 or 2 becomes 0, otherwise it becomes smaller by 2
        // this way only those points remain that have both an upper and a lower neighbor
        // the removed amounts are added to the upper and lower neighbors
        if (padded.cnts[i] >= 3) {
            padded.cnts[i-1] += 1;
            padded.cnts[i] -= 2;
            padded.cnts[i+1] += 1;
        } else {
            // it's okay to add all to this run, because it will be unified with the next run of 0s
            // by rleEliminateZeroRuns
            padded.cnts[i-1] += padded.cnts[i];
            padded.cnts[i] = 0;
        }
    }
    rleEliminateZeroRuns(&padded);

    RLE up_down_left_right; // rle of points whose all 4 neighbors are 1
    rleMerge2(&padded, &left_and_right, &up_down_left_right, BOOLFUNC_AND);
    rleFree(&left_and_right);
    rleFree(&padded);

    rleCropInplace(&up_down_left_right, 1, (uint[4]){2, 2, R->w, R->h});
    rleMerge2(R, &up_down_left_right, M, BOOLFUNC_SUB);
    rleFree(&up_down_left_right);
}

void rleHorizContours(const RLE *R, RLE *M) {
    if (R->m <= 1 || R->h == 0 || R->w == 0) {
        rleZeros(M, R->h, R->w);
        return;
    }

    RLE shifted;
    rleCopy(R, &shifted);
    shifted.cnts[0] += shifted.h;
    uint r = 0;
    for (siz i = shifted.m - 1; i > 0; i--) {
        r += shifted.cnts[i];
        if (r > R->h) {
            shifted.cnts[i] = r - R->h;
            shifted.m = i + 1;
            break;
        }
    }
    rleMerge2(R, &shifted, M, BOOLFUNC_XOR);
    rleFree(&shifted);
}


