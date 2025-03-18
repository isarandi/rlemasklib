#include <stdlib.h> // for malloc
#include "basics.h"
#include "minmax.h"
#include "moments.h"

void rleArea(const RLE *R, siz n, uint *a) {
    for (siz i = 0; i < n; i++) {
        a[i] = 0;
        for (siz j = 1; j < R[i].m; j += 2) {
            a[i] += R[i].cnts[j];
        }
    }
}

void rleCentroid(const RLE *R, double *xys, siz n) {
    for (siz i = 0; i < n; i++) {
        siz m = R[i].m;
        uint h = R[i].h;
        uint pos = 0;
        uint area = 0;
        double x = 0, y = 0;

        for (siz j = 1; j < m; j+=2) {
            pos += R[i].cnts[j-1];
            uint start_row = pos % h;
            uint start_col = pos / h;

            uint cnt = R[i].cnts[j];
            area += cnt;
            pos += cnt;

            // first part is whatever is within the first column
            // it might be a full or partial column
            uint cnt1 = uintMin(cnt, h - start_row);
            x += start_col * cnt1;
            y += (start_row + (cnt1 - 1) * 0.5) * cnt1;
            if (cnt1 == cnt) {
                continue;
            }

            // second part is one or more full columns
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

        xys[i * 2 + 0] = x / area;
        xys[i * 2 + 1] = y / area;
    }
}


void rleNonZeroIndices(const RLE *R, uint **coords_out, siz *n_out) {
    // this returns the (x,y) coordinates for all points where the mask is non-zero
    // the coordinates are stored in the coords array, which is allocated by this function
    siz m = R->m;
    siz h = R->h;
    siz w = R->w;
    uint *cnts = R->cnts;

    if (m == 0 || h == 0 || w == 0) {
        *n_out = 0;
        *coords_out = NULL;
        return;
    }

    uint area;
    rleArea(R, 1, &area);
    uint *coords = malloc(sizeof(uint) * area * 2);

    uint pos = 0;
    siz i_out = 0;
    uint x;
    uint y;

    for (siz j = 1; j < m; j += 2) {
        pos += cnts[j - 1];
        uint start_col = pos / h;
        uint start_row = pos % h;
        uint cnt = cnts[j];
        pos += cnt;

        // first part is whatever is within the first column, it might be a full or partial column
        uint cnt1 = uintMin(cnt, h - start_row);
        x = start_col;
        for (y = start_row; y < start_row + cnt1; y++) {
            coords[i_out++] = x;
            coords[i_out++] = y;
        }
        if (cnt1 == cnt) {
            continue;
        }

        // second part is one or more full columns
        uint num_full_cols = (cnt - cnt1) / h;
        uint cnt2 = num_full_cols * h;
        if (cnt2) {
            for (x = start_col + 1; x < start_col + num_full_cols + 1; x++) {
                for (y = 0; y < h; y++) {
                    coords[i_out++] = x;
                    coords[i_out++] = y;
                }
            }
        }

        // third part is a partial column
        uint cnt3 = cnt - cnt1 - cnt2;
        if (cnt3) {
            x = start_col + num_full_cols + 1;
            for (y = 0; y < cnt3; y++) {
                coords[i_out++] = x;
                coords[i_out++] = y;
            }
        }

    }

    *n_out = i_out;
    *coords_out = coords;
}