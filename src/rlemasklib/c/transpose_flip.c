#include <stdlib.h> // for malloc, realloc, free
#include <stdbool.h> // for bool
#include <string.h> // for memcpy
#include "basics.h"
#include "minmax.h"
#include "boolfuncs.h"
#include "transpose_flip.h"

void rleTranspose(const RLE *R, RLE *M) {
    if (R->m == 0 || R->h == 0 || R->w == 0) {
        rleInit(M, R->w, R->h, 0);
        return;
    }

    if (R->m == 1 || R->h == 1 || R->w == 1 || (R->m == 2 && R->cnts[0] == 0)) {
        rleFrCnts(M, R->w, R->h, R->m, R->cnts);
        return;
    }

    RLE shifted;
    rleRoll(R, &shifted);

    RLE xor;
    rleMerge2(&shifted, R, &xor, BOOLFUNC_XOR);
    rleFree(&shifted);

    uint r = 0;
    uint *n_switch_per_row = calloc(R->h, sizeof(uint));
    uint sum = 0;
    for (siz i = 1; i < xor.m; i+=2) {
        r += xor.cnts[i-1];
        uint r_end = r + xor.cnts[i];
        for (; r < r_end; r++) {
            uint row = r % R->h;
            n_switch_per_row[row]++;
            sum++;
        }
    }
    siz m_out = sum + 1;
    uint *cnts_out = rleInit(M, R->w, R->h, m_out);
    uint **ptrs = malloc(R->h * sizeof(uint*));

    uint *ptr = cnts_out;
    for (siz i = 0; i < R->h; i++) {
        ptrs[i] = ptr;
        ptr += n_switch_per_row[i];
    }

    r = 0;
    for (siz i = 1; i < xor.m; i+=2) {
        r += xor.cnts[i-1];
        uint r_end = r + xor.cnts[i];
        for (; r < r_end; r++) {
            uint row = r % R->h;
            uint col = r / R->h;
            *(ptrs[row]++) = row * R->w + col;
        }
    }

    uint prev_r = 0;
    for (siz i =0; i < m_out-1; i++) {
        uint r = cnts_out[i];
        cnts_out[i] = r - prev_r;
        prev_r = r;
    }
    cnts_out[m_out-1] = R->w*R->h - prev_r;

    // Clean up
    free(n_switch_per_row);
    free(ptrs);
    rleFree(&xor);
}



void rleRoll(const RLE *R, RLE *M) {
    uint h = R->h;
    uint w = R->w;
    siz m = R->m;
    uint *cnts_out = rleInit(M, h, w, m + 2);
    siz j_first;
    siz m_out;
    siz r_first;
    uint *cnts_in = R->cnts;
    {
        uint box_start = h * (w - 1);
        uint cnt_first;
        siz r = h * w;
        for (siz j = m - 1; j >= 0; j--) {
            uint r_end = r;
            r -= cnts_in[j];
            if (r <= box_start) {
                j_first = j;
                r_first = r_end;
                cnt_first = r_end - box_start;
                break;
            }
        }

        uint j_last = cnts_in[m - 1] == 1 ? m - 2 : m - 1;

        if (j_first == j_last) {
            // The box is fully contained in a single run.
            if (j_first % 2 == 0) {
                // The run is 0s
                cnts_out[0] = h;
                m_out = 1;
            } else {
                // The run is 1s
                cnts_out[0] = 1;
                cnts_out[1] = h - 1;
                m_out = 2;
            }
        } else {
            if (j_first % 2 == 0) {
                // The first run is 0s
                cnts_out[0] = cnt_first + 1;
                memcpy(cnts_out + 1, cnts_in + j_first + 1, sizeof(uint) * (m - j_first - 1));
                m_out = m - j_first;
                cnts_out[m_out - 1]--;
            } else {
                // The first run is 1s
                cnts_out[0] = 1;
                cnts_out[1] = cnt_first;
                memcpy(cnts_out + 2, cnts_in + j_first + 1, sizeof(uint) * (m - j_first - 1));
                m_out = m - j_first + 1;
                cnts_out[m_out - 1]--;
            }
        }

    }
    {
        uint box_w = w - 1;
        if (box_w == 0) {
            rleRealloc(M, m_out);
            return;
        }
        uint box_end = box_w * h;

        siz j_last;
        uint cnt_last;
        uint r = r_first;
        for (siz j = j_first; j >= 0; j--) {
            r -= cnts_in[j];
            if (r < box_end) {
                j_last = j;
                cnt_last = box_end - r;
                break;
            }
        }
        bool lastPixelA = m_out % 2 == 0;
        bool firstPixelB = cnts_in[0] == 0;

        if (!lastPixelA) {
            if (j_last == 0) {
                cnts_out[m_out - 1] += cnt_last;
            } else {
                cnts_out[m_out - 1] += cnts_in[0];
                memcpy(cnts_out + m_out, cnts_in + 1, sizeof(uint) * (j_last - 1));
                m_out += j_last - 1;
                cnts_out[m_out++] = cnt_last;
            }
        } else if (firstPixelB){
            if (j_last == 1) {
                cnts_out[m_out - 1] += cnt_last;
            } else {
                cnts_out[m_out - 1] += cnts_in[1];
                memcpy(cnts_out + m_out, cnts_in + 2, sizeof(uint) * (j_last - 2));
                m_out += j_last - 2;
                cnts_out[m_out++] = cnt_last;
            }
        } else {
            memcpy(cnts_out + m_out, cnts_in, sizeof(uint) * j_last);
            m_out += j_last;
            cnts_out[m_out++] = cnt_last;
        }

        rleRealloc(M, m_out);
    }
}

void rleVerticalFlip(const RLE* R, RLE* M) {
    if (R->m == 0 || R->h == 0 || R->w == 0) {
        rleInit(M, R->h, R->w, 0);
        return;
    }
    siz m = R->m;
    siz h = R->h;
    siz w = R->w;
    uint *cnts = R->cnts;

    siz m_out = uintMin(m * 5, m + w * 2);
    uint *cnts_out = rleInit(M, R->h, R->w, m_out);

    siz j_out = 0;
    siz r = 0;
    siz x = 1;

    siz j_start = (cnts[0] > 0) ? 0 : 1;
    uint prev_end_j = j_start;
    uint prev_end_remainder = cnts[j_start];
    for (siz j = j_start; j < m;) {
        // find the run that has the bottom pixel:
        uint cnt = (j == prev_end_j ? prev_end_remainder: cnts[j]);
        bool has_bottom = r + cnt >= h * x;

        if (!has_bottom) {
            // the run does not have the bottom pixel
            j++;
            r += cnt;
            continue;
        }

        // the run has the bottom pixel
        uint y = r % h;
        uint to_bottom;
        if (y ==0) {
            to_bottom = (cnt / h) * h;
            x += cnt / h;
        } else {
            to_bottom = h - y;
            x++;
        }

        uint to_add = 0;
        if (j % 2 != j_out % 2) {
            if (j_out == 0) {
                cnts_out[j_out++] = 0;
            } else{
                j_out--;
                to_add = cnts_out[j_out];
            }
        }

        for (long jj = j; jj >= prev_end_j; jj--) {
            if (jj == j) {
                cnts_out[j_out++] = to_add + to_bottom;
                cnt -= to_bottom;
            } else if (jj == prev_end_j) {
                cnts_out[j_out++] = prev_end_remainder;
            } else {
                cnts_out[j_out++] = cnts[jj];
            }
        }
        if (cnt == 0) {
            j++;
            prev_end_remainder = cnts[j];
        } else {
            prev_end_remainder = cnt;
        }
        prev_end_j = j;
        r += to_bottom;
    }

    rleRealloc(M, j_out);
}



void rleRotate180Inplace(RLE *R) {
    if (R->m <= 1 || R->h == 0 || R->w == 0) {
        return;
    }

    if (R->m % 2 == 0 && R->cnts[0] > 0) {
        // the number of runs is even, hence the last pixel is 1
        // the first run is nonempty, hence the first pixel is 0
        // in this case we need an extra empty run of 0s at the end
        // (which will be the first run in the rotated RLE)
        // (this is because each RLE has to start with a run of 0s)
        rleRealloc(R, R->m + 1);
        R->cnts[R->m - 1] = 0;
    }
    // now we need to flip the order of the runs
    // p1 starts from the beginning, p2 starts from the end

    uint *p1 = R->cnts + (R->m % 2 == 0 && R->cnts[0] == 0 ? 1 : 0);
    uint *p2 = R->cnts + R->m - 1;
    while (p1 < p2) {
        uint tmp = *p1;
        *p1 = *p2;
        *p2 = tmp;
        p1++;
        p2--;
    }
    if (R->cnts[R->m - 1] == 0) {
        // the last run is empty, so we remove it
        rleRealloc(R, R->m - 1);
    }
}

void rleRotate180(const RLE *R, RLE *M) {
    if (R->m <= 1 || R->h == 0 || R->w == 0) {
        rleCopy(R, M);
        return;
    }
    siz m_out;
    if (R->m % 2 == 0 && R->cnts[0] > 0) {
        // last pixel is 1, first pixel is 0
        // the number of runs is even, hence the last pixel is 1
        // the first run is nonempty, hence the first pixel is 0
        // in this case we need an extra empty run of 0s as the first run of the result
        // (this is because each RLE has to start with a run of 0s)
        m_out = R->m + 1;
    } else if (R->m % 2 == 1 && R->cnts[0] == 0) {
        // first pixel is 1, last pixel is 0. The initial zero run won't be needed
        m_out = R->m - 1;
    } else {
        m_out = R->m;
    }
    // now we need to flip the order of the runs
    uint *cnts_out = rleInit(M, R->h, R->w, m_out);

    siz i_start;
    if (R->m % 2 == 0) {
        // last pixel is 1 in the input
        cnts_out[0] = 0;
        i_start=1;
    } else {
        i_start = 0;
    }

    uint *p1 = cnts_out + i_start;
    uint *p2 = R->cnts + R->m - 1;
    while (p1 != cnts_out + m_out) {
        *(p1++) = *(p2--);
    }
}
