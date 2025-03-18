#include <stdlib.h> // for malloc, free...
#include <string.h> // for memcpy
#include <stdbool.h> // for bool
#include <stddef.h> // for ptrdiff_t
#include "basics.h"


uint *rleInit(RLE *R, siz h, siz w, siz m) {
    R->h = h;
    R->w = w;
    R->m = m;

    if (m == 0) {
        R->alloc = NULL;
        R->cnts = NULL;
    } else {
        R->alloc = malloc(sizeof(uint) * (m+2));
        R->cnts = R->alloc + 1;
        //R->capacity = m + 2;
    }
    return R->cnts;
}

uint *rleFrCnts(RLE *R, siz h, siz w, siz m, uint *cnts) {
    rleInit(R, h, w, m);
    memcpy(R->cnts, cnts, sizeof(uint) * m);
    return R->cnts;
}

void rleCopy(const RLE *R, RLE *M) {
    rleFrCnts(M, R->h, R->w, R->m, R->cnts);
}

void rleFree(RLE *R) {
    free(R->alloc);
    R->alloc = NULL;
    R->cnts = NULL;
}

static uint *rleRealloc(RLE *R, siz m) {
    R->m = m;
    ptrdiff_t diff = R->cnts - R->alloc;
    R->alloc = realloc(R->alloc, sizeof(uint) * (m + diff));
    R->cnts = R->alloc + diff;
    return R->cnts;
}

void rleMoveTo(RLE *R, RLE *M) {
    rleFree(M);
    memcpy(M, R, sizeof(RLE));
    memset(R, 0, sizeof(RLE));
}

static void rleSwap(RLE *R, RLE *M) {
    RLE tmp;
    memcpy(&tmp, R, sizeof(RLE));
    memcpy(R, M, sizeof(RLE));
    memcpy(M, &tmp, sizeof(RLE));
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

byte rleGet(const RLE *R, siz i, siz j) {
    siz index = R->h * j + i;
    for (siz j = 0; j < R->m; j++) {
        uint cnt = R->cnts[j];
        if (index < cnt) {
            return j % 2;
        }
        index -= cnt;
    }
    return 0;
}

void rleOnes(RLE *R, siz h, siz w) {
    if (h == 0 || w == 0) {
        rleInit(R, h, w, 0);
        return;
    }
    rleInit(R, h, w, 2);
    R->cnts[0] = 0;
    R->cnts[1] = h * w;
}

void rleZeros(RLE *R, siz h, siz w) {
    if (h == 0 || w == 0) {
        rleInit(R, h, w, 0);
        return;
    }
    rleInit(R, h, w, 1);
    R->cnts[0] = h * w;
}

void rleSetInplace(RLE *R, siz y, siz x, byte value) {
    siz index = R->h * x + y;

    for (siz j = 0; j < R->m; j++) {
        uint cnt = R->cnts[j];
        if (index < cnt) {
            if (value == j % 2) {
                // Already the correct value
                return;
            }

            if (index == 0 && j > 0) {
                // Start of the run, and not the first run
                // extend the previous run, shrink the current run
                R->cnts[j]--;
                R->cnts[j - 1]++;
            } else if (index == cnt - 1 && j < R->m - 1) {
                // End of the run, and not the last run
                // extend the next run, shrink the current run
                R->cnts[j]--;
                R->cnts[j + 1]++;
            } else if (index == cnt - 1 && j == R->m - 1) {
                // End of the last run
                // we must add a new run of length 1 and shift the current run
                rleRealloc(R, R->m + 1);
                R->cnts[j]--;
                R->cnts[j + 1] = 1;
            } else {
                // Split the current run into three, we add two new runs
                // and move the rest of the runs to make space
                rleRealloc(R, R->m + 2);
                memmove(R->cnts + j + 3, R->cnts + j + 1, sizeof(uint) * (R->m - j - 3));
                R->cnts[j] = index;
                R->cnts[j + 1] = 1;
                R->cnts[j + 2] = cnt - index - 1;
            }
            return;
        }
        index -= cnt;
    }
}

bool rleEqual(const RLE *A, const RLE *B) {
    if (A->h != B->h || A->w != B->w) {
        return false;
    }

    if (A->m != B->m) {
        // the no-pixels mask can be represented either with 0 runs or with 1 run of length 0
        if (A->h > 0 || A->w > 0) {
            return false;
        }
        if (A->m == 0 && B->m == 1 && B->cnts[0] == 0) {
            return true;
        }
        if (A->m == 1 && B->m == 0 && A->cnts[0] == 0) {
            return true;
        }
        return false;
    }


    for (siz i = 0; i < A->m; i++) {
        if (A->cnts[i] != B->cnts[i]) {
            return false;
        }
    }
    return true;
}

static void rleEliminateZeroRuns(RLE *R) {
    if (R->m <= 1) {
        return;
    }
    siz m = R->m;
    uint *cnts = R->cnts;

    siz k = 0;
    siz j = 1;
    while (j < m) {
        if (cnts[j] > 0) {
            k++;
            cnts[k] = cnts[j];
            j++;
        } else {
            j++;
            if (j < m) {
                cnts[k] += cnts[j];
                j++;
            }
        }
    }
    rleRealloc(R, k + 1);
}
