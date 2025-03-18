#include <stdlib.h> // for malloc, free
#include <stdbool.h> // for bool
#include <string.h> // for memcpy
#include <stdint.h> // for uint32_t
#include <stdio.h> // for printf
#include "boolfuncs.h"
#include "minmax.h"


const uint _X = 12;  // Binary 1100
const uint _Y = 10;  // Binary 1010
const uint BOOLFUNC_AND = _X & _Y;
const uint BOOLFUNC_OR = _X | _Y;
const uint BOOLFUNC_XOR = _X ^ _Y;
const uint BOOLFUNC_SUB = _X & ~_Y;

// Apply a boolean function (encoded as a truth table uint) to two boolean values
static inline bool _applyBoolFunc(bool x, bool y, uint boolfunc);

// _rleMergeCustom allows any custom C function to be used as a boolean function
// The input values are packed into an uint32, starting from the LSB
// custom_data is given as the third argument to the custom_func
static void _rleMergeCustom(
    const RLE **R, RLE *M, siz n, bool(*custom_func)(uint32_t, siz, void*), void* custom_data);

// _rleMergeCustom uses the following struct to keep track of the iteration of each mask
// which is done in one pass together.
struct uintIterator{
    uint curr; // remaining pixel count of the current run
    uint *start; // pointer to the first run
    uint *next; // pointer to the next run
    uint *end; // pointer to the end of the cnts array
};

// This is used in rleMergeLookup to read out the appropriate "row" from the truth table based on
// the values of the input masks encoded in a uint32 (starting from the LSB, n bits)
static inline bool _boolfunc_LookupMulti(uint32_t vs, siz n, void *multiboolfunc);
// rleMergeLookup packs the multiboolfunc and n_funcparts into a struct for custom_data in _rleMergeCustom:
struct BoolFuncParts {
    siz n;
    uint64_t *vals;
};


// rleMergeDiffOr simply calls _rleMergeCustom with the following custom function:
static inline bool _boolfunc_DiffOr(uint32_t vs, siz n, void *kp);

// rleMergeWeightedAtLeast packs the weights and threshold into a Weights struct for custom_data in _rleMergeCustom
struct Weights {
    double *weights;
    double threshold;
};
// ... and calls _rleMergeCustom with the following custom function:
static inline bool _boolfunc_weightedAtLeast(uint32_t vs, siz n, void *ptr);

static inline bool _boolfunc_atLeast(uint32_t vs, siz n, void *kp);

// Following are helpers for _boolfunc_atLeast and _boolfunc_weightedAtLeast
// _boolfunc_atLeast uses popcount to count the number of 1s in the input
// _boolfunc_weightedAtLeast uses count_trailing_zeros to find the index of the next 1 bit, and then
// uses the index to find the weight of the corresponding mask.

#if defined(__GNUC__) || defined(__clang__)
// These functions often have single instruction hardware support
// if yes, GCC and Clang will then use the hardware instruction.
static int count_trailing_zeros(uint32_t x) {
    return x == 0 ? 32 : __builtin_ctz(x);
}
static int popcount(uint32_t x) {
    return __builtin_popcount(x);
}

#else
// Fallback portable implementation. Actually, these code patterns are typically also recognized
// by compilers and optimized to the same instructions as the builtins.
static int count_trailing_zeros(uint32_t x) {
    if (x == 0) return 32;
    static const char table[32] = {
        0, 1, 28, 2, 29, 14, 24, 3, 30, 22, 20, 15, 25, 17, 4, 8,
        31, 27, 13, 23, 21, 19, 16, 7, 26, 12, 18, 6, 11, 5, 10, 9
    };
    return table[((uint32_t)((x & -x) * 0x077CB531U)) >> 27];
}
static int popcount(uint32_t x) {
    int count = 0;
    while (x) {
        x &= (x - 1); // Clears the lowest set bit
        count++;
    }
    return count;
}
#endif

//---------------------------------------


void rleComplement(const RLE *R, RLE *M, siz n) {
    for (siz i = 0; i < n; i++) {
        siz h = R[i].h;
        siz w = R[i].w;
        if (R[i].m == 0 || h == 0 || w == 0) {
            rleInit(&M[i], h, w, 0);
        } else if (R[i].m > 0 && R[i].cnts[0] == 0) {
            // if the first run has size 0, we can just remove it
            rleFrCnts(&M[i], h, w, R[i].m - 1, R[i].cnts + 1);
        } else {
            // if the first run has size > 0, we need to add a new run of 0s at the beginning
            rleInit(&M[i], h, w, R[i].m + 1);
            M[i].cnts[0] = 0;
            memcpy(M[i].cnts + 1, R[i].cnts, sizeof(uint) * R[i].m);
        }
    }
}

//void rleComplementInplace(RLE *R, siz n) {
//    for (siz i = 0; i < n; i++) {
//        siz h = R[i].h;
//        siz w = R[i].w;
//        if (R[i].m == 0 || h == 0 || w == 0) {
//            continue;
//        } else if (R[i].m > 0 && R[i].cnts[0] == 0) {
//            // if the first run has size 0, we can just remove it
//            memmove(R[i].cnts, R[i].cnts + 1, sizeof(uint) * (R[i].m - 1));
//            rleRealloc(&R[i], R[i].m - 1);
//        } else {
//            // if the first run has size > 0, we need to add a new run of 0s at the beginning
//            rleRealloc(&R[i], R[i].m + 1);
//            memmove(R[i].cnts + 1, R[i].cnts, sizeof(uint) * (R[i].m - 1));
//            R[i].cnts[0] = 0;
//        }
//    }
//}

void rleComplementInplace(RLE *R, siz n) {
    for (siz i = 0; i < n; i++) {
        siz h = R[i].h;
        siz w = R[i].w;
        if (R[i].m == 0 || h == 0 || w == 0) {
            continue;
        } else if (R[i].m > 0 && R[i].cnts[0] == 0) {
            // if the first run has size 0, we can just remove it
            R[i].cnts++;
            R[i].m--;
        } else {
            // if the first run has size > 0, we need to add a new run of 0s at the beginning
            if (R[i].alloc < R[i].cnts) {
                R[i].cnts--;
                R[i].m++;
                R[i].cnts[0] = 0;
            } else {
                rleRealloc(&R[i], R[i].m + 1);
                memmove(R[i].cnts + 1, R[i].cnts, sizeof(uint) * (R[i].m-1));
                R[i].cnts[0] = 0;
            }
        }
    }
}


void rleMerge2(const RLE *A, const RLE *B, RLE *M, uint boolfunc) {
    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    rleInit(M, A->h, A->w, sizMin(A->h * A->w + 1, A->m + B->m));
    if (M->m == 0) {
        return;
    }

    uint ca = A->cnts[0];
    uint cb = B->cnts[0];
    bool v_prev = false;
    bool va = false;
    bool vb = false;
    siz m = 0;
    siz a = 1;
    siz b = 1;
    uint cc = 0;
    uint ct;
    do {
        uint c = uintMin(ca, cb);
        cc += c; // add the current consumed amount to the output run

        // consume from the current run of A
        ca -= c;
        if (!ca && a < A->m) { // consumed a whole run from A and there are more
            ca = A->cnts[a++]; // get next run from A
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

        bool v_current = _applyBoolFunc(va, vb, boolfunc);

        if (v_current != v_prev || ct == 0) {
            // if the value changed or we consumed all runs, we need to save the current run to the output
            M->cnts[m++] = cc;
            cc = 0;
            v_prev = v_current;
        }
    } while (ct > 0); // continue until we consumed all runs from both A and B

    rleRealloc(M, m);
}

void rleMergeMultiFunc(const RLE **R, RLE *M, siz n, uint* boolfuncs) {
    if (n == 0) {
        rleInit(M, 0, 0, 0);
        return;
    }
    if (n == 1) {
        rleCopy(R[0], M);
        return;
    }

    siz h = R[0]->h;
    siz w = R[0]->w;
    if (h == 0 || w == 0) {
        rleInit(M, h, w, 0);
        return;
    }

    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    siz m_total = 0;
    for (siz i = 0; i < n; i++) {
        m_total += R[i]->m;
    }

    RLE tmp;
    siz m_max = sizMin(h * w + 1, m_total);
    rleInit(&tmp, h, w, m_max);
    rleInit(M, h, w, m_max);
    RLE *A = R[0];

    siz m;
    for (siz i = 1; i < n; i++) {
        const RLE *B = R[i];
        if (i == 2) {
            A = &tmp;
        }
        if (i >= 2) {
            rleSwap(A, M);
        }

        uint ca = A->cnts[0];
        uint cb = B->cnts[0];
        bool v_prev = false;
        bool va = false;
        bool vb = false;
        m = 0;
        siz a = 1;
        siz b = 1;
        uint cc = 0;
        uint ct;
        do {
            uint c = uintMin(ca, cb);
            cc += c; // add the current consumed amount to the output run

            // consume from the current run of A
            ca -= c;
            if (!ca && a < A->m) { // consumed a whole run from A and there are more
                ca = A->cnts[a++]; // get next run from A
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

            bool v_current = _applyBoolFunc(va, vb, boolfuncs[i-1]);

            if (v_current != v_prev || ct == 0) {
                // if the value changed or we consumed all runs, we need to save the current run to the output
                M->cnts[m++] = cc;
                cc = 0;
                v_prev = v_current;
            }
        } while (ct > 0); // continue until we consumed all runs from both A and B
        M->m = m;
    }
    rleRealloc(M, m);
    if (A != R[0]) {
        rleFree(A); // free the memory of intermediate result
    } else {
        rleFree(&tmp);
    }
}


void rleMergePtr(const RLE **R, RLE *M, siz n, uint boolfunc) {
    if (n == 0) {
        rleInit(M, 0, 0, 0);
        return;
    }
    if (n == 1) {
        rleCopy(R[0], M);
        return;
    }

    siz h = R[0]->h;
    siz w = R[0]->w;
    if (h == 0 || w == 0) {
        rleInit(M, h, w, 0);
        return;
    }

    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    siz m_total = 0;
    for (siz i = 0; i < n; i++) {
        m_total += R[i]->m;
    }

    RLE tmp;
    siz m_max = sizMin(h * w + 1, m_total);
    rleInit(&tmp, h, w, m_max);
    rleInit(M, h, w, m_max);
    RLE *A = R[0];

    siz m;
    for (siz i = 1; i < n; i++) {
        const RLE *B = R[i];
        if (i == 2) {
            A = &tmp;
        }
        if (i >= 2) {
            rleSwap(A, M);
        }

        uint ca = A->cnts[0];
        uint cb = B->cnts[0];
        bool v_prev = false;
        bool va = false;
        bool vb = false;
        m = 0;
        siz a = 1;
        siz b = 1;
        uint cc = 0;
        uint ct;
        do {
            uint c = uintMin(ca, cb);
            cc += c; // add the current consumed amount to the output run

            // consume from the current run of A
            ca -= c;
            if (!ca && a < A->m) { // consumed a whole run from A and there are more
                ca = A->cnts[a++]; // get next run from A
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

            bool v_current = _applyBoolFunc(va, vb, boolfunc);

            if (v_current != v_prev || ct == 0) {
                // if the value changed or we consumed all runs, we need to save the current run to the output
                M->cnts[m++] = cc;
                cc = 0;
                v_prev = v_current;
            }
        } while (ct > 0); // continue until we consumed all runs from both A and B
        M->m = m;
    }
    rleRealloc(M, m);
    if (A != R[0]) {
        rleFree(A); // free the memory of intermediate result
    } else {
        rleFree(&tmp);
    }
}

void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc) {
    if (n == 0) {
        rleInit(M, 0, 0, 0);
        return;
    }
    if (n == 1) {
        rleCopy(&R[0], M);
        return;
    }

    siz h = R[0].h;
    siz w = R[0].w;
    if (h == 0 || w == 0) {
        rleInit(M, h, w, 0);
        return;
    }

    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    siz m_total = 0;
    for (siz i = 0; i < n; i++) {
        m_total += R[i].m;
    }

    RLE tmp;
    siz m_max = sizMin(h * w + 1, m_total);
    rleInit(&tmp, h, w, m_max);
    rleInit(M, h, w, m_max);
    RLE *A = &R[0];

    siz m;
    for (siz i = 1; i < n; i++) {
        const RLE *B = &R[i];
        if (i == 2) {
            A = &tmp;
        }
        if (i >= 2) {
            rleSwap(A, M);
        }

        uint ca = A->cnts[0];
        uint cb = B->cnts[0];
        bool v_prev = false;
        bool va = false;
        bool vb = false;
        m = 0;
        siz a = 1;
        siz b = 1;
        uint cc = 0;
        uint ct;
        do {
            uint c = uintMin(ca, cb);
            cc += c; // add the current consumed amount to the output run

            // consume from the current run of A
            ca -= c;
            if (!ca && a < A->m) { // consumed a whole run from A and there are more
                ca = A->cnts[a++]; // get next run from A
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

            bool v_current = _applyBoolFunc(va, vb, boolfunc);

            if (v_current != v_prev || ct == 0) {
                // if the value changed or we consumed all runs, we need to save the current run to the output
                M->cnts[m++] = cc;
                cc = 0;
                v_prev = v_current;
            }
        } while (ct > 0); // continue until we consumed all runs from both A and B
        M->m = m;
    }
    rleRealloc(M, m);
    if (A != &R[0]) {
        rleFree(A); // free the memory of intermediate result
    } else {
        rleFree(&tmp);
    }
}



static void _rleMergeCustom(
    const RLE **R, RLE *M, siz n, bool(*custom_func)(uint32_t, siz, void*), void* custom_data) {

    if (n == 0) {
        rleInit(M, 0, 0, 0);
        return;
    } else if (n == 1) {
        rleCopy(R[0], M);
        return;
    }

    siz h = R[0]->h;
    siz w = R[0]->w;

    if (h == 0 || w == 0) {
        rleInit(M, h, w, 0);
        return;
    }

    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    siz m_total = 0;
    for (siz i = 0; i < n; i++) {
        m_total += R[i]->m;
    }
    rleInit(M, h, w, sizMin(h * w + 1, m_total));

    uint32_t vs = 0;  // bitset of the current value of each run, first in the lowest bit

    struct uintIterator *iters = malloc(n * sizeof(struct uintIterator)); // the pointer to the next run of each RLE
    for (siz i = 0; i < n; i++) {
        iters[i].curr = R[i]->cnts[0];
        iters[i].next = &R[i]->cnts[1];
        iters[i].end = &R[i]->cnts[R[i]->m];
    }

    bool v_prev = false;
    siz m = 0;
    uint cc = 0;
    bool more_in_any;
    do {
        // find the smallest count among the current runs
        uint c = iters[0].curr;
        for (siz i = 1; i < n; i++) {
            c = uintMin(c, iters[i].curr);
        }
        cc += c; // add the current consumed amount to the output run

        more_in_any = false; // used to check if there are more runs to consume in any
        // consume from the current runs
        for (siz i = 0; i < n; i++) {
            struct uintIterator *it = &iters[i];
            it->curr -= c;

            if (it->curr == 0 && it->next != it->end) { // consumed a whole run from this and there are more
                it->curr = *(it->next++); // get next run
                vs ^= ((uint32_t)1) << i; // toggle the value in the bitset
                more_in_any = true;
            } else if (it->curr > 0) {
                more_in_any = true;
            }
        }
        bool v_current = custom_func(vs, n, custom_data);

        if (v_current != v_prev || !more_in_any) {
            // if the value changed or we consumed all runs, we need to save the current run to the output
            M->cnts[m++] = cc;
            cc = 0;
            v_prev = v_current;
        }
    } while (more_in_any); // continue until we consumed all runs from both A and B

    rleRealloc(M, m);
    free(iters);
}

void rleMergeWeightedAtLeast2(
    const RLE **R, RLE *M, siz n, double *weights, double threshold) {

    if (n == 0) {
        rleInit(M, 0, 0, 0);
        return;
    } else if (n == 1) {
        rleCopy(R[0], M);
        return;
    }

    siz h = R[0]->h;
    siz w = R[0]->w;

    if (h == 0 || w == 0) {
        rleInit(M, h, w, 0);
        return;
    }

    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    siz m_total = 0;
    for (siz i = 0; i < n; i++) {
        m_total += R[i]->m;
    }
    rleInit(M, h, w, sizMin(h * w + 1, m_total));


    struct uintIterator *iters = malloc(n * sizeof(struct uintIterator)); // the pointer to the next run of each RLE
    for (siz i = 0; i < n; i++) {
        iters[i].curr = R[i]->cnts[0];
        iters[i].start = &R[i]->cnts[0];
        iters[i].next = &R[i]->cnts[1];
        iters[i].end = &R[i]->cnts[R[i]->m];
    }

    double sum = 0;
    double kahan_compensation = 0;

    bool v_prev = false;
    siz m = 0;
    uint cc = 0;
    bool more_in_any;
    do {
        // find the smallest count among the current runs
        uint c = iters[0].curr;
        for (siz i = 1; i < n; i++) {
            c = uintMin(c, iters[i].curr);
        }
        cc += c; // add the current consumed amount to the output run

        more_in_any = false; // used to check if there are more runs to consume in any
        // consume from the current runs
        for (siz i = 0; i < n; i++) {
            struct uintIterator *it = &iters[i];
            it->curr -= c;

            if (it->curr == 0 && it->next != it->end) { // consumed a whole run from this and there are more
                bool isset = (it->next - it->start) % 2;
                it->curr = *(it->next++); // get next run

                // Kahan summation to avoid losing low-order bits
                double y = (
                    isset ?
                    weights[i] - kahan_compensation :
                    -(weights[i] + kahan_compensation));
                double t = sum + y;
                kahan_compensation = (t - sum) - y;
                sum = t;
                more_in_any = true;
            } else if (it->curr > 0) {
                more_in_any = true;
            }
        }
        bool v_current = sum >= threshold;

        if (v_current != v_prev || !more_in_any) {
            // if the value changed or we consumed all runs, we need to save the current run to the output
            M->cnts[m++] = cc;
            cc = 0;
            v_prev = v_current;
        }
    } while (more_in_any); // continue until we consumed all runs from both A and B

    rleRealloc(M, m);
    free(iters);
}

void rleMergeAtLeast2(const RLE **R, RLE *M, siz n, uint k) {
    if (n == 0) {
        rleInit(M, 0, 0, 0);
        return;
    } else if (n == 1) {
        rleCopy(R[0], M);
        return;
    }

    siz h = R[0]->h;
    siz w = R[0]->w;

    if (h == 0 || w == 0) {
        rleInit(M, h, w, 0);
        return;
    }

    // maximum number of runs is min(h*w+1, sum(m)) (e.g., odd-height checkerboard starting with 1)
    siz m_total = 0;
    for (siz i = 0; i < n; i++) {
        m_total += R[i]->m;
    }
    rleInit(M, h, w, sizMin(h * w + 1, m_total));

    uint count = 0;

    struct uintIterator *iters = malloc(n * sizeof(struct uintIterator)); // the pointer to the next run of each RLE
    for (siz i = 0; i < n; i++) {
        iters[i].curr = R[i]->cnts[0];
        iters[i].start = &R[i]->cnts[0];
        iters[i].next = &R[i]->cnts[1];
        iters[i].end = &R[i]->cnts[R[i]->m];
    }

    bool v_prev = false;
    siz m = 0;
    uint cc = 0;
    bool more_in_any;
    do {
        // find the smallest count among the current runs
        uint c = iters[0].curr;
        for (siz i = 1; i < n; i++) {
            c = uintMin(c, iters[i].curr);
        }
        cc += c; // add the current consumed amount to the output run
        more_in_any = false; // used to check if there are more runs to consume in any
        // consume from the current runs
        for (siz i = 0; i < n; i++) {
            struct uintIterator *it = &iters[i];
            it->curr -= c;

            if (it->curr == 0 && it->next != it->end) { // consumed a whole run from this and there are more
                bool isset = (it->next - it->start) % 2;
                it->curr = *(it->next++); // get next run
                if (isset) {
                    count++;
                } else {
                    count--;
                }
                more_in_any = true;
            } else if (it->curr > 0) {
                more_in_any = true;
            }
        }
        bool v_current = count >= k;

        if (v_current != v_prev || !more_in_any) {
            // if the value changed or we consumed all runs, we need to save the current run to the output
            M->cnts[m++] = cc;
            cc = 0;
            v_prev = v_current;
        }
    } while (more_in_any); // continue until we consumed all runs from both A and B

    rleRealloc(M, m);
    free(iters);
}

static inline bool _boolfunc_atLeast(uint32_t vs, siz n, void *kp) {
    return popcount(vs) >= *(uint*)kp;
}
void rleMergeAtLeast(const RLE **R, RLE *M, siz n, uint k) {
    _rleMergeCustom(R, M, n, _boolfunc_atLeast, &k);
}

static inline bool _boolfunc_weightedAtLeast(uint32_t vs, siz n, void *ptr) {
    struct Weights *weights = (struct Weights*) ptr;
    double sum = 0;
    while (vs) {
        int i = count_trailing_zeros(vs);
        sum += weights->weights[i];
        vs &= (vs - 1); // Clears the lowest set bit
    }
    return sum >= weights->threshold;
}
void rleMergeWeightedAtLeast(const RLE **R, RLE *M, siz n, double *weights, double threshold) {
    struct Weights w;
    w.weights = weights;
    w.threshold = threshold;
    _rleMergeCustom(R, M, n, _boolfunc_weightedAtLeast, &w);
}

static inline bool _boolfunc_DiffOr(uint32_t vs, siz n, void *kp) {
    return ((vs & 1) && !(vs & 2)) || (vs & 4);
}

void rleMergeDiffOr(const RLE *A, const RLE *B, const RLE *C, RLE *M) {
    const RLE *R[3] = {A, B, C};
    _rleMergeCustom(R, M, 3, _boolfunc_DiffOr, NULL);
}


static inline bool _boolfunc_LookupMulti(uint32_t vs, siz n, void *multiboolfunc) {
    struct BoolFuncParts* mbf = (struct BoolFuncParts*) multiboolfunc;
    return (mbf->vals[vs / 64] >> (vs % 64)) & 1;
}

void rleMergeLookup(const RLE **R, RLE *M, siz n, uint64_t *multiboolfunc, siz n_funcparts) {
    struct BoolFuncParts boolFuncParts;
    boolFuncParts.n = n_funcparts;
    boolFuncParts.vals = multiboolfunc;
    _rleMergeCustom(R, M, n, _boolfunc_LookupMulti, &boolFuncParts);
}

static inline bool _applyBoolFunc(bool x, bool y, uint boolfunc) {
    // boolfunc contains in its lowest 4 bits the truth table of the boolean function
    // (x << 1 | y) is the row index of the truth table (same as x*2 + y)
    // the value of the boolean function is the bit at that index, so we shift the selected bit to
    // the lowest bit and mask it with 1 to get this last bit.
    return (boolfunc >> ((int) x << 1 | (int) y)) & 1;
}