#include <stdbool.h>
#include "basics.h"
#include "transpose_flip.h"
#include "warp_common.h"


static void transformAffine(const double in[2], double out[2], const double M[6]) {
    out[0] = M[0] * in[0] + M[1] * in[1] + M[2];
    out[1] = M[3] * in[0] + M[4] * in[1] + M[5];
}

static void transformPerspective(const double inp[2], double outp[2], double H[9]) {
    double denom = H[6] * inp[0] + H[7] * inp[1] + H[8];
    outp[0] = (H[0] * inp[0] + H[1] * inp[1] + H[2]) / denom;
    outp[1] = (H[3] * inp[0] + H[4] * inp[1] + H[5]) / denom;
}

static void invert3x3(const double A[9], double A_inv[9]) {
    double A3746 = A[3] * A[7] - A[4] * A[6];
    double A4857 = A[4] * A[8] - A[5] * A[7];
    double A5638 = A[5] * A[6] - A[3] * A[8];

    double det = A[0] * A4857 + A[1] * A5638 + A[2] * A3746;
    double inv_det = 1.0 / det;
    A_inv[0] = +A4857 * inv_det;
    A_inv[1] = -(A[1] * A[8] - A[2] * A[7]) * inv_det;
    A_inv[2] = +(A[1] * A[5] - A[2] * A[4]) * inv_det;
    A_inv[3] = +A5638 * inv_det;
    A_inv[4] = +(A[0] * A[8] - A[2] * A[6]) * inv_det;
    A_inv[5] = -(A[0] * A[5] - A[2] * A[3]) * inv_det;
    A_inv[6] = +A3746 * inv_det;
    A_inv[7] = -(A[0] * A[7] - A[1] * A[6]) * inv_det;
    A_inv[8] = +(A[0] * A[4] - A[1] * A[3]) * inv_det;
}



static void transpose_3x3(const double A[9], double A_T[9]) {
    A_T[0] = A[0];
    A_T[1] = A[3];
    A_T[2] = A[6];
    A_T[3] = A[1];
    A_T[4] = A[4];
    A_T[5] = A[7];
    A_T[6] = A[2];
    A_T[7] = A[5];
    A_T[8] = A[8];
}



static void rleBackFlipRot(RLE *tmp, RLE *M, int k, bool flip) {
    switch (k) {
        case 0:
            if (flip) {
                RLE tmp4;
                rleTranspose(tmp, &tmp4);
                rleVerticalFlip(&tmp4, M);
                rleFree(&tmp4);
            } else {
                rleTranspose(tmp, M);
            }
            break;
        case 1:
            if (flip) {
                rleRotate180Inplace(tmp);
                rleMoveTo(tmp, M);
            } else {
                rleVerticalFlip(tmp, M);
            }
            break;
        case 2:
            if (flip) {
                RLE tmp4;
                rleVerticalFlip(tmp, &tmp4);
                rleTranspose(&tmp4, M);
                rleFree(&tmp4);
            } else {
                rleTranspose(tmp, M);
                rleRotate180Inplace(M);
            }
            break;
        case 3:
            if (flip) {
                rleMoveTo(tmp, M);
            } else {
                rleVerticalFlip(tmp, M);
                rleRotate180Inplace(M);
            }
            break;
        default:
            break;
    }
}



static int int_remainder(int a, int b) {
    return (a % b + b) % b;
}