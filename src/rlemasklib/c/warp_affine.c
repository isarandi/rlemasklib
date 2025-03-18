#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "basics.h"
#include "pad_crop.h"
#include "minmax.h"
#include "boolfuncs.h"
#include "warp_common.h"
#include "warp_affine.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

static void rotate_affine(const double H[6], double anchor[2], int k, double H_new[6]);
void invert_affine(const double A[6], double A_inv[6]);
static double _transformAffineY(double x, double y, double H[6]);

static void rleWarpAffine1(const RLE *R, RLE *M, siz h_out, double *H);
static void rleWarpAffine2(const RLE *R, RLE *M, siz h_out, double *H);

void rleWarpAffine(const RLE *R, RLE *M, siz h_out, siz w_out, double *H) {
    if (h_out == 0 || w_out == 0) {
        rleInit(M, h_out, w_out, 0);
        return;
    }
    if (R->m == 0 || R->h == 0 || R->w == 0) {
        rleZeros(M, h_out, w_out);
        return;
    }

    double H_inv[6];
    invert_affine(H, H_inv);

    double pp[2] = {(w_out - 1) * 0.5, (h_out - 1) * 0.5};
    double pp_old[2];
    transformAffine(pp, pp_old, H_inv);
    double pp_old_plus_x[2] = {pp_old[0] + 1, pp_old[1]};
    double pp_x[2];
    transformAffine(pp_old_plus_x, pp_x, H);
    pp_x[0] -= pp[0];
    pp_x[1] -= pp[1];
    double rot_angle = atan2(-pp_x[1], pp_x[0]);

    int k = int_remainder(round(rot_angle / (M_PI / 2)), 4);
    double H_rot[6];

    switch (k) {
        case 0:
            memcpy(H_rot, H, sizeof(double) * 6);
            break;
        case 1:
            rotate_affine(H, (double[]){pp[1], pp[1]}, k, H_rot);
            siz tmp;
            tmp = h_out;
            h_out = w_out;
            w_out = tmp;
            break;
        case 2:
            rotate_affine(H, pp, k, H_rot);
            break;
        case 3:
            rotate_affine(H, (double[]){pp[0], pp[0]}, k, H_rot);
            tmp = h_out;
            h_out = w_out;
            w_out = tmp;
            break;
        default:
            break;
    }

    double pp_y_y = _transformAffineY(pp_old[0], pp_old[1] + 1, H_rot);
    bool flip = pp_y_y < pp[1];
    if (flip) {
        // flipmat = np.array([[1, 0, 0], [0, -1, x - 1], [0, 0, 1]], np.float64)
        // homography = flipmat @ homography
        H_rot[3] *= -1;
        H_rot[4] *= -1;
        H_rot[5] = (h_out - 1) - H_rot[5];
    }

    RLE tmp1;
    rleWarpAffine1(R, &tmp1, h_out, H_rot);

    RLE tmp2;
    rleTranspose(&tmp1, &tmp2);
    rleFree(&tmp1);

    RLE tmp3;
    rleWarpAffine2(&tmp2, &tmp3, w_out, H_rot);
    rleFree(&tmp2);

    rleBackFlipRot(&tmp3, M, k, flip);
    rleFree(&tmp3);
}

static void rleWarpAffine1(const RLE *R, RLE *M, siz h_out, double *H) {
    if (h_out == 0) {
        rleInit(M, h_out, R->w + 1, 0);
        return;
    }
    if (R->m == 0 || R->h == 0 || R->w == 0) {
        rleZeros(M, h_out, R->w + 1);
        return;
    }

    RLE tmp;
    rleZeroPad(R, &tmp, 1, (uint[4]){0, 0, 0, 1});

    siz m = tmp.m;
    siz h = tmp.h;
    siz w_out = tmp.w + 1;
    uint *cnts = tmp.cnts;

    uint *cnts_out = rleInit(M, h_out, w_out, m);
    siz m_out = 0;
    int r = 0;
    int y_out_prev = h_out;
    int x_prev = -1;
    double H3_x_p_H5;

    for (siz i = 1; i < m; i += 2) {
        r += cnts[i-1];
        int cnt = cnts[i];
        int last = r + cnt - 1;
        int x = r / h;

        if (x != x_prev) {
            H3_x_p_H5 = H[3] * x + H[5];
        }
        int y_start = r % h;
        int y_end = last % h + 1;
        double raw_y_start_out = H[4] * y_start + H3_x_p_H5;
        double raw_y_end_out = H[4] * y_end + H3_x_p_H5;
        int y_start_out = intClip((int) round(raw_y_start_out), 0, h_out);
        int y_end_out = intClip((int) round(raw_y_end_out), 0, h_out);
        //int y_start_out = intClip((int) round(raw_y_start_out), 0, h_out);
        //int y_end_out = intClip((int) round(_transformY(x, y_end, H)), 0, h_out);

        if (y_start_out > y_end_out) {
            int tmp = y_start_out;
            y_start_out = y_end_out;
            y_end_out = tmp;
        }
        int cols = x - x_prev;
        int num_zeros = y_start_out + h_out * cols - y_out_prev;
        int num_ones = y_end_out - y_start_out;

        if (num_zeros < 0) {
            // we are supposed to go backwards to start the new run...
            // we won't do that, but instead at least reduce the number of 1s that we add
            // so it ends where it is supposed to.
            num_ones += num_zeros;
            if (num_ones <= 0) {
                // if even the end is supposed to go backwards, we skip this run
                // the result will not be correct, but we produce a valid RLE
                // going backwards to change the already produced runs would be inefficient
                // the caller is supposed to ensure that the transformation does not go backwards
                continue;
            }
            num_zeros = 0;
        }

        cnts_out[m_out++] = num_zeros; // run of 0s
        cnts_out[m_out++] = num_ones; // run of 1s
        y_out_prev = y_end_out;
        x_prev = x;
        r += cnt;
    }

    int cols = w_out - x_prev;
    cnts_out[m_out++] = h_out * cols - y_out_prev; // run of 0s, already padded
    M->m = m_out;
    rleEliminateZeroRuns(M);
    rleFree(&tmp);
}




static void rleWarpAffine2(const RLE *R, RLE *M, siz h_out, double *H) {
    siz h = R->h;
    siz w = R->w;
    siz m = R->m;
    siz w_out = w;

    if (h_out == 0 || w_out == 0) {
        rleInit(M, h_out, w_out, 0);
        return;
    }
    if (R->m == 0 || R->h == 0 || R->w == 0) {
        rleZeros(M, h_out, w_out);
        return;
    }


    uint *cnts = R->cnts;
    uint *cnts_out = rleInit(M, h_out, w_out, m);
    siz m_out = 0;
    int r = 0;
    int y_out_prev = h_out;
    int x_prev = -1;


    double A1_x_p_A3;

    double A1 = H[1]/H[4];
    double A2 = -H[3] * A1 + H[0];
    double A3 = -H[5] * A1 + H[2];

    for (siz i = 1; i < m; i += 2) {
        r += cnts[i-1];
        int cnt = cnts[i];
        int last = r + cnt - 1;
        int x = r / h;

        if (x != x_prev) {
            A1_x_p_A3 = x * A1 + A3;
        }

        int y_start = r % h;
        int y_end = last % h + 1;

        double raw_y_start_out = A2 * y_start + A1_x_p_A3;
        double raw_y_end_out = A2 * y_end + A1_x_p_A3;

        int y_start_out = intClip((int) round(raw_y_start_out), 0, h_out);
        int y_end_out = intClip((int) round(raw_y_end_out), 0, h_out);

        //int y_start_out = intClip((int) round(_transformX(x, y_start, a)), 0, h_out);
        //int y_end_out = intClip((int) round(_transformX(x, y_end, a)), 0, h_out);

        if (y_start_out > y_end_out) {
            int tmp = y_start_out;
            y_start_out = y_end_out;
            y_end_out = tmp;
        }

        int cols = x - x_prev;
        int num_zeros = h_out * cols + y_start_out - (int) y_out_prev;
        int num_ones = y_end_out - y_start_out;

        if (num_zeros < 0) {
            // we are supposed to go backwards to start the new run...
            // we won't do that, but instead at least reduce the number of 1s that we add
            // so it ends where it is supposed to.
            num_ones += num_zeros;
            if (num_ones <= 0) {
                // if even the end is supposed to go backwards, we skip this run
                // the result will not be correct, but we produce a valid RLE
                // going backwards to change the already produced runs would be inefficient
                // the caller is supposed to ensure that the transformation does not go backwards
                continue;
            }
            num_zeros = 0;
        }

        cnts_out[m_out++] = num_zeros; // run of 0s
        cnts_out[m_out++] = num_ones; // run of 1s
        y_out_prev = y_end_out;
        x_prev = x;
        r += cnt;
    }

    int cols = w_out - x_prev;
    cnts_out[m_out++] = h_out * cols - y_out_prev; // run of 0s
    M->m = m_out;
    rleEliminateZeroRuns(M);
}


static void rotate_affine(const double H[6], double anchor[2], int k, double H_new[6]) {
    // Equivalent to the following code in Python:
    // trans1 = np.array([[1, 0, -anchor[0]], [0, 1, -anchor[1]], [0, 0, 1]])
    // rot = Rotation.from_euler('z', k*np.pi/2).as_matrix()
    // trans2 = np.array([[1, 0, anchor[0]], [0, 1, anchor[1]], [0, 0, 1]])
    // return trans2 @ rot @ trans1 @ H
    switch (k) {
        case 0:
            H_new[0] = H[0];
            H_new[1] = H[1];
            H_new[2] = H[2];

            H_new[3] = H[3];
            H_new[4] = H[4];
            H_new[5] = H[5];
            break;

        case 1:
            H_new[0] = -H[3];
            H_new[1] = -H[4];
            H_new[2] = -H[5] + anchor[0] + anchor[1];

            H_new[3] = H[0];
            H_new[4] = H[1];
            H_new[5] = H[2] + anchor[1] - anchor[0];
            break;

        case 2:
            H_new[0] = -H[0];
            H_new[1] = -H[1];
            H_new[2] = -H[2] + 2 * anchor[0];

            H_new[3] = -H[3];
            H_new[4] = -H[4];
            H_new[5] = -H[5] + 2 * anchor[1];
            break;

        case 3:
            H_new[0] = H[3];
            H_new[1] = H[4];
            H_new[2] = H[5] + anchor[0] - anchor[1];

            H_new[3] = -H[0];
            H_new[4] = -H[1];
            H_new[5] = -H[2] + anchor[1] + anchor[0];
            break;

        default:
            break;
    }
}


void invert_affine(const double A[6], double A_inv[6]) {
    double det = A[0] * A[4] - A[1] * A[3];
    double inv_det = 1.0 / det;
    A_inv[0] = A[4] * inv_det;
    A_inv[1] = -A[1] * inv_det;
    A_inv[2] = (A[1] * A[5] - A[2] * A[4]) * inv_det;
    A_inv[3] = -A[3] * inv_det;
    A_inv[4] = A[0] * inv_det;
    A_inv[5] = (A[2] * A[3] - A[0] * A[5]) * inv_det;
}

static double _transformAffineY(double x, double y, double H[6]) {
    return H[3] * x + H[4] * y + H[5];
}