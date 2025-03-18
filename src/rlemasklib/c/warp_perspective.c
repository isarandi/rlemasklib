#include <math.h>
#include <string.h> // for memcpy
#include <stdbool.h> // for bool
#include "basics.h"
#include "pad_crop.h"
#include "misc.h"
#include "transpose_flip.h"
#include "minmax.h"
#include "boolfuncs.h"
#include "warp_common.h"
#include "warp_perspective.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

static void rleWarpPerspective1(const RLE *R, RLE *M, siz h_out, double *H);
static void rleWarpPerspective2(const RLE *R, RLE *M, siz h_out, double *H);

static double _transformY(double x, double y, double H[9]);
static double _transformX(double x, double y, double a[6]);

static void rotate_homography(const double H[9], double anchor[2], int k, double H_new[9]);

static void _rleVerticalBlur(const RLE *R, RLE *M);


//----------------------------------------------------------

void rleWarpPerspective(const RLE *R, RLE *M, siz h_out, siz w_out, double *H) {
    if (h_out == 0 || w_out == 0) {
        rleInit(M, h_out, w_out, 0);
        return;
    }
    if (R->m == 0 || R->h == 0 || R->w == 0) {
        rleZeros(M, h_out, w_out);
        return;
    }

    double H_inv[9];
    invert3x3(H, H_inv);

    double pp[2] = {(w_out - 1) * 0.5, (h_out - 1) * 0.5};
    double pp_old[2];
    transformPerspective(pp, pp_old, H_inv);
    double pp_old_plus_x[2] = {pp_old[0] + 1, pp_old[1]};
    double pp_x[2];
    transformPerspective(pp_old_plus_x, pp_x, H);
    pp_x[0] -= pp[0];
    pp_x[1] -= pp[1];
    double rot_angle = atan2(-pp_x[1], pp_x[0]);

    int k = int_remainder(round(rot_angle / (M_PI / 2)), 4);
    double H_rot[9];

    switch (k) {
        case 0:
            memcpy(H_rot, H, sizeof(double) * 9);
            break;
        case 1:
            rotate_homography(H, (double[]){pp[1], pp[1]}, k, H_rot);
            siz tmp;
            tmp = h_out;
            h_out = w_out;
            w_out = tmp;
            break;
        case 2:
            rotate_homography(H, pp, k, H_rot);
            break;
        case 3:
            rotate_homography(H, (double[]){pp[0], pp[0]}, k, H_rot);
            tmp = h_out;
            h_out = w_out;
            w_out = tmp;
            break;
        default:
            break;
    }



    double pp_y_y = _transformY(pp_old[0], pp_old[1] + 1, H_rot);
    bool flip = pp_y_y < pp[1];
    if (flip) {
        // flipmat = np.array([[1, 0, 0], [0, -1, x - 1], [0, 0, 1]], np.float64)
        // homography = flipmat @ homography
        H_rot[3] = (h_out - 1) * H_rot[6] - H_rot[3];
        H_rot[4] = (h_out - 1) * H_rot[7] - H_rot[4];
        H_rot[5] = (h_out - 1) * H_rot[8] - H_rot[5];
    }

    RLE tmp1;
    rleWarpPerspective1(R, &tmp1, h_out, H_rot);

    RLE tmp2;
    rleTranspose(&tmp1, &tmp2);
    rleFree(&tmp1);

    //RLE tmp2_blurred;
    //_rleVerticalBlur(&tmp2, &tmp2_blurred);
    //rleFree(&tmp2);

    RLE tmp3;
    rleWarpPerspective2(&tmp2, &tmp3, w_out, H_rot);
    rleFree(&tmp2);

    rleBackFlipRot(&tmp3, M, k, flip);
    rleFree(&tmp3);
}


static void rleWarpPerspective1(const RLE *R, RLE *M, siz h_out, double *H) {
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

     // we add one more empty column so that after transpose, all runs of 1s are in the same column
    siz w_out = tmp.w + 1;
    uint *cnts = tmp.cnts;

    uint *cnts_out = rleInit(M, h_out, w_out, m);
    siz m_out = 0;
    int r = 0;
    int y_out_prev = h_out;
    int x_prev = -1;
    double H3_x_p_H5;
    double H6_x_p_H8;

    for (siz i = 1; i < m; i += 2) {
        r += cnts[i-1];
        int cnt = cnts[i];
        int last = r + cnt - 1;
        int x = r / h;

        if (x != x_prev) {
            H3_x_p_H5 = H[3] * x + H[5];
            H6_x_p_H8 = H[6] * x + H[8];
        }
        int y_start = r % h;
        int y_end = last % h + 1;
        double raw_y_start_out = (H[4] * y_start + H3_x_p_H5) / (H[7] * y_start + H6_x_p_H8);
        double raw_y_end_out = (H[4] * y_end + H3_x_p_H5) / (H[7] * y_end + H6_x_p_H8);
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


static void rleWarpPerspective2(const RLE *R, RLE *M, siz h_out, double *H) {
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

    double a[6] = {
        H[0] * H[7] - H[6] * H[1],
        H[2] * H[7] - H[8] * H[1],
        H[3] * H[1] - H[0] * H[4],
        H[5] * H[1] - H[2] * H[4],
        H[3] * H[7] - H[6] * H[4],
        H[5] * H[7] - H[8] * H[4]
    };

    uint *cnts = R->cnts;
    uint *cnts_out = rleInit(M, h_out, w_out, m);
    siz m_out = 0;
    int r = 0;
    int y_out_prev = h_out;
    int x_prev = -1;
    double A0_x_p_A2;
    double A1_x_p_A3;

    for (siz i = 1; i < m; i += 2) {
        r += cnts[i-1];
        int cnt = cnts[i];
        int last = r + cnt - 1;
        int x = r / h;

        if (x != x_prev) {
            A0_x_p_A2 = x * a[0] + a[2];
            A1_x_p_A3 = x * a[1] + a[3];
        }

        int y_start = r % h;
        int y_end = last % h + 1;
        double raw_y_start_out = (A0_x_p_A2 * y_start + A1_x_p_A3) / (a[4] * y_start + a[5]);
        double raw_y_end_out = (A0_x_p_A2 * y_end + A1_x_p_A3) / (a[4] * y_end + a[5]);
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





static double _transformY(double x, double y, double H[9]) {
    return (H[3] * x + H[4] * y + H[5]) / (H[6] * x + H[7] * y + H[8]);
}

static double _transformX(double x, double y, double a[6]) {
    double A = y * a[0] + a[1];
    double B = y * a[2] + a[3];
    double C = y * a[4] + a[5];
    return (x * A + B) / C;
}


static void rotate_homography(const double H[9], double anchor[2], int k, double H_new[9]) {
    // Equivalent to the following code in Python:
    // trans1 = np.array([[1, 0, -anchor[0]], [0, 1, -anchor[1]], [0, 0, 1]])
    // rot = Rotation.from_euler('z', k*np.pi/2).as_matrix()
    // trans2 = np.array([[1, 0, anchor[0]], [0, 1, anchor[1]], [0, 0, 1]])
    // return trans2 @ rot @ trans1 @ H
    double u, v;
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
            u = anchor[0] + anchor[1];
            v = anchor[1] - anchor[0];

            H_new[0] = -H[3] + u * H[6];
            H_new[1] = -H[4] + u * H[7];
            H_new[2] = -H[5] + u * H[8];

            H_new[3] = H[0] + v * H[6];
            H_new[4] = H[1] + v * H[7];
            H_new[5] = H[2] + v * H[8];
            break;

        case 2:
            u = 2 * anchor[0];
            v = 2 * anchor[1];

            H_new[0] = -H[0] + u * H[6];
            H_new[1] = -H[1] + u * H[7];
            H_new[2] = -H[2] + u * H[8];

            H_new[3] = -H[3] + v * H[6];
            H_new[4] = -H[4] + v * H[7];
            H_new[5] = -H[5] + v * H[8];
            break;

        case 3:
            u = anchor[0] - anchor[1];
            v = anchor[1] + anchor[0];

            H_new[0] = H[3] + u * H[6];
            H_new[1] = H[4] + u * H[7];
            H_new[2] = H[5] + u * H[8];

            H_new[3] = -H[0] + v * H[6];
            H_new[4] = -H[1] + v * H[7];
            H_new[5] = -H[2] + v * H[8];
            break;

        default:
            break;
    }

    H_new[6] = H[6];
    H_new[7] = H[7];
    H_new[8] = H[8];
}


static void _rleVerticalBlur(const RLE *R, RLE *M) {
    RLE tmp1;
    rleCopy(R, &tmp1);
    tmp1.cnts[tmp1.m - 1]++; // stretch the last
    tmp1.cnts[tmp1.cnts[0] == 0 ? 1 : 0]--; // shrink the first

    RLE tmp2;
    rleCopy(R, &tmp2);
    tmp2.cnts[0]++; // stretch the first
    tmp2.cnts[tmp2.m - 1]--; // shrink the last

    RLE *Rs[3] = {R, &tmp1, &tmp2};
    rleMergeAtLeast2(Rs, M, 3, 2);
    rleFree(&tmp1);
    rleFree(&tmp2);
}
