#include <math.h>
#include <stddef.h>
#include <string.h> // for memcpy
#include <stdbool.h> // for bool
#include "basics.h"
#include "minmax.h"
#include "transpose_flip.h"
#include "warp_common.h"
#include "warp_distorted.h"

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

static void prepareCameraChange(
    const struct Camera *old_camera, const struct Camera *new_camera, struct CameraChange *cc);
static void rotateCameraParams(
    const struct Camera *c1, struct Camera *c2, siz h, siz w, int k);
static void rleWarpDistorted1(const RLE *R, RLE *M, siz h_out, struct CameraChange *cc);
static void rleWarpDistorted2(const RLE *R, RLE *M, siz h_out, struct CameraChange *cc);
static void _distortionFormulaParts(
    double x, double y, const double d[12], double *a, double *b, double *cx, double *cy);
static void _distortionFormulaPartsForNewton(
    double x, double y, const double d[12], double *inv_j00, double *inv_j11, double *pxn_hat, double *pyn_hat);
static float interp1(float xi, const float *x, const float *y, siz n);
static void _clipToValid(double p[2], struct ValidRegion *valid, bool distorted);
static bool _hasNonZero(const double d[12]);
static void _undistortPoint(const double p_[2], double pu[2], const double d[12], struct ValidRegion *valid);
static void _distortPoint(const double pu_[2], double p[2], const double d[12], struct ValidRegion *valid);
static double _transformDistortedY(double x, double y, struct CameraChange *cc);
static double _transformDistortedX(double y, double x, struct CameraChange *cc);
static void _transformDistorted(const double inp[2], double outp[2], struct CameraChange *cc);
static void _matmul_A_BT_3x3(const double A[9], const double B[9], double C[9]);
//----------------------------------------------------------

void rleWarpDistorted(
    const RLE *R, RLE *M, siz h_out, siz w_out, struct Camera* old_camera,
    struct Camera* new_camera) {

    if (h_out == 0 || w_out == 0) {
        rleInit(M, h_out, w_out, 0);
        return;
    }
    if (R->m == 0 || R->h == 0 || R->w == 0) {
        rleZeros(M, h_out, w_out);
        return;
    }

    struct CameraChange cc;
    prepareCameraChange(old_camera, new_camera, &cc);

    struct CameraChange cc_inv;
    prepareCameraChange(new_camera, old_camera, &cc_inv);

    double pp[2] = {(w_out - 1) * 0.5, (h_out - 1) * 0.5};
    double pp_old[2];
    _transformDistorted(pp, pp_old, &cc_inv);

    double pp_old_plus_x[2] = {pp_old[0] + 1, pp_old[1]};
    double pp_x[2];
    _transformDistorted(pp_old_plus_x, pp_x, &cc);
    pp_x[0] -= pp[0];
    pp_x[1] -= pp[1];
    double rot_angle = atan2(-pp_x[1], pp_x[0]);

    int k = int_remainder(round(rot_angle / (M_PI / 2)), 4);

    struct Camera new_camera_rot;
    rotateCameraParams(new_camera, &new_camera_rot, h_out, w_out, k);
    if (k % 2 == 1) {
        siz tmp = h_out;
        h_out = w_out;
        w_out = tmp;
    }
    prepareCameraChange(old_camera, &new_camera_rot, &cc);


    double pp_y_y = _transformDistortedY(pp_old[0], pp_old[1] + 1, &cc);
    bool flip = pp_y_y < pp[1];
    if (flip) {
        // vertical flip (Y)
        cc.H[3] *= -1;
        cc.H[4] *= -1;
        cc.H[5] *= -1;
        cc.K2[5] = (h_out - 1) - cc.K2[5];
        cc.d2[2] *= -1;
        cc.d2[10] *= -1;
        cc.d2[11] *= -1;
    }

    RLE tmp1;
    rleWarpDistorted1(R, &tmp1, h_out, &cc);

    RLE tmp2;
    rleTranspose(&tmp1, &tmp2);
    rleFree(&tmp1);

    RLE tmp3;
    rleWarpDistorted2(&tmp2, &tmp3, w_out, &cc);
    rleFree(&tmp2);
    //rleMoveTo(&tmp3, M);
    //rleFree(&tmp3);

    rleBackFlipRot(&tmp3, M, k, flip);
    rleFree(&tmp3);
}


static void rleWarpDistorted1(const RLE *R, RLE *M, siz h_out, struct CameraChange *cc) {
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

    for (siz i = 1; i < m; i += 2) {
        r += cnts[i-1];
        int cnt = cnts[i];
        int last = r + cnt - 1;
        int x = r / h;
        int y_start = r % h;
        int y_end = last % h + 1;

        int y_start_out = intClip((int) round(_transformDistortedY(x, y_start, cc)), 0, h_out);
        int y_end_out = intClip((int) round(_transformDistortedY(x, y_end, cc)), 0, h_out);

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
    cnts_out[m_out++] = h_out * cols - y_out_prev; // run of 0s, already padded
    M->m = m_out;
    rleEliminateZeroRuns(M);
    rleFree(&tmp);
}

static void rleWarpDistorted2(const RLE *R, RLE *M, siz h_out, struct CameraChange *cc) {
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

    for (siz i = 1; i < m; i += 2) {
        r += cnts[i-1];
        int cnt = cnts[i];
        int last = r + cnt - 1;
        int x = r / h;
        int y_start = r % h;
        int y_end = last % h + 1;

        int y_start_out = intClip((int) round(_transformDistortedX(x, y_start, cc)), 0, h_out);
        int y_end_out = intClip((int) round(_transformDistortedX(x, y_end, cc)), 0, h_out);

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


static void _distortionFormulaParts(
    double x, double y, const double d[12], double *a, double *b, double *cx, double *cy) {

    double r2 = x * x + y * y;
    *a = ((((d[4] * r2 + d[1]) * r2 + d[0]) * r2 + 1) /
          (((d[7] * r2 + d[6]) * r2 + d[5]) * r2 + 1));
    *b = 2 * (x * d[3] + y * d[2]);
    *cx = (d[9] * r2 + d[3] + d[8]) * r2;
    *cy = (d[11] * r2 + d[2] + d[10]) * r2;
}

static void _distortionFormulaPartsForNewton(
    double x, double y, const double d[12], double *inv_j00, double *inv_j11, double *pxn_hat, double *pyn_hat) {
    double r2 = x * x + y * y;
    double _2_x = 2 * x;
    double _2_y = 2 * y;
    double k9_r2 = d[9] * r2;
    double k11_r2 = d[11] * r2;
    double k4_r2 = d[4] * r2;
    double k7_r2 = d[7] * r2;
    double x2 = d[3] + d[8] + k9_r2;
    double x16 = d[2] + d[10] + k11_r2;
    double x6 = d[1] + k4_r2;
    double x7 = d[0] + r2 * x6;
    double x10 = d[6] + k7_r2;
    double x11 = d[5] + r2 * x10;
    double x13 = 1 / (r2 * x11 + 1);
    double x29 = x13 * (r2 * x7 + 1);
    double x14 = _2_x * d[3] + _2_y * d[2] + x29;
    double x26 = x13 * x13 * ((r2 * (k4_r2 + x6) + x7) - x29 * x29 * (r2 * (k7_r2 + x10) + x11));
    double x19 = x * x26 + d[3];
    double x21 = y * x26 + d[2];
    double x27 = k9_r2 + x2;
    double x28 = k11_r2 + x16;
    double pnx_hat = x * x14 + r2 * x2;
    double pny_hat = y * x14 + r2 * x16;
    double j00 = _2_x * (x19 + x27) + x14;
    double j11 = _2_y * (x21 + x28) + x14;
    double j01 = _2_x * x21 + _2_y * x27;
    double j10 = _2_y * x19 + _2_x * x28;
    double j01_times_j10 = j01 * j10;
    double det = j00 * j11 - j01_times_j10;
    double lambda_ = 5e-1;
    if (fabs(det) < 0.05) {
        j00 += lambda_;
        j11 += lambda_;
        det = j00 * j11 - j01_times_j10;
    }
    double inv_det = 1 / det;
    *inv_j11 = inv_det * j00;
    *inv_j00 = inv_det * j11;
    *pxn_hat = pnx_hat;
    *pyn_hat = pny_hat;
}


static float interp1(float xi, const float *x, const float *y, siz n) {
    if (xi <= x[0]) {
        return y[0];
    }
    if (xi >= x[n - 1]) {
        return y[n - 1];
    }
    siz low = 0;
    siz high = n - 1;

    // Binary search to find the interval [low, low + 1] containing xi
    while (low < high - 1) {  // Ensure low and high differ by at least 1
        siz mid = (low + high) / 2;
        if (xi < x[mid]) {
            high = mid;
        } else {
            low = mid;
        }
    }
    // Perform linear interpolation within the interval [low, low + 1]
    float t = (xi - x[low]) / (x[low + 1] - x[low]);
    return y[low] * (1 - t) + y[low + 1] * t;
}



static void _clipToValid(double p[2], struct ValidRegion *valid, bool distorted) {
    float r2_min = distorted ? valid->rd2_min : valid->ru2_min;
    float *rs = distorted ? valid->rd : valid->ru;
    float *ts = distorted ? valid->td : valid->tu;

    double r2 = p[0] * p[0] + p[1] * p[1];
    if (r2 > r2_min) {
        double t = atan2(p[1], p[0]);
        double r_interp = interp1(t, ts, rs, valid->n);
        double r2_interp = r_interp*r_interp;
        if (r2 > r2_interp) {
            double scale = r_interp / sqrt(r2);
            p[0] *= scale;
            p[1] *= scale;
        }
    }
}

static bool _hasNonZero(const double d[12]) {
    if (d == NULL) {
        return false;
    }
    for (int i = 0; i < 12; i++) {
        if (d[i] != 0) {
            return true;
        }
    }
    return false;
}


static void _undistortPoint(const double p_[2], double pu[2], const double d[12], struct ValidRegion *valid) {
    if (!_hasNonZero(d)) {
        pu[0] = p_[0];
        pu[1] = p_[1];
        return;
    }
    double p[2] = {p_[0], p_[1]};
    _clipToValid(p, valid, true);
    // Initialize undistorted point at distorted point
    pu[0] = p[0];
    pu[1] = p[1];

    _clipToValid(pu, valid, false);
    // fixed-point iteration
    for (int i = 0; i < 5; i++) {
        double a, b, cx, cy;
        _distortionFormulaParts(pu[0], pu[1], d, &a, &b, &cx, &cy);
        pu[0] = (p[0] - cx - pu[0] * b) / a;
        pu[1] = (p[1] - cy - pu[1] * b) / a;
    }
    _clipToValid(pu, valid, false);
    // Newton iteration
    for (int i=0; i<2; ++i) {
        double inv_j00, inv_j11, pxn_hat, pyn_hat;
        _distortionFormulaPartsForNewton(pu[0], pu[1], d, &inv_j00, &inv_j11, &pxn_hat, &pyn_hat);
        pu[0] += inv_j00 * (p[0] - pxn_hat);
        pu[1] += inv_j11 * (p[1] - pyn_hat);
    }
}


static void _distortPoint(const double pu_[2], double p[2], const double d[12], struct ValidRegion *valid) {
    if (!_hasNonZero(d)) {
        p[0] = pu_[0];
        p[1] = pu_[1];
        return;
    }
    double pu[2] = {pu_[0], pu_[1]};
    _clipToValid(pu, valid, false);
    double a, b, cx, cy;
    _distortionFormulaParts(pu[0], pu[1], d, &a, &b, &cx, &cy);
    p[0] = pu[0] * (a + b) + cx;
    p[1] = pu[1] * (a + b) + cy;
}


static double _transformDistortedY(double x, double y, struct CameraChange *cc) {
    // TODO: if the distortion is zero, we can simplify this by merging Ks into H
    double p_old[2] = {x, y};
    double pn_old[2];
    transformAffine(p_old, pn_old, cc->K1_inv);
    double pun_old[2];
    _undistortPoint(pn_old, pun_old, cc->d1, &cc->valid1);
    double pun_new[2];
    transformPerspective(pun_old, pun_new, cc->H);
    double pn_new[2];
    _distortPoint(pun_new, pn_new, cc->d2, &cc->valid2);
    double p_new[2];
    transformAffine(pn_new, p_new, cc->K2);
    return p_new[1];
}

static double _transformDistortedX(double y, double x, struct CameraChange *cc) {
    double px_old = x;
    double py_new = y;

    double pxn_old = px_old * cc->K1_inv[0] + cc->K1_inv[2];
    double pyn_old;

    double pxn_new;
    double pyn_new = (py_new - cc->K2[5]) / cc->K2[4];

    double puyn_old;
    double puxn_old = pxn_old;

    double puxn_new;
    double puyn_new = pyn_new;

    double *H = cc->H;

    for (int i=0; i<5; ++i) {
        double A_ = -H[6] * puyn_new + H[3];
        double B_ = -H[8] * puyn_new + H[5];
        double C_ = H[7] * puyn_new - H[4];
        puyn_old = (A_ * puxn_old + B_) / C_;
        double a, b, cx, cy;
        _distortionFormulaParts(puxn_old, puyn_old, cc->d1, &a, &b, &cx, &cy);
        if (cc->K1_inv[1] !=0) {
            pyn_old = puyn_old * (a + b) + cy;
            double py_old = (pyn_old - cc->K1_inv[5]) / cc->K1_inv[4];
            pxn_old = px_old * cc->K1_inv[0] + py_old * cc->K1_inv[1] + cc->K1_inv[2];
        }
        puxn_old = (pxn_old - cx - puxn_old * b) / a;

        puxn_new = ((puxn_old * H[0] + puyn_old * H[1] + H[2]) /
                    (puxn_old * H[6] + puyn_old * H[7] + H[8]));
        _distortionFormulaParts(puxn_new, puyn_new, cc->d2, &a, &b, &cx, &cy);
        puyn_new = (pyn_new - cy - puyn_new * b) / a;
    }

    for (int i=0; i<2; ++i) {
        double A_ = -H[6] * puyn_new + H[3];
        double B_ = -H[8] * puyn_new + H[5];
        double C_ = H[7] * puyn_new - H[4];
        puyn_old = (A_ * puxn_old + B_) / C_;
        double inv_j00, inv_j11, pxn_old_hat;
        _distortionFormulaPartsForNewton(
            puxn_old, puyn_old, cc->d1, &inv_j00, &inv_j11, &pxn_old_hat, &pyn_old);
        if (cc->K1_inv[1] !=0) {
            double py_old = (pyn_old - cc->K1_inv[5]) / cc->K1_inv[4];
            pxn_old = px_old * cc->K1_inv[0] + py_old * cc->K1_inv[1] + cc->K1_inv[2];
        }
        puxn_old += inv_j00 * (pxn_old - pxn_old_hat);

        puxn_new = ((puxn_old * H[0] + puyn_old * H[1] + H[2]) /
                    (puxn_old * H[6] + puyn_old * H[7] + H[8]));
        double pyn_new_hat;
        _distortionFormulaPartsForNewton(
            puxn_new, puyn_new, cc->d2, &inv_j00, &inv_j11, &pxn_new, &pyn_new_hat);
        puyn_new += inv_j11 * (pyn_new - pyn_new_hat);
    }
    double px_new = pxn_new * cc->K2[0] + pyn_new * cc->K2[1] + cc->K2[2];
    return px_new;
}

static void _transformDistorted(const double inp[2], double outp[2], struct CameraChange *cc) {
    double pn_old[2];
    transformAffine(inp, pn_old, cc->K1_inv);

    // undo old distortion
    double pun_old[2];
    _undistortPoint(pn_old, pun_old, cc->d1, &cc->valid1);

    // apply homography (rotation)
    double pun_new[2];
    transformPerspective(pun_old, pun_new, cc->H);

    // apply new distortion
    double pn_new[2];
    _distortPoint(pun_new, pn_new, cc->d2, &cc->valid2);

    // apply new intrinsics
    transformAffine(pn_new, outp, cc->K2);
}


static void prepareCameraChange(
    const struct Camera *old_camera, const struct Camera *new_camera, struct CameraChange *cc) {
    // TODO: if the distortion is zero, we can simplify this by merging Ks into H
    cc->K1_inv[0] = 1.0 / old_camera->f[0];
    cc->K1_inv[4] = 1.0 / old_camera->f[1];
    cc->K1_inv[1] = -old_camera->s * cc->K1_inv[0] * cc->K1_inv[4];
    cc->K1_inv[2] = -cc->K1_inv[1] * old_camera->c[1] - old_camera->c[0] * cc->K1_inv[0];
    cc->K1_inv[3] = 0;
    cc->K1_inv[5] = -old_camera->c[1] * cc->K1_inv[4];

    _matmul_A_BT_3x3(new_camera->R, old_camera->R, cc->H);
    memcpy(cc->d1, old_camera->d, 12 * sizeof(double));
    memcpy(cc->d2, new_camera->d, 12 * sizeof(double));

    cc->K2[0] = new_camera->f[0];
    cc->K2[1] = new_camera->s;
    cc->K2[2] = new_camera->c[0];
    cc->K2[3] = 0;
    cc->K2[4] = new_camera->f[1];
    cc->K2[5] = new_camera->c[1];

    cc->valid1 = old_camera->valid;
    cc->valid2 = new_camera->valid;
}

static void rotateCameraParams(
    const struct Camera *c1, struct Camera *c2, siz h, siz w, int k) {

    double ox = (w - 1) / 2;
    double oy = (h - 1) / 2;
    double cx = c1->c[0];
    double cy = c1->c[1];
    double fx = c1->f[0];
    double fy = c1->f[1];

    k = int_remainder(k, 4);
    switch (k) {
        case 0:
            memcpy(c2, c1, sizeof(struct Camera));
            return;
        case 1:
            c2->f[0] = fy;
            c2->f[1] = fx;

            c2->c[0] = -cy + oy + ox;
            c2->c[1] = cx - ox + oy;

            c2->R[0] = -c1->R[3];
            c2->R[1] = -c1->R[4];
            c2->R[2] = -c1->R[5];

            c2->R[3] = c1->R[0];
            c2->R[4] = c1->R[1];
            c2->R[5] = c1->R[2];

            c2->d[2] = c1->d[3];
            c2->d[3] = -c1->d[2];
            c2->d[8] = -c1->d[9];
            c2->d[9] = c1->d[8];
            c2->d[10] = -c1->d[11];
            c2->d[11] = c1->d[10];
            break;
        case 2:
            c2->f[0] = fx;
            c2->f[1] = fy;

            c2->c[0] = -cx + 2 * ox;
            c2->c[1] = -cy + 2 * oy;

            c2->R[0] = -c1->R[0];
            c2->R[1] = -c1->R[1];
            c2->R[2] = -c1->R[2];

            c2->R[3] = -c1->R[3];
            c2->R[4] = -c1->R[4];
            c2->R[5] = -c1->R[5];

            c2->d[2] = -c1->d[2];
            c2->d[3] = -c1->d[3];
            c2->d[8] = -c1->d[8];
            c2->d[9] = -c1->d[9];
            c2->d[10] = -c1->d[10];
            c2->d[11] = -c1->d[11];
            break;
        case 3:
            c2->f[0] = fy;
            c2->f[1] = fx;

            c2->c[0] = cy - oy + ox;
            c2->c[1] = -cx + ox + oy;

            c2->R[0] = c1->R[3];
            c2->R[1] = c1->R[4];
            c2->R[2] = c1->R[5];

            c2->R[3] = -c1->R[0];
            c2->R[4] = -c1->R[1];
            c2->R[5] = -c1->R[2];

            c2->d[2] = -c1->d[3];
            c2->d[3] = c1->d[2];
            c2->d[8] = c1->d[9];
            c2->d[9] = -c1->d[8];
            c2->d[10] = c1->d[11];
            c2->d[11] = -c1->d[10];
            break;
        default:
            break;
    }


    c2->R[6] = c1->R[6];
    c2->R[7] = c1->R[7];
    c2->R[8] = c1->R[8];
    c2->d[0] = c1->d[0];
    c2->d[1] = c1->d[1];
    c2->d[4] = c1->d[4];
    c2->d[5] = c1->d[5];
    c2->d[6] = c1->d[6];
    c2->d[7] = c1->d[7];
}

static void _matmul_A_BT_3x3(const double A[9], const double B[9], double C[9]) {
    // C = A @ B.T
    C[0] = A[0] * B[0] + A[1] * B[1] + A[2] * B[2];
    C[1] = A[0] * B[3] + A[1] * B[4] + A[2] * B[5];
    C[2] = A[0] * B[6] + A[1] * B[7] + A[2] * B[8];
    C[3] = A[3] * B[0] + A[4] * B[1] + A[5] * B[2];
    C[4] = A[3] * B[3] + A[4] * B[4] + A[5] * B[5];
    C[5] = A[3] * B[6] + A[4] * B[7] + A[5] * B[8];
    C[6] = A[6] * B[0] + A[7] * B[1] + A[8] * B[2];
    C[7] = A[6] * B[3] + A[7] * B[4] + A[8] * B[5];
    C[8] = A[6] * B[6] + A[7] * B[7] + A[8] * B[8];
}