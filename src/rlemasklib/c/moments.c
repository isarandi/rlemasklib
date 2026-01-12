#include <stdlib.h> // for malloc
#include <math.h>    // for pow, sqrt
#include "basics.h"
#include "minmax.h"
#include "moments.h"

// Faulhaber's formulas for sums of powers over consecutive integers
// For integers from s to s+n-1 (inclusive), where n is the count

static inline double sum_of_integers(double s, double n) {
    // Σ_{k=s}^{s+n-1} k = n*s + n*(n-1)/2
    double nm1 = n - 1;
    return n * (s + nm1 * 0.5);
}

static inline double sum_of_squares(double s, double n) {
    // Σ_{k=s}^{s+n-1} k² = n*s² + s*n*(n-1) + n*(n-1)*(2n-1)/6
    double nm1 = n - 1;
    double n_nm1 = n * nm1;
    double s2 = s * s;
    return n * s2 + s * n_nm1 + n_nm1 * (2 * n - 1) / 6;
}

static inline double sum_of_cubes(double s, double n) {
    // Σ_{k=s}^{s+n-1} k³ = n*s³ + 3s²*n*(n-1)/2 + s*n*(n-1)*(2n-1)/2 + n²*(n-1)²/4
    double nm1 = n - 1;
    double n_nm1 = n * nm1;
    double s2 = s * s;
    double s3 = s2 * s;
    return n * s3
         + 1.5 * s2 * n_nm1
         + 0.5 * s * n_nm1 * (2 * n - 1)
         + 0.25 * n_nm1 * n_nm1;
}

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

// Accumulate raw moments for a single column segment
// x is constant, y ranges from y_start to y_start + cnt - 1
static inline void accumulate_segment_moments(
    double x, double y_start, double cnt,
    double *m00, double *m10, double *m01,
    double *m20, double *m11, double *m02,
    double *m30, double *m21, double *m12, double *m03
) {
    double x2 = x * x;
    double x3 = x2 * x;

    double sum_y = sum_of_integers(y_start, cnt);
    double sum_y2 = sum_of_squares(y_start, cnt);
    double sum_y3 = sum_of_cubes(y_start, cnt);

    *m00 += cnt;
    *m10 += x * cnt;
    *m01 += sum_y;
    *m20 += x2 * cnt;
    *m11 += x * sum_y;
    *m02 += sum_y2;
    *m30 += x3 * cnt;
    *m21 += x2 * sum_y;
    *m12 += x * sum_y2;
    *m03 += sum_y3;
}

void rleRawMoments(const RLE *R, double *moments) {
    siz m = R->m;
    uint h = R->h;

    // Initialize raw moments: m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
    double m00 = 0, m10 = 0, m01 = 0;
    double m20 = 0, m11 = 0, m02 = 0;
    double m30 = 0, m21 = 0, m12 = 0, m03 = 0;

    uint pos = 0;
    for (siz j = 1; j < m; j += 2) {
        pos += R->cnts[j - 1];
        uint start_row = pos % h;
        uint start_col = pos / h;
        uint cnt = R->cnts[j];
        pos += cnt;

        // First partial column (may be full or partial)
        uint cnt1 = uintMin(cnt, h - start_row);
        accumulate_segment_moments(
            start_col, start_row, cnt1,
            &m00, &m10, &m01, &m20, &m11, &m02, &m30, &m21, &m12, &m03);

        if (cnt1 == cnt) continue;

        // Full middle columns (O(1) per column group)
        uint num_full_cols = (cnt - cnt1) / h;
        if (num_full_cols > 0) {
            uint first_full_col = start_col + 1;
            double n = num_full_cols;
            double sum_x = sum_of_integers(first_full_col, n);
            double sum_x2 = sum_of_squares(first_full_col, n);
            double sum_x3 = sum_of_cubes(first_full_col, n);

            double sum_y_col = sum_of_integers(0, h);
            double sum_y2_col = sum_of_squares(0, h);
            double sum_y3_col = sum_of_cubes(0, h);

            double total_pixels = n * h;

            m00 += total_pixels;
            m10 += sum_x * h;
            m01 += n * sum_y_col;
            m20 += sum_x2 * h;
            m11 += sum_x * sum_y_col;
            m02 += n * sum_y2_col;
            m30 += sum_x3 * h;
            m21 += sum_x2 * sum_y_col;
            m12 += sum_x * sum_y2_col;
            m03 += n * sum_y3_col;
        }

        // Last partial column
        uint cnt3 = cnt - cnt1 - num_full_cols * h;
        if (cnt3 > 0) {
            uint last_col = start_col + num_full_cols + 1;
            accumulate_segment_moments(
                last_col, 0, cnt3,
                &m00, &m10, &m01, &m20, &m11, &m02, &m30, &m21, &m12, &m03);
        }
    }

    moments[0] = m00;
    moments[1] = m10;
    moments[2] = m01;
    moments[3] = m20;
    moments[4] = m11;
    moments[5] = m02;
    moments[6] = m30;
    moments[7] = m21;
    moments[8] = m12;
    moments[9] = m03;
}

// Moment indices for the 24-element array returned by rleMoments
enum {
    M_m00, M_m10, M_m01, M_m20, M_m11, M_m02, M_m30, M_m21, M_m12, M_m03,
    M_mu20, M_mu11, M_mu02, M_mu30, M_mu21, M_mu12, M_mu03,
    M_nu20, M_nu11, M_nu02, M_nu30, M_nu21, M_nu12, M_nu03
};

void rleMoments(const RLE *R, double *out) {
    // Get raw moments
    double raw[10];
    rleRawMoments(R, raw);

    double m00 = raw[0], m10 = raw[1], m01 = raw[2];
    double m20 = raw[3], m11 = raw[4], m02 = raw[5];
    double m30 = raw[6], m21 = raw[7], m12 = raw[8], m03 = raw[9];

    // Copy raw moments to output
    for (int i = 0; i < 10; i++) out[i] = raw[i];

    // Handle empty mask
    if (m00 == 0) {
        for (int i = 10; i < 24; i++) out[i] = 0;
        return;
    }

    // Central moments
    double x_bar = m10 / m00;
    double y_bar = m01 / m00;
    double x_bar2 = x_bar * x_bar;
    double y_bar2 = y_bar * y_bar;

    double mu20 = m20 - x_bar * m10;
    double mu02 = m02 - y_bar * m01;
    double mu11 = m11 - x_bar * m01;
    double mu30 = m30 - 3 * x_bar * m20 + 2 * x_bar2 * m10;
    double mu03 = m03 - 3 * y_bar * m02 + 2 * y_bar2 * m01;
    double mu21 = m21 - 2 * x_bar * m11 - y_bar * m20 + 2 * x_bar2 * m01;
    double mu12 = m12 - 2 * y_bar * m11 - x_bar * m02 + 2 * y_bar2 * m10;

    out[M_mu20] = mu20; out[M_mu11] = mu11; out[M_mu02] = mu02;
    out[M_mu30] = mu30; out[M_mu21] = mu21; out[M_mu12] = mu12; out[M_mu03] = mu03;

    // Normalized central moments
    double m00_2 = m00 * m00;
    double m00_2p5 = m00_2 * sqrt(m00);

    out[M_nu20] = mu20 / m00_2;
    out[M_nu11] = mu11 / m00_2;
    out[M_nu02] = mu02 / m00_2;
    out[M_nu30] = mu30 / m00_2p5;
    out[M_nu21] = mu21 / m00_2p5;
    out[M_nu12] = mu12 / m00_2p5;
    out[M_nu03] = mu03 / m00_2p5;
}

void rleHuMoments(const RLE *R, double *hu) {
    double m[24];
    rleMoments(R, m);

    if (m[M_m00] == 0) {
        for (int i = 0; i < 7; i++) hu[i] = 0;
        return;
    }

    double nu20 = m[M_nu20], nu11 = m[M_nu11], nu02 = m[M_nu02];
    double nu30 = m[M_nu30], nu21 = m[M_nu21], nu12 = m[M_nu12], nu03 = m[M_nu03];

    // Hu moments with CSE
    double nu20_p_02 = nu20 + nu02;
    double nu20_m_02 = nu20 - nu02;
    double nu30_p_12 = nu30 + nu12;
    double nu30_m_3x12 = nu30 - 3 * nu12;
    double nu21_p_03 = nu21 + nu03;
    double nu3x21_m_03 = 3 * nu21 - nu03;

    double sq_30p12 = nu30_p_12 * nu30_p_12;
    double sq_21p03 = nu21_p_03 * nu21_p_03;
    double t1 = sq_30p12 - 3 * sq_21p03;
    double t2 = 3 * sq_30p12 - sq_21p03;

    hu[0] = nu20_p_02;
    hu[1] = nu20_m_02 * nu20_m_02 + 4 * nu11 * nu11;
    hu[2] = nu30_m_3x12 * nu30_m_3x12 + nu3x21_m_03 * nu3x21_m_03;
    hu[3] = sq_30p12 + sq_21p03;
    hu[4] = nu30_m_3x12 * nu30_p_12 * t1 + nu3x21_m_03 * nu21_p_03 * t2;
    hu[5] = nu20_m_02 * (sq_30p12 - sq_21p03) + 4 * nu11 * nu30_p_12 * nu21_p_03;
    hu[6] = nu3x21_m_03 * nu30_p_12 * t1 - nu30_m_3x12 * nu21_p_03 * t2;
}