#include <stdlib.h> // for malloc, free
#include <stdbool.h> // for bool
#include <math.h> // for fmax, fmin
#include "basics.h"
#include "minmax.h"
#include "shapes.h"

static int _uintCompare(const void *a, const void *b);

void rleToBbox(const RLE *R, BB bb, siz n) {
    siz i;
    for (i = 0; i < n; i++) {
        const RLE* R_ = &R[i];
        uint h = (uint) R_->h;
        uint w = (uint) R_->w;
        siz m = (R_->m / 2) * 2;
        uint xs = w;
        uint ys = h;
        uint xe = 0;
        uint ye = 0;
        uint cc = 0;
        if (m == 0) {
            bb[4 * i + 0] = bb[4 * i + 1] = bb[4 * i + 2] = bb[4 * i + 3] = 0;
            continue;
        }
        uint xp; // x previous
        for (siz j = 0; j < m; j++) {
            cc += R_->cnts[j];
            uint t = cc - j % 2;
            uint y = t % h;
            uint x = t / h;
            if (j % 2 == 0) {
                xp = x;
            } else if (xp < x) {
                ys = 0;
                ye = h - 1;
            }
            xs = uintMin(xs, x);
            xe = uintMax(xe, x);
            ys = uintMin(ys, y);
            ye = uintMax(ye, y);

            if (ys == 0 && ye == h - 1) {
                // The bounding box covers the entire height
                if (R_->m % 2 == 0) {
                    // if the number of runs is even, the last run is a run of 1s
                    xe = w - 1;
                } else {
                    // if the number of runs is odd, the last run is a run of 0s
                    xe = w - 1 - R_->cnts[R_->m - 1] / h;
                }
                break;
            }
        }
        bb[4 * i + 0] = xs;
        bb[4 * i + 2] = xe - xs + 1;
        bb[4 * i + 1] = ys;
        bb[4 * i + 3] = ye - ys + 1;
    }
}

void rleToUintBbox(const RLE *R, uint *bb) {
    uint h = (uint) R->h;
    uint w = (uint) R->w;
    siz m = (R->m / 2) * 2;
    uint xs = w;
    uint ys = h;
    uint xe = 0;
    uint ye = 0;
    uint cc = 0;
    if (m <= 1) {
        bb[0] = bb[1] = bb[2] = bb[3] = 0;
        return;
    }
    uint xp; // x previous
    for (siz j = 0; j < m; j++) {
        cc += R->cnts[j];
        uint t = cc - j % 2;
        uint y = t % h;
        uint x = t / h;
        if (j % 2 == 0) {
            xp = x;
        } else if (xp < x) {
            ys = 0;
            ye = h - 1;
        }
        xs = uintMin(xs, x);
        xe = uintMax(xe, x);
        ys = uintMin(ys, y);
        ye = uintMax(ye, y);

        if (ys == 0 && ye == h - 1) {
            // The bounding box covers the entire height
            if (R->m % 2 == 0) {
                // if the number of runs is even, the last run is a run of 1s
                xe = w - 1;
            } else {
                // if the number of runs is odd, the last run is a run of 0s
                xe = w - 1 - R->cnts[R->m - 1] / h;
            }
            break;
        }
    }
    bb[0] = xs;
    bb[2] = xe - xs + 1;
    bb[1] = ys;
    bb[3] = ye - ys + 1;
}

void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n) {
    for (siz i = 0; i < n; i++) {
        double xs_ = fmax(0, fmin(w, bb[4 * i + 0]));
        double ys_ = fmax(0, fmin(h, bb[4 * i + 1]));
        double xe_ = fmax(0, fmin(w, bb[4 * i + 0] + bb[4 * i + 2]));
        double ye_ = fmax(0, fmin(h, bb[4 * i + 1] + bb[4 * i + 3]));
        siz xs = round(xs_);
        siz ys = round(ys_);
        siz xe = round(xe_);
        siz ye = round(ye_);
        siz bw = (xs <= xe ? xe - xs : 0);
        siz bh = (ys <= ye ? ye - ys : 0);
        siz m;
        if (bw == 0 || bh == 0) {
            uint *cnts = rleInit(&R[i], h, w, 1);
            cnts[0] = h * w;
            continue;
        }
        if (bh == h) {
            // if the bounding box spans the entire height, it will have a single run of 1s.
            m = (xe == w ? 2 : 3);
            uint *cnts = rleInit(&R[i], h, w, m);
            cnts[0] = h * xs; // run of 0s
            // if it spans the entire width, it will have no final run of 0s.
            cnts[1] = h * bw; // run of 1s
        } else {
            // runs of 1 are the columns of the box (as many as the width of the box)
            // and each will have a preceding run of 0s. If the box does not end at the end of the image,
            // there will be a final run of 0s too.
            m = bw * 2 + (xe == w && ye == h ? 0 : 1);
            uint *cnts = rleInit(&R[i], h, w, m);
            cnts[0] = h * xs + ys; // run of 0s
            cnts[1] = bh; // run of 1s
            for (siz j = 1; j < bw; j++) {
                cnts[j * 2] = h - bh; // run of 0s
                cnts[j * 2 + 1] = bh; // run of 1s
            }
        }

        if (!(xe == w && ye == h)) {
            R[i].cnts[m - 1] = h * (w - xe) + (h - ye); // run of 0s
        }
    }
}

static int _uintCompare(const void *a, const void *b) {
    uint c = *((const uint *) a), d = *((const uint *) b);
    return c > d ? 1 : c < d ? -1 : 0;
}

void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w) {
    /* upsample and get discrete points densely along entire boundary */
    double scale = 5;
    int *x = malloc(sizeof(int) * (k + 1));
    int *y = malloc(sizeof(int) * (k + 1));
    for (siz j = 0; j < k; j++) {
        x[j] = (int) (scale * xy[j * 2 + 0] + .5);
    }
    x[k] = x[0];
    for (siz j = 0; j < k; j++) {
        y[j] = (int) (scale * xy[j * 2 + 1] + .5);
    }
    y[k] = y[0];

    siz m = 0;
    for (siz j = 0; j < k; j++) {
        m += uintMax(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1;
    }
    int *u = malloc(sizeof(int) * m);
    int *v = malloc(sizeof(int) * m);
    m = 0;
    for (siz j = 0; j < k; j++) {
        int xs = x[j], xe = x[j + 1], ys = y[j], ye = y[j + 1];
        int dx = abs(xe - xs);
        int dy = abs(ys - ye);
        bool flip = (dx >= dy && xs > xe) || (dx < dy && ys > ye);
        if (flip) {
            int tmp = xs;
            xs = xe;
            xe = tmp;
            tmp = ys;
            ys = ye;
            ye = tmp;
        }
        double s = dx >= dy ? (double) (ye - ys) / dx : (double) (xe - xs) / dy;
        if (dx >= dy) {
            for (int d = 0; d <= dx; d++) {
                int t = flip ? dx - d : d;
                u[m] = t + xs;
                v[m] = (int) (ys + s * t + .5);
                m++;
            }
        } else {
            for (int d = 0; d <= dy; d++) {
                int t = flip ? dy - d : d;
                v[m] = t + ys;
                u[m] = (int) (xs + s * t + .5);
                m++;
            }
        }
    }
    /* get points along y-boundary and downsample */
    k = m;
    m = 0;
    free(x);
    free(y);
    x = malloc(sizeof(int) * k);
    y = malloc(sizeof(int) * k);
    for (siz j = 1; j < k; j++) {
        if (u[j] != u[j - 1]) {
            double xd = u[j] < u[j - 1] ? u[j] : u[j] - 1;
            xd = (xd + .5) / scale - .5;
            if (floor(xd) != xd || xd < 0 || xd > w - 1) {
                continue;
            }
            double yd = v[j] < v[j - 1] ? v[j] : v[j - 1];
            yd = doubleClip((yd + .5) / scale - .5, 0, h);

            x[m] = (int) xd;
            y[m] = (int) ceil(yd);
            m++;
        }
    }
    /* compute rle encoding given y-boundary points */
    k = m;
    uint *a = rleInit(R, h, w, k + 1);
    for (siz j = 0; j < k; j++) {
        a[j] = (uint) (x[j] * (int) (h) + y[j]);
    }
    a[k++] = (uint) (h * w);
    free(u);
    free(v);
    free(x);
    free(y);
    qsort(a, k, sizeof(uint), _uintCompare);
    uint prev = 0;
    for (siz j = 0; j < k; j++) {
        uint tmp = a[j];
        a[j] -= prev;
        prev = tmp;
    }
    rleEliminateZeroRuns(R);
}

void rleFrCircle(RLE *R, const double *center_xy, double radius, siz h, siz w) {
    double cx = center_xy[0];
    double cy = center_xy[1];
    double r_sq = radius * radius;

    uint xstart = uintClip(floor(cx - radius + 1), 0, w);
    uint xend = uintClip(ceil(cx + radius), xstart, w);

    if (radius <= 0 || h == 0 || w == 0 || xstart == xend) {
        rleInit(R, h, w, 0);
        return;
    }

    // each col, its predecessor and final run of 0s
    uint *cnts = rleInit(R, h, w, ((xend - xstart) * 2 + 1));

    siz m = 0;

    // the initial zeros before the first column of the circle
    cnts[0] = h * xstart;

    for (uint x = xstart; x < xend; x++) {
        double dx = cx - (double) x;
        double dy = sqrt(r_sq - dx * dx);
        uint ystart = uintClip(floor(cy - dy + 1), 0, h);
        uint yend = uintClip(ceil(cy + dy), ystart, h);
        cnts[m++] += ystart; // 0s on top
        cnts[m++] = yend - ystart; // 1s
        cnts[m] = h - yend; // 0s on bottom
    }
    cnts[m++] += h * (w - xend);
    R->m =m;
    rleEliminateZeroRuns(R);
}