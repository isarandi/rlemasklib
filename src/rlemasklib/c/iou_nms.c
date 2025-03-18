#include <stdlib.h> // for malloc, free
#include <math.h> // for fmin, fmax
#include "minmax.h"
#include "shapes.h"
#include "moments.h"
#include "iou_nms.h"

void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o) {
    siz g, d;
    BB db, gb;
    int crowd;
    db = malloc(sizeof(double) * m * 4);
    rleToBbox(dt, db, m);
    gb = malloc(sizeof(double) * n * 4);
    rleToBbox(gt, gb, n);
    bbIou(db, gb, m, n, iscrowd, o);
    free(db);
    free(gb);
    for (g = 0; g < n; g++) {
        for (d = 0; d < m; d++) {
            if (o[g * m + d] > 0) {
                crowd = iscrowd != NULL && iscrowd[g];
                if (dt[d].h != gt[g].h || dt[d].w != gt[g].w) {
                    o[g * m + d] = -1;
                    continue;
                }
                siz ka, kb, a, b;
                uint c, ca, cb, ct, i, u;
                int va, vb;
                ca = dt[d].cnts[0];
                ka = dt[d].m;
                va = vb = 0;
                cb = gt[g].cnts[0];
                kb = gt[g].m;
                a = b = 1;
                i = u = 0;
                ct = 1;
                while (ct > 0) {
                    c = uintMin(ca, cb);
                    if (va || vb) {
                        u += c;
                        if (va && vb) {
                            i += c;
                        }
                    }
                    ct = 0;
                    ca -= c;
                    if (!ca && a < ka) {
                        ca = dt[d].cnts[a++];
                        va = !va;
                    }
                    ct += ca;
                    cb -= c;
                    if (!cb && b < kb) {
                        cb = gt[g].cnts[b++];
                        vb = !vb;
                    }
                    ct += cb;
                }
                if (i == 0) {
                    u = 1;
                } else if (crowd) {
                    rleArea(dt + d, 1, &u);
                }
                o[g * m + d] = (double) i / (double) u;
            }
        }
    }
}

void rleNms(RLE *dt, siz n, uint *keep, double thr) {
    siz i, j;
    double u;
    for (i = 0; i < n; i++) {
        keep[i] = 1;
    }
    for (i = 0; i < n; i++) {
        if (keep[i]) {
            for (j = i + 1; j < n; j++) {
                if (keep[j]) {
                    rleIou(dt + i, dt + j, 1, 1, 0, &u);
                    if (u > thr) {
                        keep[j] = 0;
                    }
                }
            }
        }
    }
}

void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o) {
    double h, w, i, u, ga, da;
    siz g, d;
    int crowd;
    for (g = 0; g < n; g++) {
        BB G = gt + g * 4;
        ga = G[2] * G[3];
        crowd = iscrowd != NULL && iscrowd[g];
        for (d = 0; d < m; d++) {
            BB D = dt + d * 4;
            da = D[2] * D[3];
            o[g * m + d] = 0;
            w = fmin(D[2] + D[0], G[2] + G[0]) - fmax(D[0], G[0]);
            if (w <= 0) {
                continue;
            }
            h = fmin(D[3] + D[1], G[3] + G[1]) - fmax(D[1], G[1]);
            if (h <= 0) {
                continue;
            }
            i = w * h;
            u = crowd ? da : da + ga - i;
            o[g * m + d] = i / u;
        }
    }
}

void bbNms(BB dt, siz n, uint *keep, double thr) {
    siz i, j;
    double u;
    for (i = 0; i < n; i++) { keep[i] = 1; }
    for (i = 0; i < n; i++) {
        if (keep[i]) {
            for (j = i + 1; j < n; j++) {
                if (keep[j]) {
                    bbIou(dt + i * 4, dt + j * 4, 1, 1, 0, &u);
                    if (u > thr) { keep[j] = 0; }
                }
            }
        }
    }
}
