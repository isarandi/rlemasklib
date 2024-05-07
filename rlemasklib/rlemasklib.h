/**************************************************************************
* Based on code from the Microsoft COCO Toolbox.      version 2.0
* Code written by Piotr Dollar and Tsung-Yi Lin, 2015
* Modifications by Istvan Sarandi, 2023
* Licensed under the Simplified BSD License [see license.txt]
**************************************************************************/
#pragma once
#include <stdbool.h>

typedef unsigned int uint;
typedef unsigned long siz;
typedef unsigned char byte;
typedef double *BB;
typedef struct {
    siz h, w, m;
    uint *cnts;
} RLE;

/* Initialize/destroy RLE. */
void rleInit(RLE *R, siz h, siz w, siz m, uint *cnts, bool transfer_ownership);

void rleFree(RLE *R);

/* Initialize/destroy RLE array. */
void rlesInit(RLE **R, siz n);

void rlesFree(RLE **R, siz n);

/* Shrink RLE by reallocating cnts to the actual size. */
void rleShrink(RLE* R);

/* Encode binary masks using RLE. */
void rleEncode(RLE *R, const byte *mask, siz h, siz w, siz n);

/* Decode binary masks encoded via RLE. */
void rleDecode(const RLE *R, byte *mask, siz n);

/* Compute union or intersection of encoded masks. */
void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc);

/* Compute area of encoded masks. */
void rleArea(const RLE *R, siz n, uint *a);

/* Compute the complement of encoded masks. */
void rleComplement(const RLE *R, RLE *M, siz n);
void rleComplementInplace(RLE *R, siz n);

/* Crop encoded masks. */
void rleCrop(const RLE *R, RLE *M, siz n, const uint *bbox);
void rleCropInplace(RLE *R, siz n, const uint *bbox);

/* Pad encoded masks. */
void rlePad(const RLE *R, RLE *M, siz n, const uint *pad_amounts);

/* Compute intersection over union between masks. */
void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o);

/* Compute non-maximum suppression between bounding masks */
void rleNms(RLE *dt, siz n, uint *keep, double thr);

/* Compute intersection over union between bounding boxes. */
void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o);

/* Compute non-maximum suppression between bounding boxes */
void bbNms(BB dt, siz n, uint *keep, double thr);

/* Get bounding boxes surrounding encoded masks. */
void rleToBbox(const RLE *R, BB bb, siz n);

/* Convert bounding boxes to encoded masks. */
void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n);

/* Convert polygon to encoded mask. */
void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w);

/* Get compressed string representation of encoded mask. */
char *rleToString(const RLE *R);

/* Convert from compressed string representation of encoded mask. */
void rleFrString(RLE *R, char *s, siz h, siz w);

/* Remove zero runlengths from RLE encoding, and sum up the neighbors accordingly. */
void rleEliminateZeroRuns(RLE *R, siz n);

/* Compute connected components of an encoded mask */
void rleConnectedComponents(const RLE *R_in, int connectivity, siz min_size, RLE **components, siz *n_components_out);

/* Split runs that may belong to different connected components */
RLE *rleSplitRunsThatMayBelongToDifferentComponents(const RLE *R, int connectivity);

/* Compute the centroids of the encoded masks */
void rleCentroid(const RLE *R, double *xys, siz n);

// Union-find data structure for connected components
struct UnionFindNode {
    struct UnionFindNode *parent;
    siz size;
};
struct UnionFindNode *uf_find(struct UnionFindNode *x);
void uf_union(struct UnionFindNode *x, struct UnionFindNode *y);
