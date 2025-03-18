#pragma once
#include "basics.h"

void rleEncode(RLE *R, const byte *M, siz h, siz w, siz n);
void rleEncodeThresh128(RLE *R, const byte *M, siz h, siz w, siz n);
bool rleDecode(const RLE *R, byte *M, siz n, byte value);
void rlesToLabelMapZeroInit(const RLE **R, byte *M, siz n);

char *rleToString(const RLE *R);
void rleFrString(RLE *R, const char *s, siz h, siz w);



void leb128_encode(const int *cnts, siz m, char **out, siz *n_out);
void leb128_decode(const char *s, siz n, int **cnts_out, siz *m_out);
void leb_coco_encode(const int *cnts, siz m, char **out, siz *n_out);
void leb_coco_decode(const char *s, siz n, int **cnts_out, siz *m_out);
