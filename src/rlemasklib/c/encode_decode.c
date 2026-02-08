#include <string.h> // for memset
#include <stdlib.h> // for malloc, realloc, free
#include <stdbool.h> // for bool

#ifdef __SSE2__
#include <emmintrin.h>
#define USE_SSE2 1
#else
#define USE_SSE2 0
#endif

#include "basics.h"
#include "encode_decode.h"

void rleEncode(RLE *R, const byte *M, siz h, siz w, siz n) {
    if (h == 0 || w == 0) {
        for (siz i = 0; i < n; i++) {
            rleInit(&R[i], h, w, 0);
        }
        return;
    }

    siz a = w * h;
#if USE_SSE2
    __m128i zero = _mm_setzero_si128();
#endif

    for (siz i = 0; i < n; i++) {
        const byte *T = M + a * i;
        uint *cnts = rleInit(&R[i], h, w, a+1);
        siz k = 0;
        byte prev = 0;
        siz last_pos = 0;
        siz j = 0;

#if USE_SSE2
        // SSE2 path: skip uniform chunks of 16 bytes
        while (j + 16 <= a) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(T + j));
            __m128i is_zero = _mm_cmpeq_epi8(chunk, zero);
            int mask = _mm_movemask_epi8(is_zero);

            if (mask == 0xFFFF) {
                // All 16 bytes are zero (background)
                if (prev != 0) {
                    cnts[k++] = (uint)(j - last_pos);
                    last_pos = j;
                    prev = 0;
                }
                j += 16;
            } else if (mask == 0) {
                // All 16 bytes are non-zero (foreground)
                if (prev == 0) {
                    cnts[k++] = (uint)(j - last_pos);
                    last_pos = j;
                    prev = 1;
                }
                j += 16;
            } else {
                // Mixed chunk - process byte by byte
                for (int b = 0; b < 16; b++, j++) {
                    byte current = T[j] != 0;
                    if (current != prev) {
                        cnts[k++] = (uint)(j - last_pos);
                        last_pos = j;
                        prev = current;
                    }
                }
            }
        }
#endif
        // Scalar remainder
        for (; j < a; j++) {
            byte current = T[j] != 0;
            if (current != prev) {
                cnts[k++] = (uint)(j - last_pos);
                last_pos = j;
                prev = current;
            }
        }

        cnts[k++] = (uint)(a - last_pos);
        rleRealloc(&R[i], k);
    }
}

void rleEncodeThresh128(RLE *R, const byte *M, siz h, siz w, siz n) {
    if (h == 0 || w == 0) {
        for (siz i = 0; i < n; i++) {
            rleInit(&R[i], h, w, 0);
        }
        return;
    }
    siz a = w * h;
    for (siz i = 0; i < n; i++) {
        const byte *T = M + a * i;
        uint *cnts = rleInit(&R[i], h, w, a+1);
        siz k = 0;
        byte prev = 0;
        siz last_pos = 0;
        siz j = 0;

#if USE_SSE2
        // SSE2 path: _mm_movemask_epi8 extracts high bits - perfect for >= 128 check
        while (j + 16 <= a) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(T + j));
            int mask = _mm_movemask_epi8(chunk);  // high bit of each byte

            if (mask == 0) {
                // All 16 bytes < 128 (background)
                if (prev != 0) {
                    cnts[k++] = (uint)(j - last_pos);
                    last_pos = j;
                    prev = 0;
                }
                j += 16;
            } else if (mask == 0xFFFF) {
                // All 16 bytes >= 128 (foreground)
                if (prev == 0) {
                    cnts[k++] = (uint)(j - last_pos);
                    last_pos = j;
                    prev = 0x80;
                }
                j += 16;
            } else {
                // Mixed chunk - process byte by byte
                for (int b = 0; b < 16; b++, j++) {
                    byte current = T[j] & 0x80;
                    if (current != prev) {
                        cnts[k++] = (uint)(j - last_pos);
                        last_pos = j;
                        prev = current;
                    }
                }
            }
        }
#endif
        // Scalar remainder
        for (; j < a; j++) {
            byte current = T[j] & 0x80;
            if (current != prev) {
                cnts[k++] = (uint)(j - last_pos);
                last_pos = j;
                prev = current;
            }
        }

        cnts[k++] = (uint)(a - last_pos);
        rleRealloc(&R[i], k);
    }
}

void rleEncodeThreshold(RLE *R, const byte *M, siz h, siz w, siz n, int threshold) {
    if (threshold <= 1) {
        rleEncode(R, M, h, w, n);
    } else if (threshold == 128) {
        rleEncodeThresh128(R, M, h, w, n);
    } else {
        if (h == 0 || w == 0) {
            for (siz i = 0; i < n; i++) {
                rleInit(&R[i], h, w, 0);
            }
            return;
        }
        siz a = w * h;
        for (siz i = 0; i < n; i++) {
            const byte *T = M + a * i;
            uint *cnts = rleInit(&R[i], h, w, a + 1);
            siz k = 0;
            byte prev = 0;
            siz last_pos = 0;
            for (siz j = 0; j < a; j++) {
                byte current = T[j] >= (byte)threshold;
                if (current != prev) {
                    cnts[k++] = (uint)(j - last_pos);
                    last_pos = j;
                    prev = current;
                }
            }
            cnts[k++] = (uint)(a - last_pos);
            rleRealloc(&R[i], k);
        }
    }
}

bool rleDecode(const RLE *R, byte *M, siz n, byte value) {
    // Background pixels are not touched, so M should be pre-initialized for background.
    byte *end = M + R->h * R->w * n;
    for (siz i = 0; i < n; i++) {
        for (siz j = 0; j < (R[i].m/2)*2; j++) {
            uint cnt = R[i].cnts[j];
            if (j%2) {
                if (M + cnt > end) {
                    return false;
                }
                memset(M, value, cnt);
            }
            M += cnt;
        }
    }
    return true;
}

bool rleDecodeStrided(const RLE *R, byte *M, siz row_stride, siz col_stride, byte value) {
    // Decode into a strided 2D array (e.g., a channel slice of HWC image).
    // RLE is column-major, so we iterate down columns first.
    // Background pixels are not touched.
    siz h = R->h;
    siz w = R->w;
    siz pos = 0;  // linear position in column-major order

    for (siz j = 0; j < (R->m/2)*2; j++) {
        uint cnt = R->cnts[j];
        if (j % 2) {  // foreground run
            for (siz k = 0; k < cnt; k++) {
                siz col = pos / h;
                siz row = pos % h;
                M[row * row_stride + col * col_stride] = value;
                pos++;
            }
        } else {
            pos += cnt;
        }
    }
    return true;
}

bool rleDecodeBroadcast(const RLE *R, byte *M, siz num_channels, byte value) {
    // Decode into interleaved multi-channel array (HWC layout).
    // Same value written to all channels of foreground pixels.
    // Background pixels are not touched.
    byte *end = M + R->h * R->w * num_channels;
    for (siz j = 0; j < (R->m/2)*2; j++) {
        uint cnt = R->cnts[j];
        siz byte_cnt = cnt * num_channels;
        if (j % 2) {
            if (M + byte_cnt > end) {
                return false;
            }
            memset(M, value, byte_cnt);
        }
        M += byte_cnt;
    }
    return true;
}

static bool rleDecodeRGB(const RLE *R, byte *M, const byte *values) {
    byte *end = M + R->h * R->w * 3;
    byte v0 = values[0], v1 = values[1], v2 = values[2];

    for (siz j = 0; j < (R->m/2)*2; j++) {
        uint cnt = R->cnts[j];
        if (j % 2) {
            if (M + cnt * 3 > end) return false;
            for (siz k = 0; k < cnt; k++) {
                M[0] = v0; M[1] = v1; M[2] = v2;
                M += 3;
            }
        } else {
            M += cnt * 3;
        }
    }
    return true;
}

static bool rleDecodeRGBA(const RLE *R, byte *M, const byte *values) {
    byte *end = M + R->h * R->w * 4;
    uint32_t v32;
    memcpy(&v32, values, 4);

    for (siz j = 0; j < (R->m/2)*2; j++) {
        uint cnt = R->cnts[j];
        if (j % 2) {
            if (M + cnt * 4 > end) return false;
            for (siz k = 0; k < cnt; k++) {
                memcpy(M, &v32, 4);
                M += 4;
            }
        } else {
            M += cnt * 4;
        }
    }
    return true;
}

static bool rleDecodeMultiValueGeneric(const RLE *R, byte *M, siz num_channels, const byte *values) {
    byte *end = M + R->h * R->w * num_channels;

    for (siz j = 0; j < (R->m/2)*2; j++) {
        uint cnt = R->cnts[j];
        if (j % 2) {
            if (M + cnt * num_channels > end) return false;
            for (siz k = 0; k < cnt; k++) {
                memcpy(M, values, num_channels);
                M += num_channels;
            }
        } else {
            M += cnt * num_channels;
        }
    }
    return true;
}

bool rleDecodeMultiValue(const RLE *R, byte *M, siz num_channels, const byte *values) {
    // Decode into interleaved multi-channel array (HWC layout).
    // Each channel gets its own value from the values array.
    // M is not modified for background pixels.
    if (num_channels == 3) {
        return rleDecodeRGB(R, M, values);
    } else if (num_channels == 4) {
        return rleDecodeRGBA(R, M, values);
    } else {
        return rleDecodeMultiValueGeneric(R, M, num_channels, values);
    }
}


char *rleToString(const RLE *R) {
    /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
    siz m = R->m;
    // optimistic allocation (realistic masks tend to take up less than 2x the space of the rle)
    // in a single byte we can store 5 bits of the run length (difference), allowing for a range of -16 to 15
    // in two bytes we can store 10 bits, allowing for a range of -512 to 511
    // experimentally, in a realistic human mask dataset (mixed sources) the factor for the 99th percentile is
    // still below 2.0, so we use that as a heuristic. The average is around 1.5.
    siz alloc_size = 2 * m * sizeof(char);
    char *s = malloc(alloc_size); // optimistic allocation
    siz p = 0;
    bool more;
    for (siz i = 0; i < m; i++) {
        long x = R->cnts[i];
        if (i > 2) {
            x -= R->cnts[i - 2];  // take the difference from the last run of the same value
        }

        do {
            char c = x & 0x1f;  // last 5 bits of the run length difference
            x >>= 5;
            // if all the remaining bits are the same as the highest bit of the current 5 bits,
            // then this is the last 5-bit chunk we need
            more = (c & 0x10) ? x != -1 : x != 0;
            if (more) {
                c |= 0x20; // set continuation bit at the 3rd position
            }

            if (p >= alloc_size) {
                alloc_size *= 2;
                s = realloc(s, alloc_size);
            }
            s[p++] = c + 48;  // ascii 48 is '0'. 48-111 is the range of ascii chars we use
        } while (more);
    }
    s = realloc(s, sizeof(char) * (p + 1));
    s[p] = 0; // null-terminate the string
    return s;
}



void leb128_encode(const int *cnts, siz m, char **out, siz *n_out) {
    siz alloc_size = 2 * m * sizeof(char);
    char *s = malloc(alloc_size); // optimistic allocation
    siz p = 0;
    bool more;
    for (siz i = 0; i < m; i++) {
        long x = cnts[i];
        do {
            char c = x & 0x7f;  // last 7 bits of the run difference
            x >>= 7;
            // if all the remaining bits are the same as the highest bit of the current 7 bits,
            // then this is the last 7-bit chunk we need
            more = (c & 0x40) ? x != -1 : x != 0;
            if (more) {
                c |= 0x80; // set continuation bit at the 1st position
            }

            if (p >= alloc_size) {
                alloc_size *= 2;
                s = realloc(s, alloc_size);
            }
            s[p++] = c;
        } while (more);
    }
    s = realloc(s, sizeof(char) * p);
    *out = s;
    *n_out = p;
}

void leb128_decode(const char *s, siz n, int **cnts_out, siz *m_out) {
    siz p = 0;
    siz m = 0;
    int *cnts = malloc(sizeof(int) * n);
    while (p < n) {
        long x = 0; // the run length (difference)
        siz k = 0; // the number of bytes (of which 7 bits and a continuation bit are used) in the run length
        while (true) {
            char c = s[p];
            x |= (long) (c & 0x7f) << k; // take the last 7 bits of the char and shift them to the right position
            bool more = c & 0x80; // check the continuation bit
            p++;
            k += 7;
            if (!more) {
                if (c & 0x40) {
                    // if the highest bit of the last 7 bits is set, set all the remaining high bits to 1
                    x |= -1L << k;
                }
                break;
            }
        }
        cnts[m++] = x;
    }
    *cnts_out = cnts;
    *m_out = m;
}


void leb_coco_encode(const int *cnts, siz m, char **out, siz *n_out) {
    /* Similar to LEB128 but using 6 bits/char and ascii chars 48-111. */
    // optimistic allocation (realistic masks tend to take up less than 2x the space of the rle)
    // in a single byte we can store 5 bits of the run length (difference), allowing for a range of -16 to 15
    // in two bytes we can store 10 bits, allowing for a range of -512 to 511
    // experimentally, in a realistic human mask dataset (mixed sources) the factor for the 99th percentile is
    // still below 2.0, so we use that as a heuristic. The average is around 1.5.
    siz alloc_size = 2 * m * sizeof(char);
    char *s = malloc(alloc_size); // optimistic allocation
    siz p = 0;
    bool more;
    for (siz i = 0; i < m; i++) {
        long x = cnts[i];
        do {
            char c = x & 0x1f;  // last 7 bits of the run difference
            x >>= 5;
            // if all the remaining bits are the same as the highest bit of the current 7 bits,
            // then this is the last 7-bit chunk we need
            more = (c & 0x10) ? x != -1 : x != 0;
            if (more) {
                c |= 0x20; // set continuation bit at the 1st position
            }

            if (p >= alloc_size) {
                alloc_size *= 2;
                s = realloc(s, alloc_size);
            }
            s[p++] = c + 48;
        } while (more);
    }
    s = realloc(s, sizeof(char) * p);
    *out = s;
    *n_out = p;
}

void leb_coco_decode(const char *s, siz n, int **cnts_out, siz *m_out) {
    siz p = 0;
    siz m = 0;
    int *cnts = malloc(sizeof(int) * n);
    while (p < n) {
        long x = 0; // the run length (difference)
        siz k = 0; // the number of bytes (of which 7 bits and a continuation bit are used) in the run length
        while (true) {
            char c = s[p] - 48;
            x |= (long) (c & 0x1f) << k; // take the last 7 bits of the char and shift them to the right position
            bool more = c & 0x20; // check the continuation bit
            p++;
            k += 5;
            if (!more) {
                if (c & 0x10) {
                    // if the highest bit of the last 7 bits is set, set all the remaining high bits to 1
                    x |= -1L << k;
                }
                break;
            }
        }
        cnts[m++] = x;
    }
    *cnts_out = cnts;
    *m_out = m;
}

void rleFrString(RLE *R, const char *s, siz h, siz w) {
    if (h == 0 || w == 0) {
        rleInit(R, h, w, 0);
        return;
    }

    uint *cnts = rleInit(R, h, w, strlen(s));

    siz m = 0;
    for (siz p = 0; s[p];) {
        long x = 0; // the run length (difference)
        siz k = 0; // the number of bytes (of which 5 bits and a continuation bit are used) in the run length
        while (true) {
            char c = s[p];
            if (!c) {
                return; // unexpected end of string, last char promised more to come, but lied. Malformed input!
            }
            c -= 48; // subtract the offset, so the range is 0-63
            x |= (long) (c & 0x1f) << k; // take the last 5 bits of the char and shift them to the right position
            bool more = c & 0x20; // check the continuation bit
            p++;
            k += 5;
            if (!more) {
                if (c & 0x10) {
                    // if the highest bit of the last 5 bits is set, set all the remaining high bits to 1
                    x |= -1L << k;
                }
                break;
            }
        }
        if (m > 2) {
            x += cnts[m - 2]; // add the difference to the last run of the same value
        }
        cnts[m++] = x;
    }
    rleRealloc(R, m);
}


void rlesToLabelMapZeroInit(const RLE **Rs, byte *M, siz n) {
    for (siz i = 0; i < n; i++) {
        siz r = 0;
        uint *cnts = Rs[i]->cnts;
        siz m = Rs[i]->m;
        for (siz j = 1; j < m; j += 2) {
            r += cnts[j - 1];
            memset(M + r, i + 1, cnts[j]);
            r += cnts[j];
        }
    }
}

siz rleFromLabelMap(const byte *M, siz h, siz w, RLE *Rs) {
    // Single-pass conversion of label map (0-255) to up to 255 RLEs.
    // Label 0 is background (skipped). Labels 1-255 become Rs[0]-Rs[254].
    // Returns count of non-empty RLEs (labels that appeared with foreground).
    // Rs must be pre-allocated array of 255 RLEs (uninitialized on entry).

    siz a = h * w;

    // State per label: last_pos, k (run index), capacity
    // cap[i]==0 means uninitialized
    siz last_pos[255] = {0};
    siz k[255] = {0};
    siz cap[255] = {0};

    byte prev = 0;
    siz pos = 0;

    // Single pass through label map (column-major order)
    for (siz j = 0; j < a; j++) {
        byte label = M[j];

        if (label != prev) {
            // Transition from prev to label
            if (prev > 0) {
                // prev exits foreground
                siz idx = prev - 1;
                if (k[idx] >= cap[idx]) {
                    if (cap[idx] == 0) {
                        cap[idx] = 10 * w;
                        rleInit(&Rs[idx], h, w, cap[idx]);
                    } else {
                        cap[idx] *= 2;
                        rleRealloc(&Rs[idx], cap[idx]);
                    }
                }
                Rs[idx].cnts[k[idx]++] = (uint)(pos - last_pos[idx]);
                last_pos[idx] = pos;
            }
            if (label > 0) {
                // label enters foreground
                siz idx = label - 1;
                if (k[idx] >= cap[idx]) {
                    if (cap[idx] == 0) {
                        cap[idx] = 10 * w;
                        rleInit(&Rs[idx], h, w, cap[idx]);
                    } else {
                        cap[idx] *= 2;
                        rleRealloc(&Rs[idx], cap[idx]);
                    }
                }
                Rs[idx].cnts[k[idx]++] = (uint)(pos - last_pos[idx]);
                last_pos[idx] = pos;
            }
            prev = label;
        }
        pos++;
    }

    // Finalize active RLEs, leave unused as NULL
    siz n_active = 0;
    for (int i = 0; i < 255; i++) {
        if (k[i] > 0) {
            // Active label - emit final run and shrink to actual size
            if (k[i] >= cap[i]) {
                rleRealloc(&Rs[i], k[i] + 1);
            }
            Rs[i].cnts[k[i]++] = (uint)(a - last_pos[i]);
            Rs[i].m = k[i];
            n_active++;
        } else {
            // Unused label - leave as NULL (caller checks cnts != NULL)
            Rs[i].cnts = NULL;
        }
    }

    return n_active;
}