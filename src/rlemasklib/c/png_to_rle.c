#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <libdeflate.h>

#ifdef __SSE2__
#include <emmintrin.h>
#define USE_SSE2 1
#else
#define USE_SSE2 0
#endif

#include "basics.h"
#include "png_to_rle.h"
#include "transpose_flip.h"

// PNG signature
static const byte PNG_SIG[8] = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};

// Forward declarations
static inline uint32_t read_be32(const byte *p);
static void unfilter_none(byte *curr, const byte *prev, siz width);
static void unfilter_sub(byte *curr, const byte *prev, siz width);
static void unfilter_up(byte *curr, const byte *prev, siz width);
static void unfilter_avg(byte *curr, const byte *prev, siz width);
static inline int paeth_predictor(int a, int b, int c);
static void unfilter_paeth(byte *curr, const byte *prev, siz width);
static bool parse_png_grayscale(const byte *png_data, siz png_len, siz *width_out, siz *height_out, byte **filtered_out);

bool rleFromPngFile(RLE *R, const char *path, int threshold) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;

    // Get file size
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return false;
    }

    byte *data = malloc(size);
    if (!data) {
        fclose(f);
        return false;
    }

    if (fread(data, 1, size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return false;
    }
    fclose(f);

    bool result = rleFromPngBytes(R, data, size, threshold);
    free(data);
    return result;
}

bool rleFromPngBytes(
    RLE *R,
    const byte *png_data,
    siz png_len,
    int threshold
) {
    siz width, height;
    byte *filtered;
    if (!parse_png_grayscale(png_data, png_len, &width, &height, &filtered)) {
        return false;
    }

    siz row_stride = 1 + width;

    // Zero buffer for first row's "previous" (no actual previous row)
    byte *zero_row = calloc(width, 1);
    if (!zero_row) {
        free(filtered);
        return false;
    }

    // Build RLE in row-major order (will transpose at end)
    RLE row_major;
    uint *cnts = rleInit(&row_major, width, height, height * width + 1);
    siz k = 0;

    byte prev = 0;  // Start with bg=0, so first-fg emits 0-length bg run automatically
    siz last_switch_pos = 0;
    siz pixel_pos = 0;

    for (siz row = 0; row < height; row++) {
        byte *curr_row = filtered + row * row_stride + 1;  // +1 skips filter byte
        byte *prev_row = (row == 0) ? zero_row : filtered + (row - 1) * row_stride + 1;
        byte filter_type = filtered[row * row_stride];

        // Apply unfilter in-place
        switch (filter_type) {
            case 0: unfilter_none(curr_row, prev_row, width); break;
            case 1: unfilter_sub(curr_row, prev_row, width); break;
            case 2: unfilter_up(curr_row, prev_row, width); break;
            case 3: unfilter_avg(curr_row, prev_row, width); break;
            case 4: unfilter_paeth(curr_row, prev_row, width); break;
            default:
                free(filtered);
                free(zero_row);
                rleFree(&row_major);
                return false;
        }

        // Build RLE - only do work at transitions
        for (siz i = 0; i < width; i++) {
            byte current = curr_row[i] > threshold;
            if (current != prev) {
                cnts[k++] = (uint)(pixel_pos - last_switch_pos);
                last_switch_pos = pixel_pos;
                prev = current;
            }
            pixel_pos++;
        }
    }

    // Final run
    cnts[k++] = (uint)(pixel_pos - last_switch_pos);

    free(filtered);
    free(zero_row);

    row_major.m = k;

    // Transpose to column-major: (W, H) -> (H, W)
    rleTranspose(&row_major, R);
    rleFree(&row_major);

    return true;
}

// Parse 8-bit grayscale PNG and return decompressed filtered data.
// Caller must free *filtered_out on success.
static bool parse_png_grayscale(
    const byte *png_data, siz png_len,
    siz *width_out, siz *height_out,
    byte **filtered_out
) {
    if (png_len < 8 + 25 || memcmp(png_data, PNG_SIG, 8) != 0) {
        return false;
    }

    siz pos = 8;
    siz width = 0, height = 0;

    // Collect IDAT chunks
    byte *idat_data = NULL;
    siz idat_len = 0, idat_cap = 0;

    while (pos + 12 <= png_len) {
        uint32_t len = read_be32(png_data + pos);
        const byte *type = png_data + pos + 4;
        const byte *data = png_data + pos + 8;

        if (pos + 12 + len > png_len) goto fail;

        if (memcmp(type, "IHDR", 4) == 0) {
            if (len < 13) goto fail;
            width = read_be32(data);
            height = read_be32(data + 4);
            // Only 8-bit grayscale
            if (data[8] != 8 || data[9] != 0) goto fail;
        }
        else if (memcmp(type, "IDAT", 4) == 0) {
            if (idat_len + len > idat_cap) {
                idat_cap = idat_cap ? idat_cap * 2 : len * 2;
                if (idat_cap < idat_len + len) idat_cap = idat_len + len;
                byte *tmp = realloc(idat_data, idat_cap);
                if (!tmp) goto fail;
                idat_data = tmp;
            }
            memcpy(idat_data + idat_len, data, len);
            idat_len += len;
        }
        else if (memcmp(type, "IEND", 4) == 0) {
            break;
        }
        pos += 12 + len;
    }

    if (!width || !height || !idat_data) goto fail;

    // Decompress
    siz filtered_len = height * (1 + width);
    byte *filtered = malloc(filtered_len);
    if (!filtered) goto fail;

    struct libdeflate_decompressor *d = libdeflate_alloc_decompressor();
    if (!d) { free(filtered); goto fail; }

    size_t actual;
    bool ok = libdeflate_zlib_decompress(d, idat_data, idat_len, filtered, filtered_len, &actual)
              == LIBDEFLATE_SUCCESS && actual == filtered_len;
    libdeflate_free_decompressor(d);
    free(idat_data);

    if (!ok) { free(filtered); return false; }

    *width_out = width;
    *height_out = height;
    *filtered_out = filtered;
    return true;

fail:
    free(idat_data);
    return false;
}

// Read big-endian 32-bit integer
static inline uint32_t read_be32(const byte *p) {
    return ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// PNG unfilter functions
static void unfilter_none(byte *curr, const byte *prev, siz width) {
    (void)prev;
    (void)curr;
    (void)width;
    // Data already in place, nothing to do
}

static void unfilter_sub(byte *curr, const byte *prev, siz width) {
    (void)prev;
    // Sequential dependency - each pixel depends on previous
    for (siz i = 1; i < width; i++) {
        curr[i] = (curr[i] + curr[i-1]) & 0xFF;
    }
}

static void unfilter_up(byte *curr, const byte *prev, siz width) {
#if USE_SSE2
    siz i = 0;
    // Process 16 bytes at a time with SSE2
    for (; i + 16 <= width; i += 16) {
        __m128i c = _mm_loadu_si128((__m128i*)(curr + i));
        __m128i p = _mm_loadu_si128((__m128i*)(prev + i));
        _mm_storeu_si128((__m128i*)(curr + i), _mm_add_epi8(c, p));
    }
    // Remainder
    for (; i < width; i++) {
        curr[i] = (curr[i] + prev[i]) & 0xFF;
    }
#else
    for (siz i = 0; i < width; i++) {
        curr[i] = (curr[i] + prev[i]) & 0xFF;
    }
#endif
}

static void unfilter_avg(byte *curr, const byte *prev, siz width) {
    // First pixel: left = 0
    curr[0] = (curr[0] + (prev[0] >> 1)) & 0xFF;
    for (siz i = 1; i < width; i++) {
        curr[i] = (curr[i] + ((curr[i-1] + prev[i]) >> 1)) & 0xFF;
    }
}

static inline int paeth_predictor(int a, int b, int c) {
    int p = a + b - c;
    int pa = p > a ? p - a : a - p;
    int pb = p > b ? p - b : b - p;
    int pc = p > c ? p - c : c - p;
    if (pa <= pb && pa <= pc) return a;
    if (pb <= pc) return b;
    return c;
}

static void unfilter_paeth(byte *curr, const byte *prev, siz width) {
    // First pixel: left=0, up_left=0
    curr[0] = (curr[0] + prev[0]) & 0xFF;
    for (siz i = 1; i < width; i++) {
        int pred = paeth_predictor(curr[i-1], prev[i], prev[i-1]);
        curr[i] = (curr[i] + pred) & 0xFF;
    }
}

siz rlesFromLabelMapPngBytes(RLE *Rs, const byte *png_data, siz png_len) {
    siz width, height;
    byte *filtered;
    if (!parse_png_grayscale(png_data, png_len, &width, &height, &filtered)) {
        return 0;
    }

    siz row_stride = 1 + width;
    byte *zero_row = calloc(width, 1);
    if (!zero_row) {
        free(filtered);
        return 0;
    }

    // State per label for row-major RLEs
    siz last_pos[255] = {0};
    siz k[255] = {0};
    siz cap[255] = {0};
    RLE row_major[255];  // Row-major RLEs, will transpose at end

    byte prev = 0;
    siz pos = 0;
    siz a = height * width;

    for (siz row = 0; row < height; row++) {
        byte *curr_row = filtered + row * row_stride + 1;
        byte *prev_row = (row == 0) ? zero_row : filtered + (row - 1) * row_stride + 1;
        byte filter_type = filtered[row * row_stride];

        switch (filter_type) {
            case 0: unfilter_none(curr_row, prev_row, width); break;
            case 1: unfilter_sub(curr_row, prev_row, width); break;
            case 2: unfilter_up(curr_row, prev_row, width); break;
            case 3: unfilter_avg(curr_row, prev_row, width); break;
            case 4: unfilter_paeth(curr_row, prev_row, width); break;
            default:
                free(filtered);
                free(zero_row);
                for (int i = 0; i < 255; i++) {
                    if (cap[i] > 0) rleFree(&row_major[i]);
                }
                return 0;
        }

        // Build RLEs for each label
        for (siz i = 0; i < width; i++) {
            byte label = curr_row[i];
            if (label != prev) {
                if (prev > 0) {
                    siz idx = prev - 1;
                    if (k[idx] >= cap[idx]) {
                        if (cap[idx] == 0) {
                            cap[idx] = 10 * height;
                            rleInit(&row_major[idx], width, height, cap[idx]);
                        } else {
                            cap[idx] *= 2;
                            rleRealloc(&row_major[idx], cap[idx]);
                        }
                    }
                    row_major[idx].cnts[k[idx]++] = (uint)(pos - last_pos[idx]);
                    last_pos[idx] = pos;
                }
                if (label > 0) {
                    siz idx = label - 1;
                    if (k[idx] >= cap[idx]) {
                        if (cap[idx] == 0) {
                            cap[idx] = 10 * height;
                            rleInit(&row_major[idx], width, height, cap[idx]);
                        } else {
                            cap[idx] *= 2;
                            rleRealloc(&row_major[idx], cap[idx]);
                        }
                    }
                    row_major[idx].cnts[k[idx]++] = (uint)(pos - last_pos[idx]);
                    last_pos[idx] = pos;
                }
                prev = label;
            }
            pos++;
        }
    }

    free(filtered);
    free(zero_row);

    // Finalize and transpose active RLEs
    siz n_active = 0;
    for (int i = 0; i < 255; i++) {
        if (k[i] > 0) {
            if (k[i] >= cap[i]) {
                rleRealloc(&row_major[i], k[i] + 1);
            }
            row_major[i].cnts[k[i]++] = (uint)(a - last_pos[i]);
            row_major[i].m = k[i];
            // Transpose from row-major (W,H) to column-major (H,W)
            rleTranspose(&row_major[i], &Rs[i]);
            rleFree(&row_major[i]);
            n_active++;
        } else {
            Rs[i].cnts = NULL;
        }
    }

    return n_active;
}

siz rlesFromLabelMapPngFile(RLE *Rs, const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size <= 0) {
        fclose(f);
        return 0;
    }

    byte *data = malloc(size);
    if (!data) {
        fclose(f);
        return 0;
    }

    if (fread(data, 1, size, f) != (size_t)size) {
        free(data);
        fclose(f);
        return 0;
    }
    fclose(f);

    siz result = rlesFromLabelMapPngBytes(Rs, data, size);
    free(data);
    return result;
}
