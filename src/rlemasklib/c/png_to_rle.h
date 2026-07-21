#pragma once
#include "basics.h"

// Convert PNG bytes to RLE mask.
// Supports 8-bit grayscale, gray+alpha, RGB, and RGBA PNGs.
// Pixels >= threshold become foreground (1), others background (0).
// channel: -1 = grayscale only (reject multi-channel), 0+ = extract that channel index.
// Returns true on success, false on error.
bool rleFromPngBytes(
    RLE *R,
    const byte *png_data,
    siz png_len,
    int threshold,
    int channel
);

// Convert PNG file to RLE mask.
// Convenience wrapper that reads file then calls rleFromPngBytes.
bool rleFromPngFile(RLE *R, const char *path, int threshold, int channel);

// Convert PNG label map to multiple RLE masks.
// Label 0 is background, labels 1-255 become Rs[0]-Rs[254].
// Active labels have cnts != NULL, unused labels have cnts = NULL.
// Returns number of active labels, or 0 on error.
siz rlesFromLabelMapPngBytes(RLE *Rs, const byte *png_data, siz png_len);
siz rlesFromLabelMapPngFile(RLE *Rs, const char *path);
