#pragma once
#include "basics.h"

// Convert PNG bytes to RLE mask.
// Only supports 8-bit grayscale PNG.
// Pixels >= threshold become foreground (1), others background (0).
// Returns true on success, false on error.
bool rleFromPngBytes(
    RLE *R,
    const byte *png_data,
    siz png_len,
    int threshold
);

// Convert PNG file to RLE mask.
// Convenience wrapper that reads file then calls rleFromPngBytes.
bool rleFromPngFile(RLE *R, const char *path, int threshold);

// Convert PNG label map to multiple RLE masks.
// Label 0 is background, labels 1-255 become Rs[0]-Rs[254].
// Active labels have cnts != NULL, unused labels have cnts = NULL.
// Returns number of active labels, or 0 on error.
siz rlesFromLabelMapPngBytes(RLE *Rs, const byte *png_data, siz png_len);
siz rlesFromLabelMapPngFile(RLE *Rs, const char *path);
