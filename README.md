# Run-Length Encoded Mask Operations

This library is an extended and improved version of the [COCO API](https://github.com/cocodataset/cocoapi)'s `pycocotools.mask` module (which was originally written by Piotr Dollár and Tsung-Yi Lin).

It offers the following additional features:

- Further set operations (complement, difference, symmetric difference) in RLE, without decoding
- Mask cropping in RLE (`rlemasklib.crop`), without decoding
- Direct creation of full and empty masks in RLE
- Decompression of COCO's compressed RLE format to integer run-lengths, and vice versa
- Extra compression (optional) using gzip on top of the LEB128-like encoding used by the COCO API (~40% reduction beyond
  the COCO compression)
- More streamlined API

## List of functions

### Encoding / decoding
- `rle_mask = rlemasklib.encode(binary_mask)`: Encode a binary mask as RLE
- `binary_mask = rlemasklib.decode(rle_mask)`: Decode an RLE mask to a binary mask

### Compression / decompression
- `rle_mask = rlemasklib.compress(rle_mask)`: Compress an RLE mask using LEB128 (and optionally gzip)
- `rle_mask = rlemasklib.decompress(rle_mask)`: Decompress an RLE mask from LEB128 or gzip to an array of integers (run-lengths)

### Set operations
- `rle_mask = rlemasklib.empty(imshape)`: Create an empty RLE mask of given size
- `rle_mask = rlemasklib.full(imshape)`: Create a full RLE mask of given size
- `rle_mask = rlemasklib.intersection(rle_masks)`: Compute the intersection of multiple RLE masks.
- `rle_mask = rlemasklib.union(rle_masks)`: Compute the union of multiple RLE masks.
- `rle_mask = rlemasklib.complement(rle_mask)`: Compute the complement of an RLE mask.
- `rle_mask = rlemasklib.difference(rle_mask1, rle_mask2)`: Compute the difference of two RLE masks.
- `rle_mask = rlemasklib.symmetric_difference(rle_mask1, rle_mask2)`: Compute the symmetric difference of two RLE masks.

- `area = rlemasklib.area(rle_mask)`: Compute the area of an RLE mask
- `iou = rlemasklib.iou(rle_mask1, rle_mask2)`: Compute the intersection-over-union of two RLE masks.

### Cropping
- `rle_mask = rlemasklib.crop(rle_mask, bbox)`: Crop an RLE mask to a given bounding box, yielding a mask with smaller height and/or width.

### Conversions
- `[x_start, y_start, width, height] = rlemasklib.to_bbox(rle_mask)`: Convert an RLE mask to a bounding box.
- `rle_mask = rlemasklib.from_bbox([x_start, y_start, width, height], imshape)`: Convert a bounding box to an RLE mask inside a given image size.
- `rle_mask = rlemasklib.from_polygon(polygon, imshape)`: Convert a polygon to an RLE mask inside a given image size.
