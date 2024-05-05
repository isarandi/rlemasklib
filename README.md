# Run-Length Encoded Mask Operations

This library is an extended and improved version of the [COCO API](https://github.com/cocodataset/cocoapi)'s `pycocotools.mask` module (which was originally written by Piotr Dollár and Tsung-Yi Lin).

It offers the following additional features:

- Further set operations (complement, difference, symmetric difference) in RLE, without decoding
- Mask cropping and padding in RLE
- Connected components extraction in RLE (`rlemasklib.connected_components`) 
- Direct creation of full and empty masks in RLE
- Decompression of COCO's compressed RLE format to integer run-lengths, and vice versa
- Extra compression (optional) using gzip on top of the LEB128-like encoding used by the COCO API (~40% reduction beyond
  the COCO compression)
- More streamlined API

## List of functions

### Encoding / decoding (between run lengths and binary masks)
- `rle_mask = rlemasklib.encode(binary_mask)`: Encode a binary mask as RLE
- `binary_mask = rlemasklib.decode(rle_mask)`: Decode an RLE mask to a binary mask

### Compression / decompression (between compressed and uncompressed run lengths)
- `rle_mask = rlemasklib.compress(rle_mask)`: Compress an RLE mask using LEB128 (and optionally gzip)
- `rle_mask = rlemasklib.decompress(rle_mask)`: Decompress an RLE mask from LEB128 or gzip to an array of integers (run-lengths)

### Initialization
- `rle_mask = rlemasklib.empty(imshape)`: Create an empty RLE mask of given size
- `rle_mask = rlemasklib.full(imshape)`: Create a full RLE mask of given size

### Set operations
- `rle_mask = rlemasklib.intersection(rle_masks)`: Compute the intersection of multiple RLE masks.
- `rle_mask = rlemasklib.union(rle_masks)`: Compute the union of multiple RLE masks.
- `rle_mask = rlemasklib.complement(rle_mask)`: Compute the complement of an RLE mask.
- `rle_mask = rlemasklib.difference(rle_mask1, rle_mask2)`: Compute the difference of two RLE masks.
- `rle_mask = rlemasklib.symmetric_difference(rle_mask1, rle_mask2)`: Compute the symmetric difference of two RLE masks.

### Measurements
- `area = rlemasklib.area(rle_mask)`: Compute the area of an RLE mask
- `centroid = rlemasklib.centroid(rle_mask)`: Compute the centroid of an RLE mask (or multiple masks). Returns [x, y] coordinates. The centroid is the average position of the foreground pixels. 
- `iou = rlemasklib.iou(rle_masks)`: Compute the intersection-over-union of multiple (typically two) RLE masks.


### Crop / pad / shift by offset
- `rle_mask = rlemasklib.crop(rle_mask, bbox)`: Crop an RLE mask to a given bounding box, yielding a mask with smaller height and/or width.
- `rle_mask = rlemasklib.pad(rle_mask, paddings, value=0)`: Pad an RLE mask with given amount of [left, right, top, bottom] pixels with given value (0 or 1).
- `rle_mask = rlemasklib.shift(rle_mask, offset, border_value=0)`: Shift an RLE mask by a given pixel offset [dx, dy], filling the border with a given value.

### Connected components
- `rle_masks = rlemasklib.connected_components(rle_mask, connectivity=4, min_size=1)`: Extract the connected components of the foreground from an RLE mask. Connectivity can be 4 or 8. Minimum size can be set to filter out small components.
- `rle_mask = rlemasklib.largest_connected_component(rle_mask, connectivity=4)`: Returns the largest connected component of the foreground from an RLE mask. Returns None if there is no foreground.
- `rle_mask = rlemasklib.remove_small_components(rle_mask, min_size)`: Remove small connected components from the foreground of an RLE mask.
- `rle_mask = rlemasklib.fill_small_holes(rle_mask, min_size)`: Fill small holes (connected components of the background) in an RLE mask.

### Conversions (bounding box, polygon)
- `[x_start, y_start, width, height] = rlemasklib.to_bbox(rle_mask)`: Convert an RLE mask to a bounding box.
- `rle_mask = rlemasklib.from_bbox([x_start, y_start, width, height], imshape)`: Convert a bounding box to an RLE mask inside a given image size.
- `rle_mask = rlemasklib.from_polygon(polygon, imshape)`: Convert a polygon to an RLE mask inside a given image size.
