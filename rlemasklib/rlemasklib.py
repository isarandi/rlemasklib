import zlib

import numpy as np
import rlemasklib.rlemasklib_cython as rlemasklib_cython


# Interface for manipulating masks stored in RLE format.
#
# RLE is a simple yet efficient format for storing binary masks. RLE
# first divides a vector (or vectorized image) into a series of piecewise
# constant regions and then for each piece simply stores the length of
# that piece. For example, given M=[0 0 1 1 1 0 1] the RLE counts would
# be [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts would be [0 6 1]
# (note that the odd counts are always the numbers of zeros). Instead of
# storing the counts directly, additional compression is achieved with a
# variable bitrate representation based on a common scheme called LEB128.
#
# Compression is greatest given large piecewise constant regions.
# Specifically, the size of the RLE is proportional to the number of
# *boundaries* in M (or for an image the number of boundaries in the y
# direction). Assuming fairly simple shapes, the RLE representation is
# O(sqrt(n)) where n is number of pixels in the object. Hence space usage
# is substantially lower, especially for large simple objects (large n).
#
# Many common operations on masks can be computed directly using the RLE
# (without need for decoding). This includes computations such as area,
# union, intersection, etc. All of these operations are linear in the
# size of the RLE, in other words they are O(sqrt(n)) where n is the area
# of the object. Computing these operations on the original mask is O(n).
# Thus, using the RLE can result in substantial computational savings.
#
# To compile run "python setup.py build_ext --inplace"
#
# Based on the Microsoft COCO Toolbox version 2.0
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Modified by Istvan Sarandi, 2023.
# Licensed under the Simplified BSD License [see coco/license.txt]


def area(rleObjs):
    """Compute the foreground area for a mask or multiple masks.

    Args:
        rleObjs: either a single RLE or a list of RLEs

    Returns:
        A scalar if input was a single RLE, otherwise a list of scalars.
    """
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.area(rleObjs)
    else:
        return rlemasklib_cython.area([rleObjs])[0]


def complement(rleObjs):
    """Compute the complement of a mask or multiple masks.

    Args:
        rleObjs: either a single RLE or a list of RLEs

    Returns:
        A single RLE or a list of RLEs, depending on input type.
    """
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.complement(rleObjs)
    else:
        return rlemasklib_cython.complement([rleObjs])[0]


def encode(mask, compressed=True, zlevel=None):
    """Encode binary mask into a compressed RLE.

    Args:
        mask: a binary mask (numpy 2D array of any type, where zero is background and nonzero is foreground)
        compressed: whether to compress the RLE using the LEB128-like algorithm from COCO (and potentially zlib afterwards).
        zlevel: zlib compression level. None means no zlib compression, numbers up to 9 are increasing zlib compression
            levels and -1 is the default level in zlib. It has no effect if compressed=False.

    Returns:
        An encoded RLE object with a size field:
            size: (height, width) of the mask
        And one of the following fields:
            ucounts: uncompressed run-length counts
            counts: LEB128-like compressed run-length counts
            zcounts: zlib-compressed LEB128-like compressed run-length counts
    """
    encoded = _encode(np.asfortranarray(mask.astype(np.uint8)))

    if not compressed:
        # TODO: we could directly generate uncompressed RLEs as well
        return decompress(encoded)

    if zlevel is not None:
        return compress(encoded, zlevel=zlevel)

    return encoded


def decode(encoded_mask):
    """Decode a (potentially compressed) RLE encoded mask.

    Args:
        encoded_mask: encoded RLE object

    Returns:
        A binary mask (numpy 2D array of type uint8, where 0 is background and 1 is foreground)
    """

    if 'zcounts' in encoded_mask:
        encoded_mask = dict(
            size=encoded_mask['size'],
            counts=zlib.decompress(encoded_mask['zcounts']))

    if 'ucounts' in encoded_mask:
        return _decode_uncompressed(encoded_mask)

    return _decode(encoded_mask)


def crop(rleObjs, bbox):
    """Crop a mask or multiple masks (RLEs) by the given bounding box.
    The size of each output RLE is the same as the size of the corresponding bounding box.

    Args:
        rleObjs: either a single RLE or a list of RLEs
        bbox: either a single bounding box or a list of bounding boxes, in the format [x_start, y_start, width, height]

    Returns:
        Either a single RLE or a list of RLEs, depending on input type.
    """
    bbox = np.asanyarray(bbox, dtype=np.uint32)
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.crop(rleObjs, bbox)
    else:
        rleObjs_out = rlemasklib_cython.crop([rleObjs], bbox[np.newaxis])
        return rleObjs_out[0]


def to_bbox(rleObjs):
    """Convert an RLE mask or multiple RLE masks to a bounding box or a list of bounding boxes.

    Args:
        rleObjs: either a single RLE or a list of RLEs

    Returns:
        bbox(es): either a single bounding box or a list of bounding boxes, in the format [x_start, y_start, width, height]
    """
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.toBbox(rleObjs).astype(np.float32)
    else:
        return rlemasklib_cython.toBbox([rleObjs])[0].astype(np.float32)


def get_imshape(imshape=None, imsize=None):
    assert imshape is not None or imsize is not None
    if imshape is None:
        imshape = [imsize[1], imsize[0]]
    return imshape[:2]


def from_bbox(bbox, imshape=None, imsize=None):
    """Connvert a bounding box to an RLE mask of the given size.

    Args:
        bbox: a bounding box, in the format [x_start, y_start, width, height]
        imshape: [height, width] of the desired mask (either this or imsize must be provided)
        imsize: [width, height] of the desired mask (either this or imshape must be provided)

    Returns:
        An RLE mask.
    """
    imshape = get_imshape(imshape, imsize)
    bbox = np.asanyarray(bbox, dtype=np.float64)
    return rlemasklib_cython.frBbox(bbox[np.newaxis], imshape[0], imshape[1])[0]


def from_polygon(poly, imshape=None, imsize=None):
    """Convert a polygon to an RLE mask of the given size.

    Args:
        poly: a polygon (list of xy coordinates)
        imshape: [height, width] of the desired mask (either this or imsize must be provided)
        imsize: [width, height] of the desired mask (either this or imshape must be provided)

    Returns:
        An RLE mask.
    """
    imshape = get_imshape(imshape, imsize)
    poly = np.asanyarray(poly, dtype=np.float64)
    return rlemasklib_cython.frPoly(poly[np.newaxis], imshape[0], imshape[1])[0]


def empty(imshape=None, imsize=None):
    """Create an empty (fully background) RLE mask of the given size.

    Args:
        imshape: [height, width] of the desired mask (either this or imsize must be provided)
        imsize: [width, height] of the desired mask (either this or imshape must be provided)

    Returns:
        An empty RLE mask.
    """
    imshape = get_imshape(imshape, imsize)
    return compress({'size': imshape[:2], 'ucounts': [imshape[0] * imshape[1]]})


def full(imshape=None, imsize=None):
    """Create a full (fully foreground) RLE mask of the given size.

    Args:
        imshape: [height, width] of the desired mask (either this or imsize must be provided)
        imsize: [width, height] of the desired mask (either this or imshape must be provided)

    Returns:
        A full RLE mask.
    """
    imshape = get_imshape(imshape, imsize)
    return compress({'size': imshape[:2], 'ucounts': [0, imshape[0] * imshape[1]]})


def decompress(encoded_mask):
    """Decompress a compressed RLE mask to a decompressed RLE. Note that this does not decode the RLE into a binary mask.

    Args:
        encoded_mask:

    Returns:
        An RLE mask dictionary
           'size': [height, width]
           'ucounts': uint32 array of uncompressed run-lengths.
    """
    if 'zcounts' in encoded_mask:
        encoded_mask = dict(
            size=encoded_mask['size'],
            counts=zlib.decompress(encoded_mask['zcounts']))

    return _decompress(encoded_mask)


def compress(rle, zlevel=None):
    """Compress an RLE mask to a compressed RLE. Note that the input needs to be an RLE, not a decoded binary mask.

    Args:
        rle: a mask in RLE format
        zlevel: optional zlib compression level, None means no zlib compression, -1 is zlib's default compression level
           and 0-9 are zlib's compression levels where 9 is maximum compression.

    Returns:
        A compressed RLE mask.
    """
    if 'ucounts' in rle:
        rle = _compress(rle)

    if 'counts' in rle and zlevel is not None:
        rle['zcounts'] = zlib.compress(rle['counts'], zlevel)
        del rle['counts']

    return rle


def union(masks):
    """Compute the union of multiple RLE masks."""
    return rlemasklib_cython.merge(masks, intersect=False)


def intersection(masks):
    """Compute the intersection of multiple RLE masks."""
    return rlemasklib_cython.merge(masks, intersect=True)


def difference(mask1, mask2):
    """Compute the difference between two RLE masks, i.e., the mask where mask1 is foreground and mask2 is background."""
    return intersection([mask1, complement(mask2)])


def symmetric_difference(mask1, mask2):
    """Compute the symmetric difference between two RLE masks, i.e., the mask where either mask1 or mask2 is foreground but not both."""
    return difference(union([mask1, mask2]), intersection([mask1, mask2]))


def _compress(uncompressed_rle):
    if isinstance(uncompressed_rle, (tuple, list)):
        return rlemasklib_cython.frUncompressedRLE(uncompressed_rle)
    return rlemasklib_cython.frUncompressedRLE([uncompressed_rle])[0]


def _decompress(compressed_rle):
    if isinstance(compressed_rle, (tuple, list)):
        return rlemasklib_cython.decompress(compressed_rle)
    return rlemasklib_cython.decompress([compressed_rle])[0]


def _encode(bimask):
    if len(bimask.shape) == 3:
        return rlemasklib_cython.encode(bimask)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return rlemasklib_cython.encode(bimask.reshape((h, w, 1), order='F'))[0]


def _decode(rleObjs):
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.decode(rleObjs)
    else:
        return rlemasklib_cython.decode([rleObjs])[:, :, 0]


def _decode_uncompressed(rleObjs):
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.decodeUncompressed(rleObjs)
    else:
        return rlemasklib_cython.decodeUncompressed([rleObjs])[:, :, 0]


def iou(masks):
    union_ = area(union(masks))
    if union_ == 0:
        return 0
    intersection_ = area(intersection(masks))
    return intersection_ / union_
