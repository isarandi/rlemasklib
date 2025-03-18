import zlib
from typing import Optional

import numpy as np
import rlemasklib.rlemasklib_cython as rlemasklib_cython
from rlemasklib.boolfunc import BoolFunc


# Based on the Microsoft COCO Toolbox version 2.0
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Modified by Istvan Sarandi, 2023.
# Licensed under the Simplified BSD License [see coco/license.txt]
# Original comments:
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


def encode(
        mask: np.ndarray, compressed: bool = True, zlevel: Optional[int] = None, batch_first=False) -> dict:
    """Encode binary mask into a compressed RLE.

    Args:
        mask: a binary mask (numpy 2D array of any type, where zero is background and nonzero
            is foreground)
        compressed: whether to compress the RLE using the LEB128-like algorithm from COCO (and
            potentially zlib afterwards).
        zlevel: zlib compression level. None means no zlib compression, numbers up to 9 are
            increasing zlib compression
            levels and -1 is the default level in zlib. It has no effect if compressed=False.

    Returns:
        An encoded RLE dictionary with a size key
            - ``"size"`` -- (height, width) of the mask,
        and one of the following
            - ``"ucounts"`` -- uncompressed run-length counts
            - ``"counts"`` -- LEB128-like compressed run-length counts
            - ``"zcounts"`` -- zlib-compressed LEB128-like compressed run-length counts
    """
    if mask.dtype == np.bool_:
        mask = mask.view(np.uint8)
    elif mask.dtype != np.uint8:
        mask = np.asfortranarray(mask, dtype=np.uint8)

    if mask.flags.c_contiguous and (batch_first or len(mask.shape) == 2):
        encoded = _encode_C_order(mask, compress_leb128=compressed)
    else:
        if batch_first:
            mask = mask.transpose((1, 2, 0))
        encoded = _encode(np.asfortranarray(mask, dtype=np.uint8), compress_leb128=compressed)

    if compressed and zlevel is not None:
        if isinstance(encoded, (tuple, list)):
            return [compress(m, zlevel=zlevel) for m in encoded]
        else:
            return compress(encoded, zlevel=zlevel)

    return encoded


def decode(encoded_mask: dict) -> np.ndarray:
    """Decode a (potentially compressed) RLE encoded mask.

    Args:
        encoded_mask: encoded RLE object

    Returns:
        A binary mask (numpy 2D array of type uint8, where 0 is background and 1 is foreground)
    """

    if "zcounts" in encoded_mask:
        encoded_mask = dict(
            size=encoded_mask["size"], counts=zlib.decompress(encoded_mask["zcounts"])
        )

    if "ucounts" in encoded_mask:
        return _decode_uncompressed(encoded_mask)

    return _decode(encoded_mask)


def crop(rleObjs, bbox: np.ndarray):
    """Crop a mask or multiple masks (RLEs) by the given bounding box.
    The size of each output RLE is the same as the size of the corresponding bounding box.

    Args:
        rleObjs: either a single RLE or a list of RLEs
        bbox: either a single bounding box or a list of bounding boxes, in the format
            [x_start, y_start, width, height]

    Returns:
        Either a single RLE or a list of RLEs, depending on input type.
    """
    bbox = np.asanyarray(bbox, dtype=np.uint32)
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.crop(rleObjs, bbox)
    else:
        rleObjs_out = rlemasklib_cython.crop([rleObjs], bbox[np.newaxis])
        return rleObjs_out[0]


def _pad(rleObjs, paddings):
    paddings = np.asanyarray(paddings, dtype=np.uint32)
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.pad(rleObjs, paddings)
    else:
        rleObjs_out = rlemasklib_cython.pad([rleObjs], paddings)
        return rleObjs_out[0]


def pad(rleObjs, paddings, value: int = 0):
    """Pad a mask or multiple masks (RLEs) by the given padding amounts.

    Args:
        rleObjs: either a single RLE or a list of RLEs
        paddings: left,right,top,bottom pixel amounts to pad

    Returns:
        Either a single RLE or a list of RLEs, depending on input type.
    """
    if value == 0:
        return _pad(rleObjs, paddings)
    else:
        return complement(_pad(complement(rleObjs), paddings))


def to_bbox(rleObjs):
    """Convert an RLE mask or multiple RLE masks to a bounding box or a list of bounding boxes.

    Args:
        rleObjs: either a single RLE or a list of RLEs

    Returns:
        bbox(es): either a single bounding box or a list of bounding boxes, in the format
            [x_start, y_start, width, height]
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
        An RLE mask corresponding to the input bounding box, i.e., a mask with all zeros except
            for 1s within the bounding box.
    """
    imshape = get_imshape(imshape, imsize)
    bbox = np.asanyarray(bbox, dtype=np.float64)

    if len(bbox.shape) == 2:
        return rlemasklib_cython.frBbox(bbox, imshape[0], imshape[1])
    else:
        return rlemasklib_cython.frBbox(bbox[np.newaxis], imshape[0], imshape[1])[0]


def from_polygon(poly, imshape=None, imsize=None):
    """Convert a polygon to an RLE mask of the given size.

    Args:
        poly: a polygon (list of xy coordinates)
        imshape: [height, width] of the desired mask (either this or imsize must be provided)
        imsize: [width, height] of the desired mask (either this or imshape must be provided)

    Returns:
        An RLE mask, where the polygon is filled with 1s and the rest is 0.
    """
    imshape = get_imshape(imshape, imsize)
    poly = np.asanyarray(poly, dtype=np.float64)
    return rlemasklib_cython.frPoly(poly[np.newaxis], imshape[0], imshape[1])[0]


def zeros(imshape=None, imsize=None):
    """Create an empty (fully background) RLE mask of the given size.

    Args:
        imshape: [height, width] of the desired mask (either this or imsize must be provided)
        imsize: [width, height] of the desired mask (either this or imshape must be provided)

    Returns:
        A fully-background RLE mask dictionary.
    """
    imshape = get_imshape(imshape, imsize)
    return compress({"size": imshape[:2], "ucounts": [imshape[0] * imshape[1]]})


def ones(imshape=None, imsize=None) -> dict:
    """Create a full (fully foreground) RLE mask of the given size.

    Args:
        imshape: [height, width] of the desired mask (either this or imsize must be provided)
        imsize: [width, height] of the desired mask (either this or imshape must be provided)

    Returns:
        A fully-foreground RLE mask.
    """
    imshape = get_imshape(imshape, imsize)
    return compress({"size": imshape[:2], "ucounts": [0, imshape[0] * imshape[1]]})


def ones_like(mask: dict) -> dict:
    """Create a full (fully foreground) RLE mask of the same size as the input mask.

    Args:
        mask: an RLE mask dictionary

    Returns:
        A fully-foreground RLE mask dictionary.
    """
    return ones(mask["size"])


def zeros_like(mask):
    """Create an empty (fully background) RLE mask of the same size as the input mask.

    Args:
        mask: an RLE mask dictionary

    Returns:
        A fully-background RLE mask dictionary.
    """
    return zeros(mask["size"])


def decompress(encoded_mask: dict, only_gzip: bool = False) -> dict:
    """Decompress a compressed RLE mask to a decompressed RLE.

    Note that this does not decode the RLE into a binary mask.

    Args:
        encoded_mask: an RLE mask dictionary
        only_gzip: whether to only decompress the zlib-compression, but not the LEB128-like
            compression

    Returns:
        An RLE mask dictionary
           - ``"size"`` -- [height, width]
           - ``"ucounts"`` -- uint32 array of uncompressed run-lengths.
    """
    if "zcounts" in encoded_mask:
        encoded_mask = dict(
            size=encoded_mask["size"], counts=zlib.decompress(encoded_mask["zcounts"])
        )
    if only_gzip:
        return encoded_mask

    return _decompress(encoded_mask)


def compress(rle: dict, zlevel: Optional[int] = None):
    """Compress an RLE mask to a compressed RLE.

    Note that the input needs to be an RLE, not a decoded binary mask.

    Args:
        rle: a mask in RLE format
        zlevel: optional zlib compression level, None means no zlib compression, -1 is zlib's
            default compression level and 0-9 are zlib's compression levels where 9 is maximum
            compression.

    Returns:
        A compressed RLE mask.
    """
    if "ucounts" in rle:
        rle = _compress(rle)

    if "counts" in rle and zlevel is not None:
        rle = dict(size=rle["size"], zcounts=zlib.compress(rle["counts"], zlevel))

    return rle


def union(masks):
    """Compute the union of multiple RLE masks."""
    return merge(masks, BoolFunc.UNION)


def intersection(masks):
    """Compute the intersection of multiple RLE masks."""
    return merge(masks, BoolFunc.INTERSECTION)


def difference(mask1, mask2):
    """Compute the difference between two RLE masks.

    This keeps the pixels where mask1 is foreground and mask2 is background.

    Args:
        mask1: an RLE mask dictionary
        mask2: an RLE mask dictionary

    Returns:
        An RLE mask dictionary of the difference, i.e., the mask of pixels where mask1 is
            foreground and mask2 is

    """
    return merge([mask1, mask2], BoolFunc.DIFFERENCE)


def any(mask: dict) -> bool:
    """Check if any of the pixels in the mask are foreground.

    Args:
        mask: an RLE mask dictionary

    Returns:
        True if any of the pixels are foreground, False otherwise.
    """
    return len(mask["counts"]) > 1


def all(mask):
    """Check if all pixels in the mask are foreground.

    Args:
        mask: an RLE mask dictionary

    Returns:
        True if all pixels are foreground, False otherwise.
    """
    h, w = mask["size"]
    if h * w == 0:
        return True

    return len(mask["counts"]) == 2 and mask["counts"][0] == b"\x00"


def symmetric_difference(mask1, mask2):
    """Compute the symmetric difference between two RLE masks.

    Args:
        mask1: an RLE mask dictionary
        mask2: an RLE mask dictionary

    Returns:
        An RLE mask dictionary of the symmetric difference, i.e., the mask of pixels where either
            mask1 or mask2 is foreground but not both.
    """

    return merge([mask1, mask2], BoolFunc.SYMMETRIC_DIFFERENCE)


def merge(masks, boolfunc: BoolFunc):
    """Merge multiple RLE masks using a Boolean function.

    The masks will be pairwise merged (reduced) from left to right, in the style of "reduce"
    (or foldl) from functional programming.

    Args:
        masks: a list of RLE masks
        boolfunc: a Boolean function to apply to the masks

    Returns:
        An RLE mask dictionary of the merged masks.
    """
    return rlemasklib_cython.merge(masks, boolfunc.value)


def _compress(uncompressed_rle):
    if isinstance(uncompressed_rle, (tuple, list)):
        return rlemasklib_cython.frUncompressedRLE(uncompressed_rle)
    return rlemasklib_cython.frUncompressedRLE([uncompressed_rle])[0]


def _decompress(compressed_rle):
    if isinstance(compressed_rle, (tuple, list)):
        return rlemasklib_cython.decompress(compressed_rle)
    return rlemasklib_cython.decompress([compressed_rle])[0]


def _encode(bimask, compress_leb128=True):
    if len(bimask.shape) == 3:
        return rlemasklib_cython.encode(bimask, compress_leb128)
    elif len(bimask.shape) == 2:
        h, w = bimask.shape
        return rlemasklib_cython.encode(bimask.reshape((h, w, 1), order="F"), compress_leb128)[0]


def _encode_C_order(bimask, compress_leb128=True):
    if len(bimask.shape) == 3:
        return rlemasklib_cython.encode_C_order_sparse(bimask, compress_leb128)
    elif len(bimask.shape) == 2:
        return rlemasklib_cython.encode_C_order_sparse(bimask[np.newaxis], compress_leb128)[0]


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
    """Compute the intersection-over-union (IoU) between the input masks.

    This is typically used with two input masks, but more are also supported, in which case the
    IoU as the ratio between the overall intersection and the overall union.

    Args:
        masks: a list of RLE masks

    Returns:
        A scalar IoU value, expressing the ratio of the intersection area to the union area of
            the masks.
    """
    return rlemasklib_cython.iouMulti(masks)


def connected_components(rle: dict, connectivity: int = 4, min_size: int = 1) -> list[dict]:
    """Compute the connected components of a mask.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the connected components. 4 means only
            horizontal and vertical connections are considered, while 8 means also diagonal
            connections are considered.
        min_size: the minimum size of the connected components to keep. Smaller components will be
            ignored.

    Returns:
        A list of RLE masks, each representing a connected component.
    """
    return rlemasklib_cython.connectedComponents(rle, connectivity, min_size)


def shift(rle: dict, offset: tuple[int, int], border_value: int = 0) -> dict:
    """Shift a mask by the given offset.

    Args:
        rle: an RLE mask dictionary
        offset: a tuple of (y, x) pixel offset
        border_value: the value to fill the border with (0 or 1)

    Returns:
        An RLE mask dictionary of the shifted mask.
    """
    if offset == (0, 0):
        return rle
    h, w = rle["size"]
    paddings = np.maximum(0, np.array([offset[0], -offset[0], offset[1], -offset[1]]))
    cropbox = np.maximum(0, np.array([-offset[0], -offset[1], w, h]))
    return crop(pad(rle, paddings, border_value), cropbox)


def erode(rle: dict, connectivity: int = 4) -> dict:
    """Erode a mask with a 3x3 kernel.

    After erosion, only those pixels remain foreground that were foreground before and all its
        neighbors (according to the specified connectivity, 4-way or 8-way) are also foreground.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the erosion. 4 means a cross-shaped
            kernel, 8 means a square kernel.

    Returns:
        An RLE mask dictionary of the eroded mask.
    """
    return complement(dilate(complement(rle), connectivity))


def dilate(rle: dict, connectivity: int = 4) -> dict:
    """Dilate a mask with a 3x3 kernel.

    After dilation, all pixels that were foreground before remain foreground and additionally any
    pixel with at least one foreground neighbor (according to the specified connectivity, 4-way or
    8-way) becomes also foreground.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the dilation. 4 means a cross-shaped
            kernel, 8 means a square kernel.

    Returns:
        An RLE mask dictionary of the dilated mask.
    """
    if connectivity == 4:
        neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    else:
        neighbor_offsets = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    return union([rle] + [shift(rle, offset) for offset in neighbor_offsets])


def opening(rle: dict, connectivity: int = 4) -> dict:
    """Compute the opening of a mask.

    The opening is defined as the dilation of the erosion.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the opening. 4 means a cross-shaped
            kernel, 8 means a square kernel.

    Returns:
        An RLE mask dictionary of the opened mask.
    """
    return dilate(erode(rle, connectivity), connectivity)


def closing(rle: dict, connectivity: int = 4) -> dict:
    """Compute the closing of a mask.

    The closing is defined as the erosion of the dilation.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the closing. 4 means a cross-shaped
            kernel, 8 means a square kernel.

    Returns:
        An RLE mask dictionary of the closed mask.
    """
    return erode(dilate(rle, connectivity), connectivity)


def erode2(rle: dict) -> dict:
    """Erode a mask with a round 5x5 kernel.

    The kernel is 0 in the four corners, otherwise 1.

    ::

        0 1 1 1 0
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        0 1 1 1 0


    Args:
        rle: an RLE mask dictionary

    Returns:
        An RLE mask dictionary of the eroded mask.
    """

    return complement(dilate2(complement(rle)))


def dilate2(rle: dict) -> dict:
    """Dilate a mask with a round 5x5 kernel.

    The kernel is 0 in the four corners, otherwise 1.

    ::

        0 1 1 1 0
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        0 1 1 1 0


    Args:
        rle: an RLE mask dictionary

    Returns:
        An RLE mask dictionary of the dilated mask.
    """
    return dilate(dilate(rle, 4), 8)


def opening2(rle: dict) -> dict:
    """Compute the opening of a mask with a round 5x5 kernel.

    The kernel is 0 in the four corners, otherwise 1.

    ::

        0 1 1 1 0
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        0 1 1 1 0


    The opening is defined as the dilation of the erosion.

    Args:
        rle: an RLE mask dictionary

    Returns:
        An RLE mask dictionary of the opened mask.
    """

    return dilate2(erode2(rle))


def closing2(rle: dict) -> dict:
    """Compute the closing of a mask with a round 5x5 kernel.

    The kernel is 0 in the four corners, otherwise 1.

    ::

        0 1 1 1 0
        1 1 1 1 1
        1 1 1 1 1
        1 1 1 1 1
        0 1 1 1 0


    The closing is defined as the erosion of the dilation.

    Args:
        rle: an RLE mask dictionary

    Returns:
        An RLE mask dictionary of the closed mask.

    """

    return erode2(dilate2(rle))


def remove_small_components(rle: dict, connectivity: int = 4, min_size: int = 1) -> dict:
    """Remove small connected components from a mask.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the connected components. 4 means only
            horizontal and vertical connections are considered, while 8 means also diagonal
            connections are considered.
        min_size: the minimum size of the connected components to keep. Smaller components will be
            removed (set to 0).

    Returns:
        An RLE mask dictionary of the mask with small connected components removed.
    """
    components = connected_components(rle, connectivity, min_size)
    return union(components)


def fill_small_holes(rle: dict, connectivity: int = 4, min_size: int = 1) -> dict:
    """Fill small holes in a mask.

    Holes are defined as connected components of the background.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the connected components. 4 means only
            horizontal and vertical connections are considered, while 8 means also diagonal
            connections are considered.
        min_size: the minimum size of the holes to keep. Smaller holes will be filled (set to 1).

    Returns:
        An RLE mask dictionary of the mask with small holes filled.
    """
    return complement(remove_small_components(complement(rle), connectivity, min_size))


def largest_connected_component(rle: dict, connectivity=4) -> Optional[dict]:
    """Return the largest connected component.

    Args:
        rle: an RLE mask dictionary
        connectivity: either 4 or 8, the connectivity of the connected components. 4 means only
            horizontal and vertical connections are considered, while 8 means also diagonal
            connections are considered.

    Returns:
        An RLE mask dictionary of the largest connected component, or None if the mask has no
            foreground pixels.
    """
    components = connected_components(rle, connectivity)
    if not components:
        return None
    areas = area(components)
    return components[np.argmax(areas)]


def centroid(rleObjs):
    """Compute the foreground centroid for a mask or multiple masks.

    Args:
        rleObjs: either a single RLE or a list of RLEs

    Returns:
        A scalar if input was a single RLE, otherwise a list of scalars.
    """
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.centroid(rleObjs).astype(np.float32)
    else:
        return rlemasklib_cython.centroid([rleObjs])[0].astype(np.float32)
