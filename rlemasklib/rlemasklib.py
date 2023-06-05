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
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.area(rleObjs)
    else:
        return rlemasklib_cython.area([rleObjs])[0]


def complement(rleObjs):
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.complement(rleObjs)
    else:
        return rlemasklib_cython.complement([rleObjs])[0]


def encode(mask, zlevel=None):
    encoded = _encode(np.asfortranarray(mask.astype(np.uint8)))
    if zlevel is not None:
        return compress(encoded, zlevel=zlevel)
    return encoded


def decode(encoded_mask):
    if 'zcounts' in encoded_mask:
        encoded_mask = dict(
            size=encoded_mask['size'],
            counts=zlib.decompress(encoded_mask['zcounts']))

    if 'ucounts' in encoded_mask:
        return _decode_uncompressed(encoded_mask)

    return _decode(encoded_mask)


def crop(rleObjs, bbox):
    bbox = np.asanyarray(bbox, dtype=np.uint32)
    if isinstance(rleObjs, (tuple, list)):
        return rlemasklib_cython.crop(rleObjs, bbox)
    else:
        rleObjs_out = rlemasklib_cython.crop([rleObjs], bbox[np.newaxis])
        return rleObjs_out[0]


def to_bbox(rleObjs):
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
    imshape = get_imshape(imshape, imsize)
    bbox = np.asanyarray(bbox, dtype=np.float64)
    return rlemasklib_cython.frBbox(bbox[np.newaxis], imshape[0], imshape[1])[0]


def from_polygon(poly, imshape=None, imsize=None):
    imshape = get_imshape(imshape, imsize)
    poly = np.asanyarray(poly, dtype=np.float64)
    return rlemasklib_cython.frPoly(poly[np.newaxis], imshape[0], imshape[1])[0]


def empty(imshape=None, imsize=None):
    imshape = get_imshape(imshape, imsize)
    return compress({'size': imshape[:2], 'ucounts': [imshape[0] * imshape[1]]})


def full(imshape=None, imsize=None):
    imshape = get_imshape(imshape, imsize)
    return compress({'size': imshape[:2], 'ucounts': [0, imshape[0] * imshape[1]]})


def decompress(encoded_mask):
    if 'zcounts' in encoded_mask:
        encoded_mask = dict(
            size=encoded_mask['size'],
            counts=zlib.decompress(encoded_mask['zcounts']))

    return _decompress(encoded_mask)


def compress(rle, zlevel=None):
    if 'ucounts' in rle:
        rle = _compress(rle)

    if 'counts' in rle and zlevel is not None:
        rle['zcounts'] = zlib.compress(rle['counts'], zlevel)
        del rle['counts']

    return rle


def union(masks):
    return rlemasklib_cython.merge(masks, intersect=False)


def intersection(masks):
    return rlemasklib_cython.merge(masks, intersect=True)


def difference(mask1, mask2):
    return intersection([mask1, complement(mask2)])


def symmetric_difference(mask1, mask2):
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
