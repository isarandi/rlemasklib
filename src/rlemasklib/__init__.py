"""Library for manipulating masks stored in run-length-encoded format.

This library is an extended version of the pycocotools library's RLE functions, originally developed by Piotr Dollár and
Tsung-Yi Linfor the COCO dataset :footcite:`lin2014coco`.

There are two ways to use this library:

1. with global functions, which take RLE masks in a dictionary representation, with the keys 'counts' and 'size'
2. with the :class:`RLEMask` class, which is an object-oriented way to manipulate RLE masks

"""

__all__ = [
    "RLEMask",
    "encode",
    "decode",
    "compress",
    "decompress",
    "ones",
    "zeros",
    "ones_like",
    "zeros_like",
    "full",
    "empty",
    "area",
    "centroid",
    "intersection",
    "union",
    "iou",
    "difference",
    "symmetric_difference",
    "from_bbox",
    "to_bbox",
    "from_polygon",
    "crop",
    "pad",
    "connected_components",
    "largest_connected_component",
    "remove_small_components",
    "fill_small_holes",
    "erode",
    "dilate",
    "opening",
    "closing",
    "erode2",
    "dilate2",
    "opening2",
    "closing2",
    "shift",
    "BoolFunc",
    "merge",
]

from rlemasklib.oop import RLEMask


from rlemasklib.rlemasklib import (
    encode,
    decode,
    compress,
    decompress,
    ones,
    zeros,
    ones_like,
    zeros_like,
    ones as full,
    zeros as empty,
    area,
    centroid,
    intersection,
    union,
    iou,
    difference,
    symmetric_difference,
    from_bbox,
    to_bbox,
    from_polygon,
    crop,
    pad,
    connected_components,
    largest_connected_component,
    remove_small_components,
    fill_small_holes,
    erode,
    dilate,
    opening,
    closing,
    erode2,
    dilate2,
    opening2,
    closing2,
    shift,
    BoolFunc,
    merge,
)

# set the __module__ attribute of all functions/classes to this module
for x in __all__:
    obj = globals()[x]
    obj._module_original_ = obj.__module__
    obj.__module__ = __name__