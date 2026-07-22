"""Library for manipulating masks stored in run-length-encoded format.

This library is an extended version of the pycocotools library's RLE functions, originally developed by Piotr Dollár and
Tsung-Yi Lin for the COCO dataset :footcite:`lin2014coco`.

There are two ways to use this library:

1. with the :class:`RLEMask` class, which is an object-oriented way to manipulate RLE masks (recommended)
2. with global functions, which take RLE masks in a dictionary representation, with the keys 'counts' and 'size'


"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "RLEMask",
    "encode",
    "decode",
    "compress",
    "complement",
    "decompress",
    "ones",
    "zeros",
    "ones_like",
    "zeros_like",
    "full",
    "empty",
    "any",
    "all",
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

from .oop import RLEMask


from ._functional import (
    encode,
    decode,
    compress,
    complement,
    decompress,
    ones,
    zeros,
    ones_like,
    zeros_like,
    ones as full,
    zeros as empty,
    any,
    all,
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

def _set_module_for_docs(module_name, module_globals, all_names):
    """Override ``__module__`` on exported objects so sphinx-codeautolink resolves names.

    sphinx-codeautolink uses ``__module__`` to find the docs page for a name in a code block;
    without this, e.g. ``RLEMask`` imported from ``rlemasklib.oop`` would not link to
    ``rlemasklib.RLEMask``. The true module is saved as ``_module_original_`` so that
    ``docs/conf.py``'s ``module_restored`` context manager can still resolve source-code links.

    Doing this in a function (rather than a module-level loop) keeps the loop variables out of
    the package namespace. The ``hasattr`` guard makes it idempotent across re-imports and stops
    aliased exports (``full`` is ``ones``, ``empty`` is ``zeros`` -- the same object under two
    names) from re-reading the already-patched module into ``_module_original_``.
    """
    for name in all_names:
        obj = module_globals.get(name)
        if obj is None:
            continue
        if not hasattr(obj, '_module_original_'):
            obj._module_original_ = obj.__module__  # noqa: vulture
        obj.__module__ = module_name


_set_module_for_docs(__name__, globals(), __all__)
