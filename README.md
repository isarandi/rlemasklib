# RLEMaskLib: Run-Length Encoded Mask Operations
[![Read the Docs](https://img.shields.io/readthedocs/rlemasklib)](https://rlemasklib.readthedocs.io/)
[![PyPI - Version](https://img.shields.io/pypi/v/rlemasklib)](https://pypi.org/project/rlemasklib/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Fast run-length encoded (RLE) binary mask operations in Python. Core implemented in C via Cython for efficiency.

The API is designed to feel like NumPy and OpenCV, with familiar slicing, boolean operators, and method names.

Fully compatible with the COCO mask format.

## Installation

```bash
pip install rlemasklib
```

## Quick Start

```python
import numpy as np
from rlemasklib import RLEMask

# Create from numpy array
mask = RLEMask(np.array([
    [0, 1, 1],
    [1, 1, 0],
    [0, 0, 1]
]))

# Boolean operations
union = mask1 | mask2
intersection = mask1 & mask2
difference = mask1 - mask2
complement = ~mask

# Morphology
eroded = mask.erode3x3()
dilated = mask.dilate3x3()

# Geometric transforms (NumPy/OpenCV style)
flipped = mask.fliplr()
rotated = mask.rot90()
cropped = mask[10:50, 20:80]  # slicing like numpy
resized = mask.resize((256, 256))

# Analysis
area = mask.area()
bbox = mask.bbox()
components = mask.connected_components()

# COCO format I/O
coco_dict = mask.to_dict()  # {'size': [h, w], 'counts': b'...'}
mask = RLEMask.from_dict(coco_dict)
```

## Features

- **Boolean operations**: union, intersection, difference, complement, XOR, custom functions
- **Morphology**: erode, dilate, open, close (3x3, 5x5, arbitrary kernels)
- **Geometric**: crop, pad, tile, flip, transpose, rotate, resize, warp (affine/perspective)
- **Analysis**: area, bounding box, connected components, largest interior rectangle, IoU
- **I/O**: COCO format, numpy arrays, polygons, bounding boxes, circles
- **Compression**: LEB128 (COCO-compatible) + optional gzip (~40% smaller)

## APIs

**Object-oriented** (recommended): Work with `RLEMask` objects for chained operations.

**Functional** (legacy): Dict-to-dict operations, compatible with COCO's `pycocotools.mask`.

## Documentation

Full documentation with visual examples: [rlemasklib.readthedocs.io](https://rlemasklib.readthedocs.io/)

## Origin

Fork of [COCO API](https://github.com/cocodataset/cocoapi)'s `pycocotools.mask` (by Piotr Dollár and Tsung-Yi Lin), now mostly new code.
