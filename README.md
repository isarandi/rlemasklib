# RLEMaskLib: Run-Length Encoded Mask Operations


This library provides efficient run-length encoded (RLE) operations for binary masks in Python. It is designed to be fast and memory efficient, and is particularly useful for working with large datasets. The library provides an intuitive and extensive object-oriented interface as well as a simpler functional one. To achieve high efficiency, the core functionality is implemented in C, and wrapped via Cython.

RLEMaskLib is fully compatible with the COCO mask format (in the form of dictionaries) but can also work directly with runlength sequences.

The library provides many operations on masks, including:

- Set operations (complement, difference, symmetric difference) and custom boolean functions.
- Crop, pad, tile, concatenate
- Connected components extraction
- Warp (affine, perspective, lens distortion)
- Transpose, flip, rotate by multiples of 90 degrees
- Binary morphology: dilate, erode, open, close
- Convolve with arbitrary kernels
- Directly create fully foreground and fully background masks
- Decompress of COCO's compressed RLE format to integer run-lengths, and vice versa
- Extra compression (optional) using gzip on top of the LEB128-like encoding used by the COCO API (~40% reduction beyond
  the COCO compression)
- Object-oriented and functional APIs.


This library originates as a fork of the [COCO API](https://github.com/cocodataset/cocoapi)'s `pycocotools.mask` module (which was originally written by Piotr Dollár and Tsung-Yi Lin) but now mostly consists of new code.


## Installation

The library can be installed with pip from GitHub:

```bash
pip install git+https://github.com/isarandi/rlemasklib.git
```

## Documentation

The documentation can be found at [https://istvansarandi.com/docs/rlemasklib/](https://istvansarandi.com/docs/rlemasklib/).
