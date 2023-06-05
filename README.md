# Run-Length Encoded Mask Operations

This library is an extended and improved version of the [COCO API](https://github.com/cocodataset/cocoapi)'s `pycocotools.mask` module written by Piotr Dollár and Tsung-Yi Lin.

It offers the following additional features:

- Further set operations (complement, difference, symmetric difference) in RLE, without decoding
- Mask cropping in RLE (`rlemasklib.crop`), without decoding
- Direct creation of full and empty masks in RLE
- Decompression of COCO's compressed RLE format to integer run-lengths, and vice versa
- Extra compression (optional) using gzip on top of the LEB128-like encoding used by the COCO API (~40% reduction beyond the COCO compression)
- More streamlined API
