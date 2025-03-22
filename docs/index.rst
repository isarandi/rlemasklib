RLEMaskLib: Run-Length Encoded Mask Operations
==============================================

This library provides efficient run-length encoded (RLE) operations for binary masks in Python. It is designed to be fast and memory efficient, and is particularly useful for working with large datasets. The library provides an intuitive and extensive object-oriented interface as well as a simpler functional one. To achieve high efficiency, the core functionality is implemented in C, and wrapped via Cython.

RLEMaskLib is fully compatible with the COCO mask format (in the form of dictionaries) but can also work directly with runlength sequences.

The library provides many operations on masks, including:

- Set operations (and, or, xor, complement, difference) and custom boolean functions.
- Crop, pad, tile, concatenate
- Connected components extraction
- Warp (affine, perspective, lens distortion)
- Transpose, flip, rotate by multiples of 90 degrees
- Binary morphology: dilate, erode, open, close
- Determine the bounding box and the largest internal rectangle
- Convolve with arbitrary kernels
- Directly create fully foreground and fully background masks
- Decompress of COCO's compressed RLE format to integer run-lengths, and vice versa
- Extra compression (optional) using gzip on top of the LEB128-like encoding used by the COCO API (~40% reduction beyond
  the COCO compression)
- Object oriented and functional APIs.


This library originates as a fork of the COCO API's :footcite:`lin2014coco`. `pycocotools.mask` module (which was originally written by Piotr Dollár and Tsung-Yi Lin) but now mostly consists of new code.

Installation
------------

.. code-block:: bash

    pip install rlemasklib

Object-Oriented Usage
---------------------

The object-oriented API is the more recent one and is centered around the :class:`rlemasklib.RLEMask` class, which represents a single binary mask in run-length encoded form. The foreground is considered as 1s, and the background as 0s.

The :class:`rlemasklib.RLEMask` class provides a NumPy array-like interface, with many additional methods inspired in part by OpenCV.

This is the main recommended way to use the library.


Creating an RLEMask from a NumPy Array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from rlemasklib import RLEMask

    # Create a simple binary mask
    mask = np.array([
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1]
    ])

    # Convert the NumPy mask into an RLEMask
    rle_mask = RLEMask.from_array(mask)

    # Print the RLE representation
    print(rle_mask)

This creates a run-length encoded (RLE) version of the given binary mask, allowing for more efficient storage and operations.

Boolean Operations on Masks
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rlemasklib import RLEMask

    mask1 = RLEMask.from_array(np.array([
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]))

    mask2 = RLEMask.from_array(np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ]))

    union_mask = mask1 | mask2
    intersection_mask = mask1 & mask2

    union_mask = RLEMask.union([mask1, mask2])  # Any number of masks can be used

    difference_mask = mask1 - mask2
    complement = ~mask1


Slicing
~~~~~~~

To extract a subregion of the mask, you can use NumPy-style slicing:

.. code-block:: python

    sliced_mask = mask1[1:3, 1:3]
    print(np.array(sliced_mask))


A single pixel's value can also be retrieved using indexing:

.. code-block:: python

    print(mask1[1, 1])  # prints 1


To assign a value to a region of the mask, you can use NumPy-style slicing again:

.. code-block:: python

    mask = RLEMask.zeros((3, 3))
    mask[1:3, 1:3] = 1
    mask[0, 0] = 1


Morphological Operations
~~~~~~~~~~~~~~~~~~~~~~~~

Erosion and dilation are supported with different kernel sizes and connectivity options.


.. code-block:: python

    eroded_mask = mask1.erode3x3(connectivity=4)
    dilated_mask = mask1.dilate3x3(connectivity=4)
    print(np.array(eroded_mask))

Thresholded convolution is also supported.

Flipping, Rotation, Transpose
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mask can be flipped, rotated, and transposed without decoding the RLE representation.

.. code-block:: python

    flipped_mask = mask1.fliplr()
    transposed_mask = mask1.transpose()  # Equivalent to mask1.T
    rotated_mask = mask1.rot90(k=1)  # Rotate 90 degrees counterclockwise


Connected Components and Filtering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


.. code-block:: python

    mask = RLEMask.from_array(np.array([
        [0, 1, 1],
        [1, 1, 1],
        [0, 0, 0]
    ]))

    components = mask.connected_components(connectivity=4)
    for component in components:
        print(component.area())

    mask2 = mask.remove_small_components(connectivity=4, min_size=3)
    largest_comp = mask.largest_connected_component(connectivity=8)


See the full documentation of available features at :class:`rlemasklib.RLEMask`.


Functional Usage
----------------

The functional API is the original one and it works with dictionaries as in the COCO API. In this case, the masks are represented as
dictionary entries with the keys 'counts' and 'size'. The 'counts' field contains the runlengths compressed
with a COCO-style encoding (difference encoding with LEB128-like byte encoding), and the 'size' field contains the mask's dimensions (height, width).
Alternatively, the dictionary can contain the 'ucounts' field with the uncompressed runlength sequence, or 'zcounts' which is the compressed 'counts' further compressed with with gzip.

The functional API is more lightweight and is suitable for one-off operations or when the masks are already in the COCO dict format and need to be output in the same format.
The object-oriented API is better if you need to perform multiple operations on the same mask, so you can work directly with runlengths without repeatedly decoding and encoding the COCO format. To clarify, the OOP version uses runlength encoding but the runlengths themselves are not compressed with COCO's encoding unless explicitly requested.

The functional API does not support inplace operations.

Examples of the functional API and the equivalent object-oriented API are shown below.

.. code-block:: python

    import numpy as np
    import rlemasklib
    from rlemasklib import RLEMask

    mask1 = np.array([
        [0, 1, 1],
        [1, 1, 0],
        [0, 0, 1]
    ])
    mask2 = np.array([
        [1, 1, 1],
        [1, 1, 0],
        [0, 0, 1]
    ])

    rle_dict1 = rlemasklib.encode(mask1)  # functional API: array to dict directly)
    rle1 = RLEMask.from_dict(rle_dict1)  # OOP: dict to RLEMask
    rle2 = RLEMask.from_array(mask2)  # OOP: array to RLEMask
    rle_dict2 = rle2.to_dict()  # OOP: RLEMask to dict
    intersection_dict = rlemasklib.intersection([rle_dict1, rle_dict2])
    intersection_rle = RLEMask.intersection([rle1, rle2])


In some cases the functional API can be slightly faster, as it avoids the overhead of object creation and destruction. However, the object-oriented API is more flexible and provides a richer set of operations.

In other cases the object-oriented API is faster, especially when multiple operations are performed on the same mask.

See the full documentation of the functional API at :mod:`rlemasklib`.

.. toctree::
   :maxdepth: 3
   :caption: Contents


* :ref:`genindex`

.. footbibliography::