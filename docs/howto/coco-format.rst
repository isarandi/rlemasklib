Work with COCO Format
=====================

COCO datasets store masks as compressed RLE dictionaries.
Here's how to read, write, and convert them.

COCO RLE format
---------------

A COCO mask dict has two keys::

    {
        "size": [height, width],
        "counts": b"..."  # compressed RLE bytes
    }

Load from COCO dict
-------------------

::

    from rlemasklib import RLEMask

    coco_dict = annotation["segmentation"]
    mask = RLEMask.from_dict(coco_dict)

Save to COCO dict
-----------------

::

    coco_dict = mask.to_dict()
    # {"size": [h, w], "counts": b"..."}

Functional API
--------------

For batch operations without creating objects::

    import rlemasklib

    # Encode numpy array to COCO dict
    coco_dict = rlemasklib.encode(binary_array)

    # Decode COCO dict to numpy array
    array = rlemasklib.decode(coco_dict)

    # Operations on dicts
    union_dict = rlemasklib.merge([dict1, dict2], intersect=False)
    intersection_dict = rlemasklib.merge([dict1, dict2], intersect=True)

Uncompressed counts
-------------------

Sometimes you need raw run lengths instead of the compressed string::

    # Get uncompressed run lengths
    counts = mask.counts  # numpy array of run lengths

    # Create from run lengths
    mask = RLEMask.from_counts(counts, shape=(height, width))

Extra compression
-----------------

For even smaller storage, gzip the compressed counts::

    # Returns dict with "zcounts" instead of "counts" (~40% smaller)
    compressed_dict = mask.to_dict(compressed=True)

    # Loading auto-detects the format
    mask = RLEMask.from_dict(compressed_dict)

