"""Tests for PNG-to-RLE direct conversion."""

import io
import os
import tempfile

import numpy as np
import pytest

from rlemasklib import RLEMask

skimage_data = pytest.importorskip("skimage.data")


def _grayscale_png_bytes(image_uint8):
    """Save a uint8 grayscale image to PNG bytes via PIL."""
    PIL = pytest.importorskip("PIL")
    from PIL import Image
    img = Image.fromarray(image_uint8, mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


class TestFromPng:
    def test_roundtrip_bytes(self):
        gray = skimage_data.camera()  # 512x512 uint8
        png_bytes = _grayscale_png_bytes(gray)
        mask_from_array = RLEMask.from_array(gray)
        mask_from_png = RLEMask.from_png(data=png_bytes)
        assert mask_from_png == mask_from_array

    def test_roundtrip_file(self):
        gray = skimage_data.camera()
        png_bytes = _grayscale_png_bytes(gray)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(png_bytes)
            path = f.name
        try:
            mask_from_array = RLEMask.from_array(gray)
            mask_from_png = RLEMask.from_png(path=path)
            assert mask_from_png == mask_from_array
        finally:
            os.unlink(path)

    def test_threshold(self):
        gray = skimage_data.camera()
        png_bytes = _grayscale_png_bytes(gray)
        mask_from_array = RLEMask.from_array(gray, thresh128=True)
        mask_from_png = RLEMask.from_png(data=png_bytes, threshold=128)
        assert mask_from_png == mask_from_array

    def test_coins(self):
        gray = skimage_data.coins()
        png_bytes = _grayscale_png_bytes(gray)
        mask_from_array = RLEMask.from_array(gray)
        mask_from_png = RLEMask.from_png(data=png_bytes)
        assert mask_from_png == mask_from_array

    def test_error_no_args(self):
        with pytest.raises(ValueError):
            RLEMask.from_png()

    def test_error_both_args(self):
        with pytest.raises(ValueError):
            RLEMask.from_png(path="foo.png", data=b"\x00")


class TestFromLabelMapPng:
    def test_roundtrip_bytes(self):
        label_map = np.zeros((100, 100), dtype=np.uint8)
        label_map[10:40, 10:40] = 1
        label_map[50:80, 50:80] = 2
        label_map[20:60, 30:70] = 3

        png_bytes = _grayscale_png_bytes(label_map)
        result_array = RLEMask.from_label_map(label_map)
        result_png = RLEMask.from_label_map_png(data=png_bytes)

        assert result_array.keys() == result_png.keys()
        for label in result_array:
            assert result_array[label] == result_png[label]

    def test_roundtrip_file(self):
        label_map = np.zeros((64, 64), dtype=np.uint8)
        label_map[5:25, 5:25] = 1
        label_map[30:50, 30:50] = 2

        png_bytes = _grayscale_png_bytes(label_map)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(png_bytes)
            path = f.name
        try:
            result_array = RLEMask.from_label_map(label_map)
            result_png = RLEMask.from_label_map_png(path=path)
            assert result_array.keys() == result_png.keys()
            for label in result_array:
                assert result_array[label] == result_png[label]
        finally:
            os.unlink(path)

    def test_many_labels(self):
        rng = np.random.default_rng(42)
        label_map = rng.integers(0, 10, size=(50, 50), dtype=np.uint8)

        png_bytes = _grayscale_png_bytes(label_map)
        result_array = RLEMask.from_label_map(label_map)
        result_png = RLEMask.from_label_map_png(data=png_bytes)

        assert result_array.keys() == result_png.keys()
        for label in result_array:
            assert result_array[label] == result_png[label]
