"""Tests for image moments and Hu moments against cv2 oracle."""

import numpy as np
import cv2
import pytest

skimage_data = pytest.importorskip("skimage.data")

import rlemasklib


class TestMoments:
    """Test moments() method against cv2.moments()."""

    def test_circle(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 1, -1)
        self._compare_moments(mask)

    def test_polygon(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        pts = np.array(
            [[20, 10], [80, 30], [90, 80], [40, 90], [10, 50]], dtype=np.int32
        )
        cv2.fillPoly(mask, [pts], 1)
        self._compare_moments(mask)

    def test_rotated_ellipse(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.ellipse(mask, (50, 50), (40, 20), 30, 0, 360, 1, -1)
        self._compare_moments(mask)

    def test_rectangle(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 1
        self._compare_moments(mask)

    def test_coins_thresholded(self):
        coins = skimage_data.coins()
        mask = (coins > 100).astype(np.uint8)
        self._compare_moments(mask)

    def test_camera_thresholded(self):
        camera = skimage_data.camera()
        mask = (camera > 128).astype(np.uint8)
        self._compare_moments(mask)

    def test_text_thresholded(self):
        text = skimage_data.text()
        mask = (text < 100).astype(np.uint8)
        self._compare_moments(mask)

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        rle = rlemasklib.RLEMask(mask)
        m = rle.moments()
        assert m["m00"] == 0
        for key in m:
            assert m[key] == 0.0

    def test_single_pixel(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1
        self._compare_moments(mask)

    def _compare_moments(self, mask):
        cv2_m = cv2.moments(mask)
        rle = rlemasklib.RLEMask(mask)
        rle_m = rle.moments()

        for key in cv2_m:
            assert np.isclose(
                cv2_m[key], rle_m[key]
            ), f"Mismatch for {key}: cv2={cv2_m[key]}, rle={rle_m[key]}"


class TestHuMoments:
    """Test hu_moments() method against cv2.HuMoments()."""

    def test_circle(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 30, 1, -1)
        self._compare_hu_moments(mask)

    def test_polygon(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        pts = np.array(
            [[20, 10], [80, 30], [90, 80], [40, 90], [10, 50]], dtype=np.int32
        )
        cv2.fillPoly(mask, [pts], 1)
        self._compare_hu_moments(mask)

    def test_rotated_ellipse(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.ellipse(mask, (50, 50), (40, 20), 30, 0, 360, 1, -1)
        self._compare_hu_moments(mask)

    def test_rectangle(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 30:70] = 1
        self._compare_hu_moments(mask)

    def test_coins_thresholded(self):
        coins = skimage_data.coins()
        mask = (coins > 100).astype(np.uint8)
        self._compare_hu_moments(mask)

    def test_camera_thresholded(self):
        camera = skimage_data.camera()
        mask = (camera > 128).astype(np.uint8)
        self._compare_hu_moments(mask)

    def test_text_thresholded(self):
        text = skimage_data.text()
        mask = (text < 100).astype(np.uint8)
        self._compare_hu_moments(mask)

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        rle = rlemasklib.RLEMask(mask)
        hu = rle.hu_moments()
        assert hu.shape == (7,)
        assert np.all(hu == 0)

    def _compare_hu_moments(self, mask):
        cv2_m = cv2.moments(mask)
        cv2_hu = cv2.HuMoments(cv2_m).flatten()

        rle = rlemasklib.RLEMask(mask)
        rle_hu = rle.hu_moments()

        assert rle_hu.shape == (7,)
        assert np.allclose(
            cv2_hu, rle_hu
        ), f"Hu moments mismatch:\ncv2={cv2_hu}\nrle={rle_hu}"
