"""Tests for the merge_to_label_map operation.

These tests are isolated to help diagnose segfault issues with merge_to_label_map.
"""

import numpy as np
from rlemasklib import RLEMask


class TestMergeToLabelMapBasic:
    """Basic merge_to_label_map functionality tests."""

    def test_basic_non_overlapping(self):
        """merge_to_label_map with non-overlapping masks."""
        m1 = RLEMask.from_array(np.array([[1, 0], [0, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0, 1], [0, 0]], dtype=np.uint8))
        m3 = RLEMask.from_array(np.array([[0, 0], [1, 1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3])
        assert labelmap[0, 0] == 1
        assert labelmap[0, 1] == 2
        assert labelmap[1, 0] == 3
        assert labelmap[1, 1] == 3

    def test_single_mask(self):
        """merge_to_label_map with single mask."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        labelmap = RLEMask.merge_to_label_map([rle])
        assert labelmap.shape == (2, 2)
        assert labelmap[0, 0] == 1
        assert labelmap[0, 1] == 0
        assert labelmap[1, 0] == 0
        assert labelmap[1, 1] == 1

    def test_two_masks(self):
        """merge_to_label_map with two masks."""
        m1 = RLEMask.from_array(np.array([[1, 1, 0], [0, 0, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0, 0, 1], [1, 1, 1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2])
        assert labelmap[0, 0] == 1
        assert labelmap[0, 1] == 1
        assert labelmap[0, 2] == 2
        assert labelmap[1, 0] == 2
        assert labelmap[1, 1] == 2
        assert labelmap[1, 2] == 2


class TestMergeToLabelMapOverlap:
    """Test merge_to_label_map with overlapping masks."""

    def test_overlapping_masks(self):
        """merge_to_label_map with overlapping masks (last wins)."""
        m1 = RLEMask.from_array(np.array([[1, 1], [1, 1]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0, 1], [1, 0]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2])
        # m2 should overwrite m1 where they overlap
        assert labelmap[0, 0] == 1  # only m1
        assert labelmap[0, 1] == 2  # both, m2 wins
        assert labelmap[1, 0] == 2  # both, m2 wins
        assert labelmap[1, 1] == 1  # only m1

    def test_fully_overlapping(self):
        """merge_to_label_map where second mask fully covers first."""
        m1 = RLEMask.from_array(np.array([[1, 1], [1, 1]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[1, 1], [1, 1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2])
        # m2 should completely overwrite m1
        assert np.all(labelmap == 2)


class TestMergeToLabelMapEdgeCases:
    """Edge cases for merge_to_label_map."""

    def test_empty_masks(self):
        """merge_to_label_map with all-zero masks."""
        m1 = RLEMask.zeros((3, 3))
        m2 = RLEMask.zeros((3, 3))
        labelmap = RLEMask.merge_to_label_map([m1, m2])
        assert labelmap.shape == (3, 3)
        assert np.all(labelmap == 0)

    def test_full_masks(self):
        """merge_to_label_map with all-ones masks."""
        m1 = RLEMask.ones((3, 3))
        m2 = RLEMask.ones((3, 3))
        labelmap = RLEMask.merge_to_label_map([m1, m2])
        # Last one wins
        assert np.all(labelmap == 2)

    def test_mixed_empty_and_full(self):
        """merge_to_label_map with mix of empty and full masks."""
        m1 = RLEMask.zeros((3, 3))
        m2 = RLEMask.ones((3, 3))
        m3 = RLEMask.zeros((3, 3))
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3])
        # Only m2 has content, so all should be labeled 2
        assert np.all(labelmap == 2)


class TestMergeToLabelMapSizes:
    """Test merge_to_label_map with various sizes."""

    def test_single_pixel_masks(self):
        """merge_to_label_map with 1x1 masks."""
        m1 = RLEMask.from_array(np.array([[1]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2])
        assert labelmap.shape == (1, 1)
        assert labelmap[0, 0] == 1

    def test_single_row_masks(self):
        """merge_to_label_map with single row masks."""
        m1 = RLEMask.from_array(np.array([[1, 0, 0, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0, 1, 0, 0]], dtype=np.uint8))
        m3 = RLEMask.from_array(np.array([[0, 0, 1, 0]], dtype=np.uint8))
        m4 = RLEMask.from_array(np.array([[0, 0, 0, 1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3, m4])
        assert labelmap.shape == (1, 4)
        np.testing.assert_array_equal(labelmap[0], [1, 2, 3, 4])

    def test_single_column_masks(self):
        """merge_to_label_map with single column masks."""
        m1 = RLEMask.from_array(np.array([[1], [0], [0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0], [1], [0]], dtype=np.uint8))
        m3 = RLEMask.from_array(np.array([[0], [0], [1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3])
        assert labelmap.shape == (3, 1)
        np.testing.assert_array_equal(labelmap[:, 0], [1, 2, 3])

    def test_large_masks(self):
        """merge_to_label_map with larger masks."""
        np.random.seed(42)
        m1 = RLEMask.from_array((np.random.rand(50, 50) > 0.7).astype(np.uint8))
        m2 = RLEMask.from_array((np.random.rand(50, 50) > 0.7).astype(np.uint8))
        m3 = RLEMask.from_array((np.random.rand(50, 50) > 0.7).astype(np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3])
        assert labelmap.shape == (50, 50)
        assert labelmap.max() <= 3
        assert labelmap.min() >= 0


class TestMergeToLabelMapManyMasks:
    """Test merge_to_label_map with many masks."""

    def test_five_masks(self):
        """merge_to_label_map with 5 masks."""
        masks = []
        for i in range(5):
            arr = np.zeros((10, 10), dtype=np.uint8)
            arr[i * 2 : (i + 1) * 2, :] = 1
            masks.append(RLEMask.from_array(arr))
        labelmap = RLEMask.merge_to_label_map(masks)
        assert labelmap.shape == (10, 10)
        for i in range(5):
            assert np.all(labelmap[i * 2 : (i + 1) * 2, :] == i + 1)

    def test_ten_masks(self):
        """merge_to_label_map with 10 masks."""
        masks = []
        for i in range(10):
            arr = np.zeros((10, 10), dtype=np.uint8)
            arr[i, :] = 1
            masks.append(RLEMask.from_array(arr))
        labelmap = RLEMask.merge_to_label_map(masks)
        assert labelmap.shape == (10, 10)
        for i in range(10):
            assert np.all(labelmap[i, :] == i + 1)


class TestMergeToLabelMapOutput:
    """Test output properties of merge_to_label_map."""

    def test_output_dtype(self):
        """Output should be uint8."""
        m1 = RLEMask.from_array(np.array([[1, 0], [0, 1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1])
        assert labelmap.dtype == np.uint8

    def test_output_is_contiguous(self):
        """Output should be contiguous array."""
        m1 = RLEMask.from_array(np.array([[1, 0], [0, 1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1])
        assert labelmap.flags["C_CONTIGUOUS"] or labelmap.flags["F_CONTIGUOUS"]

    def test_background_is_zero(self):
        """Background pixels should be 0."""
        m1 = RLEMask.from_array(np.array([[1, 0], [0, 0]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1])
        assert labelmap[0, 1] == 0
        assert labelmap[1, 0] == 0
        assert labelmap[1, 1] == 0


class TestMergeToLabelMapPatterns:
    """Test merge_to_label_map with specific patterns."""

    def test_checkerboard_split(self):
        """merge_to_label_map with checkerboard patterns."""
        m1 = RLEMask.from_array(np.array([[1, 0], [0, 1]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0, 1], [1, 0]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2])
        expected = np.array([[1, 2], [2, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(labelmap, expected)

    def test_quadrants(self):
        """merge_to_label_map with quadrant masks."""
        # Each mask covers one quadrant
        m1 = RLEMask.from_array(
            np.array(
                [[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
            )
        )
        m2 = RLEMask.from_array(
            np.array(
                [[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8
            )
        )
        m3 = RLEMask.from_array(
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]], dtype=np.uint8
            )
        )
        m4 = RLEMask.from_array(
            np.array(
                [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.uint8
            )
        )
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3, m4])
        assert np.all(labelmap[:2, :2] == 1)
        assert np.all(labelmap[:2, 2:] == 2)
        assert np.all(labelmap[2:, :2] == 3)
        assert np.all(labelmap[2:, 2:] == 4)

    def test_concentric_squares(self):
        """merge_to_label_map with concentric squares."""
        m1 = RLEMask.ones((5, 5))
        m2 = RLEMask.from_array(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 1, 1, 1, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
        )
        m3 = RLEMask.from_array(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.uint8,
            )
        )
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3])
        assert labelmap[0, 0] == 1  # outer ring
        assert labelmap[1, 1] == 2  # middle ring
        assert labelmap[2, 2] == 3  # center
