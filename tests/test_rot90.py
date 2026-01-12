"""Tests for the rot90 operation.

These tests are isolated to help diagnose segfault issues with rot90.
"""

import numpy as np
from rlemasklib import RLEMask


class TestRot90Basic:
    """Basic rot90 functionality tests."""

    def test_rot90_k0(self):
        """rot90 with k=0 should return original."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=0)
        np.testing.assert_array_equal(np.array(rotated), mask)

    def test_rot90_k1(self):
        """rot90 with k=1 should rotate 90 degrees CCW."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=1)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=1))

    def test_rot90_k2(self):
        """rot90 with k=2 should rotate 180 degrees."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=2)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=2))

    def test_rot90_k3(self):
        """rot90 with k=3 should rotate 270 degrees CCW."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=3)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=3))

    def test_rot90_k4(self):
        """rot90 with k=4 should be same as k=0."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=4)
        np.testing.assert_array_equal(np.array(rotated), mask)


class TestRot90Negative:
    """Test rot90 with negative k values."""

    def test_rot90_k_neg1(self):
        """rot90 with k=-1 should rotate 90 degrees CW."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=-1)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=-1))

    def test_rot90_k_neg2(self):
        """rot90 with k=-2 should be same as k=2."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=-2)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=-2))

    def test_rot90_k_neg3(self):
        """rot90 with k=-3 should be same as k=1."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=-3)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=-3))


class TestRot90Sizes:
    """Test rot90 with various input sizes."""

    def test_rot90_single_pixel(self):
        """rot90 of single pixel."""
        mask = np.array([[1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        for k in range(4):
            rotated = rle.rot90(k=k)
            assert rotated.shape == (1, 1)
            assert np.array(rotated)[0, 0] == 1

    def test_rot90_single_row(self):
        """rot90 of single row."""
        mask = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=1)
        assert rotated.shape == (4, 1)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=1))

    def test_rot90_single_column(self):
        """rot90 of single column."""
        mask = np.array([[1], [0], [1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=1)
        assert rotated.shape == (1, 3)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=1))

    def test_rot90_rectangular(self):
        """rot90 of rectangular (non-square) mask."""
        mask = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        for k in range(4):
            rotated = rle.rot90(k=k)
            expected = np.rot90(mask, k=k)
            assert (
                rotated.shape == expected.shape
            ), f"k={k}: Expected {expected.shape}, got {rotated.shape}"
            np.testing.assert_array_equal(np.array(rotated), expected)

    def test_rot90_large(self):
        """rot90 of larger mask."""
        np.random.seed(42)
        mask = (np.random.rand(50, 80) > 0.5).astype(np.uint8)
        rle = RLEMask.from_array(mask)
        for k in range(4):
            rotated = rle.rot90(k=k)
            expected = np.rot90(mask, k=k)
            assert rotated.shape == expected.shape
            np.testing.assert_array_equal(np.array(rotated), expected)


class TestRot90EdgeCases:
    """Edge cases for rot90."""

    def test_rot90_empty_mask(self):
        """rot90 of all-zeros mask."""
        mask = np.zeros((5, 7), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        for k in range(4):
            rotated = rle.rot90(k=k)
            expected = np.rot90(mask, k=k)
            assert rotated.shape == expected.shape
            assert np.array(rotated).sum() == 0

    def test_rot90_full_mask(self):
        """rot90 of all-ones mask."""
        mask = np.ones((5, 7), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        for k in range(4):
            rotated = rle.rot90(k=k)
            expected = np.rot90(mask, k=k)
            assert rotated.shape == expected.shape
            assert np.array(rotated).sum() == 35


class TestRot90Inplace:
    """Test rot90 with inplace=True."""

    def test_rot90_inplace_k1(self):
        """rot90 inplace with k=1."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        result = rle.rot90(k=1, inplace=True)
        assert result is rle
        np.testing.assert_array_equal(np.array(rle), np.rot90(mask, k=1))

    def test_rot90_inplace_k2(self):
        """rot90 inplace with k=2."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        result = rle.rot90(k=2, inplace=True)
        assert result is rle
        np.testing.assert_array_equal(np.array(rle), np.rot90(mask, k=2))

    def test_rot90_inplace_rectangular(self):
        """rot90 inplace of rectangular mask."""
        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rle.rot90(k=1, inplace=True)
        np.testing.assert_array_equal(np.array(rle), np.rot90(mask, k=1))


class TestRot90Sequential:
    """Test sequential rot90 operations."""

    def test_rot90_four_times(self):
        """Four 90-degree rotations should return to original."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=1).rot90(k=1).rot90(k=1).rot90(k=1)
        np.testing.assert_array_equal(np.array(rotated), mask)

    def test_rot90_k1_then_k3(self):
        """k=1 followed by k=3 should return to original."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=1).rot90(k=3)
        np.testing.assert_array_equal(np.array(rotated), mask)

    def test_rot90_k2_twice(self):
        """Two 180-degree rotations should return to original."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        rotated = rle.rot90(k=2).rot90(k=2)
        np.testing.assert_array_equal(np.array(rotated), mask)


class TestRot90Patterns:
    """Test rot90 with specific patterns."""

    def test_rot90_l_shape(self):
        """rot90 of L-shaped pattern."""
        mask = np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        for k in range(4):
            rotated = rle.rot90(k=k)
            expected = np.rot90(mask, k=k)
            np.testing.assert_array_equal(np.array(rotated), expected)

    def test_rot90_diagonal(self):
        """rot90 of diagonal pattern."""
        mask = np.eye(5, dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        for k in range(4):
            rotated = rle.rot90(k=k)
            expected = np.rot90(mask, k=k)
            np.testing.assert_array_equal(np.array(rotated), expected)

    def test_rot90_corner(self):
        """rot90 with single corner pixel."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        rle = RLEMask.from_array(mask)

        rot1 = rle.rot90(k=1)
        assert np.array(rot1)[4, 0] == 1

        rot2 = rle.rot90(k=2)
        assert np.array(rot2)[4, 4] == 1

        rot3 = rle.rot90(k=3)
        assert np.array(rot3)[0, 4] == 1
