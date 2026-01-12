"""Tests for the tile operation.

These tests are isolated to help diagnose issues with tile producing
incorrect shapes or content.
"""

import numpy as np
from rlemasklib import RLEMask


class TestTileBasic:
    """Basic tile functionality tests."""

    def test_tile_2x2(self):
        """Tile a 2x2 mask 2x3 times."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 3)
        assert tiled.shape == (4, 6), f"Expected (4, 6), got {tiled.shape}"

    def test_tile_content(self):
        """Verify tile content matches np.tile."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 3)
        expected = np.tile(mask, (2, 3))
        np.testing.assert_array_equal(np.array(tiled), expected)

    def test_tile_1x1(self):
        """Tile 1x1 should return identical mask."""
        mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(1, 1)
        assert tiled.shape == mask.shape
        np.testing.assert_array_equal(np.array(tiled), mask)

    def test_tile_horizontal_only(self):
        """Tile only horizontally (num_h=1)."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(1, 3)
        assert tiled.shape == (2, 6), f"Expected (2, 6), got {tiled.shape}"
        expected = np.tile(mask, (1, 3))
        np.testing.assert_array_equal(np.array(tiled), expected)

    def test_tile_vertical_only(self):
        """Tile only vertically (num_w=1)."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(3, 1)
        assert tiled.shape == (6, 2), f"Expected (6, 2), got {tiled.shape}"
        expected = np.tile(mask, (3, 1))
        np.testing.assert_array_equal(np.array(tiled), expected)


class TestTileSizes:
    """Test tile with various input sizes."""

    def test_tile_single_pixel(self):
        """Tile a single pixel mask."""
        mask = np.array([[1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(5, 7)
        assert tiled.shape == (5, 7), f"Expected (5, 7), got {tiled.shape}"
        # Should be all ones
        assert np.array(tiled).sum() == 35

    def test_tile_single_row(self):
        """Tile a single row."""
        mask = np.array([[1, 0, 1, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(3, 2)
        assert tiled.shape == (3, 8), f"Expected (3, 8), got {tiled.shape}"
        expected = np.tile(mask, (3, 2))
        np.testing.assert_array_equal(np.array(tiled), expected)

    def test_tile_single_column(self):
        """Tile a single column."""
        mask = np.array([[1], [0], [1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 4)
        assert tiled.shape == (6, 4), f"Expected (6, 4), got {tiled.shape}"
        expected = np.tile(mask, (2, 4))
        np.testing.assert_array_equal(np.array(tiled), expected)

    def test_tile_large(self):
        """Tile to a larger size."""
        mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(10, 10)
        assert tiled.shape == (30, 30), f"Expected (30, 30), got {tiled.shape}"
        expected = np.tile(mask, (10, 10))
        np.testing.assert_array_equal(np.array(tiled), expected)


class TestTileEdgeCases:
    """Edge cases for tile."""

    def test_tile_empty_mask(self):
        """Tile an all-zeros mask."""
        mask = np.zeros((3, 3), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 2)
        assert tiled.shape == (6, 6)
        assert np.array(tiled).sum() == 0

    def test_tile_full_mask(self):
        """Tile an all-ones mask."""
        mask = np.ones((3, 3), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 2)
        assert tiled.shape == (6, 6)
        assert np.array(tiled).sum() == 36

    def test_tile_asymmetric(self):
        """Tile an asymmetric mask."""
        mask = np.array([[1, 1, 0], [0, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(3, 4)
        assert tiled.shape == (6, 12), f"Expected (6, 12), got {tiled.shape}"
        expected = np.tile(mask, (3, 4))
        np.testing.assert_array_equal(np.array(tiled), expected)

    def test_tile_rectangular_input(self):
        """Tile a rectangular (non-square) input."""
        mask = np.array([[1, 0, 1, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(4, 2)
        assert tiled.shape == (4, 10), f"Expected (4, 10), got {tiled.shape}"


class TestTilePatterns:
    """Verify specific patterns are preserved after tiling."""

    def test_tile_checkerboard(self):
        """Tile a checkerboard pattern."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(3, 3)
        expected = np.tile(mask, (3, 3))
        np.testing.assert_array_equal(np.array(tiled), expected)
        # Verify checkerboard property
        arr = np.array(tiled)
        assert arr[0, 0] == 1
        assert arr[0, 1] == 0
        assert arr[1, 0] == 0
        assert arr[1, 1] == 1

    def test_tile_stripes_horizontal(self):
        """Tile horizontal stripes."""
        mask = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 2)
        expected = np.tile(mask, (2, 2))
        np.testing.assert_array_equal(np.array(tiled), expected)

    def test_tile_stripes_vertical(self):
        """Tile vertical stripes."""
        mask = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 2)
        expected = np.tile(mask, (2, 2))
        np.testing.assert_array_equal(np.array(tiled), expected)
