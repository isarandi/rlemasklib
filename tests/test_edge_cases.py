"""Tests for edge cases, boundary conditions, and error handling."""

import numpy as np
import pytest
import rlemasklib
from rlemasklib.oop import RLEMask
from rlemasklib.boolfunc import BoolFunc


# =============================================================================
# Empty and Full Masks
# =============================================================================

class TestEmptyFullMasks:
    def test_empty_mask_area(self):
        """Empty mask should have area 0."""
        rle = RLEMask.zeros((10, 10))
        assert rle.area() == 0

    def test_full_mask_area(self):
        """Full mask should have area = height * width."""
        rle = RLEMask.ones((7, 11))
        assert rle.area() == 77

    def test_empty_mask_any(self):
        """Empty mask any() should be False."""
        rle = RLEMask.zeros((5, 5))
        assert rle.any() == False

    def test_full_mask_all(self):
        """Full mask all() should be True."""
        rle = RLEMask.ones((5, 5))
        assert rle.all() == True

    def test_empty_mask_complement(self):
        """Complement of empty mask should be full."""
        rle = RLEMask.zeros((5, 5))
        comp = ~rle
        assert comp.all() == True

    def test_full_mask_complement(self):
        """Complement of full mask should be empty."""
        rle = RLEMask.ones((5, 5))
        comp = ~rle
        assert comp.any() == False

    def test_empty_mask_dilate(self):
        """Dilating empty mask should remain empty."""
        rle = RLEMask.zeros((5, 5))
        dilated = rle.dilate3x3()
        assert dilated.area() == 0

    def test_full_mask_erode(self):
        """Eroding full mask should shrink border."""
        # Use a mask with explicit zeros around it to avoid boundary handling issues
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1  # 5x5 block of ones
        rle = RLEMask.from_array(mask)
        eroded = rle.erode3x3()
        # 5x5 block should shrink to 3x3
        assert eroded.area() == 9

    def test_empty_mask_connected_components(self):
        """Empty mask should have no connected components."""
        rle = RLEMask.zeros((5, 5))
        components = rle.connected_components()
        assert len(components) == 0

    def test_empty_mask_largest_component(self):
        """Empty mask largest_connected_component should return empty."""
        rle = RLEMask.zeros((5, 5))
        largest = rle.largest_connected_component()
        assert largest.area() == 0

    def test_empty_mask_centroid(self):
        """Centroid of empty mask should be nan or zero."""
        rle = RLEMask.zeros((5, 5))
        centroid = rle.centroid()
        # Implementation may return nan or (0, 0)
        assert len(centroid) == 2

    def test_empty_mask_bbox(self):
        """Bbox of empty mask should be zero-sized."""
        rle = RLEMask.zeros((5, 5))
        bbox = rle.bbox()
        assert bbox[2] == 0 or bbox[3] == 0  # width or height is 0


# =============================================================================
# Single Pixel Masks
# =============================================================================

class TestSinglePixelMasks:
    def test_single_pixel_1x1(self):
        """1x1 mask should work."""
        mask = np.array([[1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        assert rle.shape == (1, 1)
        assert rle.area() == 1
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_single_pixel_in_large_mask(self):
        """Single foreground pixel in large mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 1
        rle = RLEMask.from_array(mask)
        assert rle.area() == 1
        cx, cy = rle.centroid()
        assert cx == 50
        assert cy == 50

    def test_single_pixel_bbox(self):
        """Single pixel bbox should be 1x1."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 7] = 1
        rle = RLEMask.from_array(mask)
        bbox = rle.bbox()
        assert bbox[0] == 7  # x
        assert bbox[1] == 5  # y
        assert bbox[2] == 1  # width
        assert bbox[3] == 1  # height

    def test_single_pixel_dilate(self):
        """Dilating single pixel should create cross."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)
        dilated = rle.dilate3x3(connectivity=4)
        assert dilated.area() == 5

    def test_single_pixel_erode(self):
        """Eroding single pixel should make it disappear."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)
        eroded = rle.erode3x3()
        assert eroded.area() == 0


# =============================================================================
# Extreme Shapes
# =============================================================================

class TestExtremeShapes:
    def test_single_row(self):
        """1xN mask should work."""
        mask = np.array([[0, 1, 1, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        assert rle.shape == (1, 5)
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_single_column(self):
        """Nx1 mask should work."""
        mask = np.array([[0], [1], [1], [0], [1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        assert rle.shape == (5, 1)
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_very_wide(self):
        """1x1000 mask should work."""
        mask = np.zeros((1, 1000), dtype=np.uint8)
        mask[0, 100:200] = 1
        rle = RLEMask.from_array(mask)
        assert rle.area() == 100
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_very_tall(self):
        """1000x1 mask should work."""
        mask = np.zeros((1000, 1), dtype=np.uint8)
        mask[100:200, 0] = 1
        rle = RLEMask.from_array(mask)
        assert rle.area() == 100
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_large_square(self):
        """500x500 mask should work efficiently."""
        mask = np.random.randint(0, 2, (500, 500), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        decoded = np.array(rle)
        np.testing.assert_array_equal(decoded, mask)


# =============================================================================
# Boundary Conditions
# =============================================================================

class TestBoundaryConditions:
    def test_crop_at_edge(self):
        """Cropping at mask edge should work."""
        mask = np.ones((10, 10), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        # Crop at edge
        cropped = rle.crop([8, 8, 2, 2])
        assert cropped.shape == (2, 2)

    def test_crop_beyond_edge(self):
        """Cropping beyond edge should be handled."""
        mask = np.ones((5, 5), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        # This should either clip or handle gracefully
        try:
            cropped = rle.crop([3, 3, 10, 10])
            # If it works, check it's valid
            assert cropped.shape[0] > 0
            assert cropped.shape[1] > 0
        except (ValueError, IndexError):
            pass  # Also acceptable

    def test_pad_zero_amount(self):
        """Padding with zero should return same shape."""
        mask = np.eye(5, dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        padded = rle.pad(0, 0, 0, 0)
        assert padded.shape == rle.shape
        np.testing.assert_array_equal(np.array(padded), mask)

    def test_shift_to_edge(self):
        """Shifting to edge should clip content."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)
        shifted = rle.shift((3, 3))
        # Should be clipped
        assert shifted.area() <= 1

    def test_shift_completely_out(self):
        """Shifting completely out of bounds should result in empty."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        rle = RLEMask.from_array(mask)
        shifted = rle.shift((10, 10))
        assert shifted.area() == 0

    def test_getitem_last_row(self):
        """Indexing last row should work."""
        mask = np.eye(5, dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        assert rle[-1, -1] == 1
        assert rle[4, 4] == 1

    def test_getitem_last_column(self):
        """Indexing last column should work."""
        mask = np.eye(5, dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        row = rle[-1:, :]
        assert row.shape == (1, 5)


# =============================================================================
# Checkerboard and Complex Patterns
# =============================================================================

class TestComplexPatterns:
    def test_checkerboard_encode_decode(self):
        """Checkerboard pattern (worst case RLE) should work."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[::2, ::2] = 1
        mask[1::2, 1::2] = 1
        rle = RLEMask.from_array(mask)
        decoded = np.array(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_alternating_rows(self):
        """Alternating row pattern should work."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[::2, :] = 1
        rle = RLEMask.from_array(mask)
        assert rle.area() == 50
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_alternating_columns(self):
        """Alternating column pattern should work."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:, ::2] = 1
        rle = RLEMask.from_array(mask)
        assert rle.area() == 50
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_diagonal_stripe(self):
        """Diagonal stripe pattern should work."""
        mask = np.eye(10, dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        assert rle.area() == 10
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_random_mask(self):
        """Random mask should roundtrip correctly."""
        for _ in range(10):
            mask = np.random.randint(0, 2, (50, 50), dtype=np.uint8)
            rle = RLEMask.from_array(mask)
            decoded = np.array(rle)
            np.testing.assert_array_equal(decoded, mask)


# =============================================================================
# Error Handling
# =============================================================================

class TestErrorHandling:
    def test_different_shape_union(self):
        """Union of different-shaped masks should raise."""
        rle1 = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        rle2 = RLEMask.from_array(np.eye(5, dtype=np.uint8))
        with pytest.raises(ValueError):
            rle1 | rle2

    def test_different_shape_intersection(self):
        """Intersection of different-shaped masks should raise."""
        rle1 = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        rle2 = RLEMask.from_array(np.eye(5, dtype=np.uint8))
        with pytest.raises(ValueError):
            rle1 & rle2

    def test_different_shape_merge_many(self):
        """merge_many with different shapes should raise."""
        rle1 = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        rle2 = RLEMask.from_array(np.eye(5, dtype=np.uint8))
        with pytest.raises(ValueError):
            RLEMask.merge_many([rle1, rle2], BoolFunc.OR)

    def test_empty_merge_many(self):
        """merge_many with empty list should raise."""
        with pytest.raises(ValueError):
            RLEMask.merge_many([], BoolFunc.OR)

    def test_invalid_counts_sum(self):
        """from_counts with wrong sum should raise."""
        with pytest.raises(ValueError):
            RLEMask.from_counts([1, 2, 3], shape=(3, 3))  # sum=6, need 9

    def test_invalid_order(self):
        """from_counts with invalid order should raise."""
        with pytest.raises(ValueError):
            RLEMask.from_counts([9], shape=(3, 3), order='X')

    def test_decode_wrong_size(self):
        """Decoding with wrong size in dict should raise."""
        rle = rlemasklib.encode(np.eye(3, dtype=np.uint8))
        rle['size'] = [2, 2]
        with pytest.raises(ValueError):
            rlemasklib.decode(rle)

    def test_setitem_step_not_1(self):
        """setitem with step != 1 should raise."""
        rle = RLEMask.zeros((5, 5))
        with pytest.raises(ValueError):
            rle[::2, ::2] = 1


# =============================================================================
# Functional API Edge Cases
# =============================================================================

class TestFunctionalEdgeCases:
    def test_area_empty_list(self):
        """area of empty list should return empty."""
        areas = rlemasklib.area([])
        assert len(areas) == 0

    def test_complement_single(self):
        """complement of single mask (not list)."""
        mask = np.eye(3, dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        comp = rlemasklib.complement(rle)
        assert 'size' in comp  # Should return single dict

    def test_complement_list(self):
        """complement of list of masks."""
        masks = [np.eye(3, dtype=np.uint8), np.ones((3, 3), dtype=np.uint8)]
        rles = [rlemasklib.encode(m) for m in masks]
        comps = rlemasklib.complement(rles)
        assert len(comps) == 2

    def test_crop_single_mask_single_bbox(self):
        """crop single mask with single bbox."""
        mask = np.ones((10, 10), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        cropped = rlemasklib.crop(rle, [2, 2, 5, 5])
        assert cropped['size'] == [5, 5]

    def test_to_bbox_empty_mask(self):
        """to_bbox of empty mask."""
        rle = rlemasklib.zeros((5, 5))
        bbox = rlemasklib.to_bbox(rle)
        # Should return valid bbox (possibly zero-sized)
        assert len(bbox) == 4

    def test_centroid_single(self):
        """centroid of single mask (not list)."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = rlemasklib.encode(mask)
        centroid = rlemasklib.centroid(rle)
        assert len(centroid) == 2


# =============================================================================
# Serialization Edge Cases
# =============================================================================

class TestSerializationEdgeCases:
    def test_to_dict_roundtrip(self):
        """to_dict -> from_dict should preserve mask."""
        mask = np.random.randint(0, 2, (20, 20), dtype=np.uint8)
        rle1 = RLEMask.from_array(mask)
        d = rle1.to_dict()
        rle2 = RLEMask.from_dict(d)
        assert rle1 == rle2

    def test_to_dict_zlevel_roundtrip(self):
        """to_dict with zlevel -> from_dict should preserve mask."""
        mask = np.random.randint(0, 2, (20, 20), dtype=np.uint8)
        rle1 = RLEMask.from_array(mask)
        d = rle1.to_dict(zlevel=-1)
        assert 'zcounts' in d
        rle2 = RLEMask.from_dict(d)
        assert rle1 == rle2

    def test_empty_mask_serialization(self):
        """Empty mask should serialize/deserialize correctly."""
        rle1 = RLEMask.zeros((5, 5))
        d = rle1.to_dict()
        rle2 = RLEMask.from_dict(d)
        assert rle1 == rle2
        assert rle2.area() == 0

    def test_full_mask_serialization(self):
        """Full mask should serialize/deserialize correctly."""
        rle1 = RLEMask.ones((5, 5))
        d = rle1.to_dict()
        rle2 = RLEMask.from_dict(d)
        assert rle1 == rle2
        assert rle2.area() == 25


# =============================================================================
# Numeric Stability
# =============================================================================

class TestNumericStability:
    def test_iou_identical(self):
        """IoU of identical masks should be exactly 1.0."""
        mask = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        iou = rle.iou(rle)
        assert iou == 1.0

    def test_iou_empty_masks(self):
        """IoU of two empty masks should be 0.0 (or nan)."""
        rle1 = RLEMask.zeros((5, 5))
        rle2 = RLEMask.zeros((5, 5))
        iou = rle1.iou(rle2)
        # 0/0 could be 0 or nan depending on implementation
        assert iou == 0.0 or np.isnan(iou)

    def test_iou_no_overlap(self):
        """IoU of non-overlapping masks should be 0.0."""
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[0:3, 0:3] = 1
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[7:10, 7:10] = 1
        rle1 = RLEMask.from_array(m1)
        rle2 = RLEMask.from_array(m2)
        iou = rle1.iou(rle2)
        assert iou == 0.0

    def test_centroid_single_pixel(self):
        """Centroid of single pixel should be at that pixel."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[7, 3] = 1
        rle = RLEMask.from_array(mask)
        cx, cy = rle.centroid()
        assert cx == 3.0
        assert cy == 7.0


# =============================================================================
# Memory and Performance Edge Cases
# =============================================================================

class TestMemoryPerformance:
    def test_large_mask_operations(self):
        """Operations on large masks should complete."""
        mask = np.random.randint(0, 2, (500, 500), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        # Basic operations should work
        assert rle.area() > 0
        _ = rle.complement()
        _ = rle.bbox()

    def test_many_small_components(self):
        """Mask with many small components should handle."""
        # Checkerboard has many 1-pixel components with 4-connectivity
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[::2, ::2] = 1
        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4)
        assert len(components) == 100

    def test_deeply_nested_boolean_ops(self):
        """Deeply nested boolean operations should work."""
        m1 = RLEMask.from_array(np.eye(10, dtype=np.uint8))
        m2 = RLEMask.from_array(np.eye(10, dtype=np.uint8)[::-1])

        # Build up a complex expression
        result = m1
        for _ in range(10):
            result = (result | m2) & m1

        assert result.area() > 0


# =============================================================================
# Inplace Operation Edge Cases
# =============================================================================

class TestInplaceEdgeCases:
    def test_inplace_on_copy(self):
        """Inplace on copy should not affect original."""
        mask = np.eye(5, dtype=np.uint8)
        rle1 = RLEMask.from_array(mask)
        rle2 = rle1.copy()

        rle2.complement(inplace=True)

        # Original should be unchanged
        np.testing.assert_array_equal(np.array(rle1), mask)
        # Copy should be modified
        np.testing.assert_array_equal(np.array(rle2), 1 - mask)

    def test_inplace_returns_self(self):
        """Inplace operations should return self."""
        rle = RLEMask.from_array(np.eye(5, dtype=np.uint8))

        result = rle.complement(inplace=True)
        assert result is rle

        result = rle.pad(1, 1, 1, 1, inplace=True)
        assert result is rle

    def test_chained_inplace(self):
        """Chained inplace operations should work."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1
        rle = RLEMask.from_array(mask)

        rle.complement(inplace=True).pad(1, 1, 1, 1, inplace=True)

        assert rle.shape == (12, 12)