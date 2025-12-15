"""Tests for warp and geometric transformation operations."""

import numpy as np
import pytest
from rlemasklib.oop import RLEMask


# =============================================================================
# Affine Warp Tests
# =============================================================================

class TestWarpAffine:
    def test_identity_transform(self):
        """Identity affine transform should preserve mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1
        rle = RLEMask.from_array(mask)

        # Identity matrix
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        warped = rle.warp_affine(M, (10, 10))
        np.testing.assert_array_equal(np.array(warped), mask)

    def test_translation(self):
        """Translation should shift mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:3] = 1
        rle = RLEMask.from_array(mask)

        # Translate by (2, 3)
        M = np.array([[1, 0, 2], [0, 1, 3]], dtype=np.float64)
        warped = rle.warp_affine(M, (10, 10))
        arr = np.array(warped)

        # Original position should be empty
        assert arr[0:3, 0:3].sum() == 0
        # New position should be filled (accounting for clipping)
        assert arr[3:6, 2:5].sum() > 0

    def test_scale_up(self):
        """Scaling up should enlarge mask."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        rle = RLEMask.from_array(mask)

        # Scale by 2x
        M = np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float64)
        warped = rle.warp_affine(M, (10, 10))

        # Scaled mask should have larger area
        assert warped.area() > rle.area()

    def test_scale_down(self):
        """Scaling down should shrink mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        rle = RLEMask.from_array(mask)

        # Scale by 0.5x
        M = np.array([[0.5, 0, 0], [0, 0.5, 0]], dtype=np.float64)
        warped = rle.warp_affine(M, (5, 5))

        # Scaled mask should have smaller area
        assert warped.area() < rle.area()

    def test_rotation_90(self):
        """90 degree rotation should rotate mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:5] = 1  # wider than tall
        rle = RLEMask.from_array(mask)

        # 90 degree CCW rotation around origin, then translate to visible area
        # For 90 CCW: [cos(90), -sin(90)] = [0, -1], [sin(90), cos(90)] = [1, 0]
        M = np.array([[0, -1, 10], [1, 0, 0]], dtype=np.float64)
        warped = rle.warp_affine(M, (10, 10))

        # After rotation, should have non-zero area
        assert warped.area() > 0

    def test_output_shape(self):
        """warp_affine should produce output of specified shape."""
        mask = np.eye(5, dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        warped = rle.warp_affine(M, (20, 15))

        assert warped.shape == (20, 15)


# =============================================================================
# Perspective Warp Tests
# =============================================================================

class TestWarpPerspective:
    def test_identity_homography(self):
        """Identity homography should preserve mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1
        rle = RLEMask.from_array(mask)

        # Identity homography
        H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        warped = rle.warp_perspective(H, (10, 10))
        np.testing.assert_array_equal(np.array(warped), mask)

    def test_translation_homography(self):
        """Translation homography should shift mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:3] = 1
        rle = RLEMask.from_array(mask)

        # Translate by (2, 1)
        H = np.array([[1, 0, 2], [0, 1, 1], [0, 0, 1]], dtype=np.float64)
        warped = rle.warp_perspective(H, (10, 10))
        arr = np.array(warped)

        # Original position should be empty
        assert arr[0, 0] == 0

    def test_scale_homography(self):
        """Scale homography should resize mask."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        rle = RLEMask.from_array(mask)

        # Scale by 0.5
        H = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]], dtype=np.float64)
        warped = rle.warp_perspective(H, (5, 5))

        assert warped.shape == (5, 5)
        assert warped.area() > 0

    def test_output_shape_perspective(self):
        """warp_perspective should produce output of specified shape."""
        mask = np.eye(5, dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        H = np.eye(3, dtype=np.float64)
        warped = rle.warp_perspective(H, (20, 15))

        assert warped.shape == (20, 15)


# =============================================================================
# Resize Tests
# =============================================================================

class TestResize:
    def test_resize_to_shape(self):
        """resize with output_imshape should resize to specified size."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        rle = RLEMask.from_array(mask)

        resized = rle.resize((20, 20))
        assert resized.shape == (20, 20)

    def test_resize_with_fx_fy(self):
        """resize with fx/fy should scale by factors."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        rle = RLEMask.from_array(mask)

        resized = rle.resize(None, fx=2.0, fy=2.0)
        assert resized.shape == (20, 20)

    def test_resize_scale_up_area(self):
        """Scaling up should increase area (approximately)."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1  # 16 pixels
        rle = RLEMask.from_array(mask)

        resized = rle.resize(None, fx=2.0, fy=2.0)
        # Area should approximately quadruple
        assert resized.area() > rle.area() * 3

    def test_resize_scale_down_area(self):
        """Scaling down should decrease area."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:16, 4:16] = 1  # 144 pixels
        rle = RLEMask.from_array(mask)

        resized = rle.resize(None, fx=0.5, fy=0.5)
        assert resized.area() < rle.area()

    def test_resize_anisotropic(self):
        """Anisotropic resize should use different scales."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        rle = RLEMask.from_array(mask)

        resized = rle.resize(None, fx=2.0, fy=1.0)
        assert resized.shape == (10, 20)

    def test_resize_missing_args(self):
        """resize without output_imshape or fx/fy should raise."""
        mask = np.eye(5, dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        with pytest.raises(ValueError):
            rle.resize(None)


# =============================================================================
# Transpose and Rotation Tests
# =============================================================================

class TestTransposeRotation:
    def test_transpose_preserves_content(self):
        """Transpose should swap dimensions correctly."""
        mask = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)  # 2x3
        rle = RLEMask.from_array(mask)

        transposed = rle.transpose()
        assert transposed.shape == (3, 2)
        np.testing.assert_array_equal(np.array(transposed), mask.T)

    def test_double_transpose(self):
        """Double transpose should return original."""
        mask = np.random.randint(0, 2, (5, 7), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        double_transposed = rle.transpose().transpose()
        np.testing.assert_array_equal(np.array(double_transposed), mask)

    def test_T_property(self):
        """T property should be same as transpose."""
        mask = np.random.randint(0, 2, (4, 6), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        assert rle.T == rle.transpose()

    def test_rot90_k0(self):
        """rot90 with k=0 should return original."""
        mask = np.random.randint(0, 2, (5, 5), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        rotated = rle.rot90(k=0)
        np.testing.assert_array_equal(np.array(rotated), mask)

    def test_rot90_k1(self):
        """rot90 with k=1 should rotate 90 degrees CCW."""
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        rotated = rle.rot90(k=1)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=1))

    def test_rot90_k2(self):
        """rot90 with k=2 should rotate 180 degrees."""
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        rotated = rle.rot90(k=2)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=2))

    def test_rot90_k3(self):
        """rot90 with k=3 should rotate 270 degrees CCW (90 CW)."""
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        rotated = rle.rot90(k=3)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=3))

    def test_rot90_k4_same_as_k0(self):
        """rot90 with k=4 should be same as k=0."""
        mask = np.random.randint(0, 2, (5, 5), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        rotated = rle.rot90(k=4)
        np.testing.assert_array_equal(np.array(rotated), mask)

    def test_rot90_negative_k(self):
        """rot90 with negative k should work."""
        mask = np.random.randint(0, 2, (5, 5), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        rotated = rle.rot90(k=-1)
        np.testing.assert_array_equal(np.array(rotated), np.rot90(mask, k=-1))

    def test_rot90_inplace(self):
        """rot90 with inplace=True should modify in place."""
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        result = rle.rot90(k=2, inplace=True)
        assert result is rle
        np.testing.assert_array_equal(np.array(rle), np.rot90(mask, k=2))


# =============================================================================
# Flip Tests
# =============================================================================

class TestFlip:
    def test_flipud(self):
        """flipud should flip vertically."""
        mask = np.array([[1, 0], [0, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        flipped = rle.flipud()
        np.testing.assert_array_equal(np.array(flipped), mask[::-1])

    def test_fliplr(self):
        """fliplr should flip horizontally."""
        mask = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        flipped = rle.fliplr()
        np.testing.assert_array_equal(np.array(flipped), mask[:, ::-1])

    def test_flip_axis0(self):
        """flip with axis=0 should flip vertically."""
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        flipped = rle.flip(axis=0)
        np.testing.assert_array_equal(np.array(flipped), mask[::-1])

    def test_flip_axis1(self):
        """flip with axis=1 should flip horizontally."""
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        flipped = rle.flip(axis=1)
        np.testing.assert_array_equal(np.array(flipped), mask[:, ::-1])

    def test_flip_invalid_axis(self):
        """flip with invalid axis should raise."""
        rle = RLEMask.from_array(np.eye(3, dtype=np.uint8))

        with pytest.raises(ValueError):
            rle.flip(axis=2)

    def test_double_flipud(self):
        """Double flipud should return original."""
        mask = np.random.randint(0, 2, (5, 7), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        double_flipped = rle.flipud().flipud()
        np.testing.assert_array_equal(np.array(double_flipped), mask)

    def test_double_fliplr(self):
        """Double fliplr should return original."""
        mask = np.random.randint(0, 2, (5, 7), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        double_flipped = rle.fliplr().fliplr()
        np.testing.assert_array_equal(np.array(double_flipped), mask)


# =============================================================================
# Repeat Tests
# =============================================================================

class TestRepeat:
    def test_repeat_basic(self):
        """repeat should expand each pixel."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        repeated = rle.repeat(2, 3)  # 2x vertical, 3x horizontal
        assert repeated.shape == (4, 6)

        arr = np.array(repeated)
        # Top-left pixel (1) should be repeated to 2x3 block
        assert arr[0:2, 0:3].sum() == 6

    def test_repeat_1x1(self):
        """repeat with (1, 1) should return copy."""
        mask = np.random.randint(0, 2, (5, 5), dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        repeated = rle.repeat(1, 1)
        np.testing.assert_array_equal(np.array(repeated), mask)

    def test_repeat_inplace(self):
        """repeat with inplace=True should modify in place."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        result = rle.repeat(2, 2, inplace=True)
        assert result is rle
        assert rle.shape == (4, 4)


# =============================================================================
# Integration Tests
# =============================================================================

class TestTransformIntegration:
    def test_resize_then_crop(self):
        """resize followed by crop should work."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        rle = RLEMask.from_array(mask)

        resized = rle.resize((20, 20))
        cropped = resized.crop([0, 0, 10, 10])

        assert cropped.shape == (10, 10)

    def test_multiple_transforms(self):
        """Chain of transforms should work."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        rle = RLEMask.from_array(mask)

        # Rotate, flip, pad
        result = rle.rot90(k=1).flipud().pad(1, 1, 1, 1)

        # Should have valid result
        assert result.shape == (12, 12)
        assert result.area() > 0

    def test_transform_preserves_area_for_identity(self):
        """Identity transforms should preserve area."""
        mask = np.random.randint(0, 2, (10, 10), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        original_area = rle.area()

        # Identity affine
        M = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
        warped = rle.warp_affine(M, (10, 10))

        assert warped.area() == original_area