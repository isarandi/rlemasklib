import rlemasklib
import numpy as np
import pytest

# Set 1 second timeout for all tests in this file
pytestmark = pytest.mark.timeout(1)


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================

@pytest.fixture
def eye3_mask():
    """3x3 identity matrix mask."""
    return np.eye(3, dtype=np.uint8)


@pytest.fixture
def eye3_flipped():
    """3x3 anti-diagonal mask."""
    return np.eye(3, dtype=np.uint8)[::-1]


@pytest.fixture
def sample_5x5():
    """5x5 sample mask with a cross pattern."""
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, :] = 1  # horizontal line
    mask[:, 2] = 1  # vertical line
    return mask


@pytest.fixture
def checker_4x4():
    """4x4 checkerboard pattern."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0::2, 0::2] = 1
    mask[1::2, 1::2] = 1
    return mask


def assert_mask_equal(rle, expected):
    """Helper to compare RLE mask with expected numpy array."""
    decoded = rlemasklib.decode(rle)
    np.testing.assert_array_equal(decoded, expected.astype(np.uint8))


# =============================================================================
# Core Encode/Decode Tests
# =============================================================================

class TestEncodeDecode:
    def test_encode_decode_roundtrip(self, eye3_mask):
        """Encoding then decoding should return original mask."""
        encoded = rlemasklib.encode(eye3_mask)
        decoded = rlemasklib.decode(encoded)
        np.testing.assert_array_equal(decoded, eye3_mask)

    def test_encode_decode_all_zeros(self):
        """Empty mask roundtrip."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        encoded = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_encode_decode_all_ones(self):
        """Full mask roundtrip."""
        mask = np.ones((10, 10), dtype=np.uint8)
        encoded = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(encoded)
        np.testing.assert_array_equal(decoded, mask)

    def test_encode_preserves_size(self, sample_5x5):
        """Encoded mask should have correct size metadata."""
        encoded = rlemasklib.encode(sample_5x5)
        assert encoded['size'] == [5, 5]

    def test_decode_error_wrong_size(self):
        """Decoding with wrong size should raise ValueError."""
        d1 = rlemasklib.encode(np.eye(3))
        d1['size'] = [2, 2]
        with pytest.raises(ValueError):
            rlemasklib.decode(d1)

    def test_encode_non_contiguous(self):
        """Encoding non-contiguous array should work."""
        mask = np.eye(5, dtype=np.uint8)[::2, ::2]  # non-contiguous
        encoded = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(encoded)
        np.testing.assert_array_equal(decoded, np.ascontiguousarray(mask))


# =============================================================================
# Mask Creation Tests
# =============================================================================

class TestMaskCreation:
    def test_ones(self):
        """ones() should create all-foreground mask."""
        rle = rlemasklib.ones((5, 5))
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, np.ones((5, 5), dtype=np.uint8))

    def test_zeros(self):
        """zeros() should create all-background mask."""
        rle = rlemasklib.zeros((5, 5))
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, np.zeros((5, 5), dtype=np.uint8))

    def test_ones_like(self, eye3_mask):
        """ones_like() should create all-foreground mask with same shape."""
        orig = rlemasklib.encode(eye3_mask)
        rle = rlemasklib.ones_like(orig)
        decoded = rlemasklib.decode(rle)
        assert decoded.shape == eye3_mask.shape
        np.testing.assert_array_equal(decoded, np.ones_like(eye3_mask))

    def test_zeros_like(self, eye3_mask):
        """zeros_like() should create all-background mask with same shape."""
        orig = rlemasklib.encode(eye3_mask)
        rle = rlemasklib.zeros_like(orig)
        decoded = rlemasklib.decode(rle)
        assert decoded.shape == eye3_mask.shape
        np.testing.assert_array_equal(decoded, np.zeros_like(eye3_mask))

    def test_from_bbox(self):
        """from_bbox() should create rectangular mask."""
        # bbox format: [x, y, width, height]
        rle = rlemasklib.from_bbox([1, 1, 3, 2], (5, 5))
        decoded = rlemasklib.decode(rle)
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1:3, 1:4] = 1
        np.testing.assert_array_equal(decoded, expected)

    def test_from_polygon(self):
        """from_polygon() should create polygon mask."""
        # Rectangle polygon (simpler, more predictable)
        poly = [1, 1, 3, 1, 3, 3, 1, 3]  # x1,y1, x2,y2, x3,y3, x4,y4
        rle = rlemasklib.from_polygon(poly, (5, 5))
        decoded = rlemasklib.decode(rle)
        # Check that it's not empty
        assert decoded.sum() > 0
        # Check interior point is filled
        assert decoded[2, 2] == 1


# =============================================================================
# Set Operations Tests
# =============================================================================

class TestSetOperations:
    def test_union(self, eye3_mask, eye3_flipped):
        """Union should combine two masks."""
        d1 = rlemasklib.encode(eye3_mask)
        d2 = rlemasklib.encode(eye3_flipped)
        d3 = rlemasklib.union([d1, d2])
        expected = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)
        assert_mask_equal(d3, expected)

    def test_intersection(self, eye3_mask, eye3_flipped):
        """Intersection should find common area."""
        d1 = rlemasklib.encode(eye3_mask)
        d2 = rlemasklib.encode(eye3_flipped)
        d3 = rlemasklib.intersection([d1, d2])
        expected = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.uint8)
        assert_mask_equal(d3, expected)

    def test_difference(self, eye3_mask, eye3_flipped):
        """Difference should subtract second from first."""
        d1 = rlemasklib.encode(eye3_mask)
        d2 = rlemasklib.encode(eye3_flipped)
        d3 = rlemasklib.difference(d1, d2)
        expected = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=np.uint8)
        assert_mask_equal(d3, expected)

    def test_symmetric_difference(self, eye3_mask, eye3_flipped):
        """Symmetric difference should find non-overlapping areas."""
        d1 = rlemasklib.encode(eye3_mask)
        d2 = rlemasklib.encode(eye3_flipped)
        d3 = rlemasklib.symmetric_difference(d1, d2)
        expected = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
        assert_mask_equal(d3, expected)

    def test_complement_via_difference(self, eye3_mask):
        """Complement can be computed via difference from full mask."""
        d1 = rlemasklib.encode(eye3_mask)
        full = rlemasklib.ones((3, 3))
        d2 = rlemasklib.difference(full, d1)
        expected = 1 - eye3_mask
        assert_mask_equal(d2, expected)

    def test_union_single_mask(self, eye3_mask):
        """Union of single mask should return that mask."""
        d1 = rlemasklib.encode(eye3_mask)
        d2 = rlemasklib.union([d1])
        assert_mask_equal(d2, eye3_mask)

    def test_union_multiple_masks(self):
        """Union of multiple masks."""
        m1 = np.zeros((5, 5), dtype=np.uint8)
        m1[0, 0] = 1
        m2 = np.zeros((5, 5), dtype=np.uint8)
        m2[2, 2] = 1
        m3 = np.zeros((5, 5), dtype=np.uint8)
        m3[4, 4] = 1

        d1, d2, d3 = [rlemasklib.encode(m) for m in [m1, m2, m3]]
        result = rlemasklib.union([d1, d2, d3])

        expected = m1 | m2 | m3
        assert_mask_equal(result, expected)

    def test_merge_with_boolfunc(self, eye3_mask, eye3_flipped):
        """merge() with custom BoolFunc should work."""
        d1 = rlemasklib.encode(eye3_mask)
        d2 = rlemasklib.encode(eye3_flipped)
        # Test XOR (same as symmetric_difference)
        d3 = rlemasklib.merge([d1, d2], rlemasklib.BoolFunc.XOR)
        expected = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
        assert_mask_equal(d3, expected)


# =============================================================================
# Geometric Operations Tests
# =============================================================================

class TestGeometricOperations:
    def test_crop(self, sample_5x5):
        """crop() should extract rectangular region."""
        rle = rlemasklib.encode(sample_5x5)
        # bbox format: [x, y, width, height]
        cropped = rlemasklib.crop(rle, [1, 1, 3, 3])
        decoded = rlemasklib.decode(cropped)
        expected = sample_5x5[1:4, 1:4]
        np.testing.assert_array_equal(decoded, expected)

    def test_crop_preserves_content(self):
        """Cropped region should match original."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:7, 3:7] = 1
        rle = rlemasklib.encode(mask)
        cropped = rlemasklib.crop(rle, [2, 2, 6, 6])
        decoded = rlemasklib.decode(cropped)
        expected = mask[2:8, 2:8]
        np.testing.assert_array_equal(decoded, expected)

    def test_pad(self, eye3_mask):
        """pad() should add border around mask."""
        rle = rlemasklib.encode(eye3_mask)
        # Pad 1 pixel on each side: (top, bottom, left, right)
        padded = rlemasklib.pad(rle, (1, 1, 1, 1), 0)
        decoded = rlemasklib.decode(padded)
        assert decoded.shape == (5, 5)
        # Original content should be in center
        np.testing.assert_array_equal(decoded[1:4, 1:4], eye3_mask)
        # Borders should be 0
        assert decoded[0, :].sum() == 0
        assert decoded[-1, :].sum() == 0

    def test_pad_with_ones(self, eye3_mask):
        """pad() with value=1 should add foreground border."""
        rle = rlemasklib.encode(eye3_mask)
        padded = rlemasklib.pad(rle, (1, 1, 1, 1), 1)
        decoded = rlemasklib.decode(padded)
        assert decoded[0, :].sum() == 5  # top row all ones
        assert decoded[-1, :].sum() == 5  # bottom row all ones

    def test_shift_down(self, eye3_mask):
        """shift() should translate mask."""
        rle = rlemasklib.encode(eye3_mask)
        shifted = rlemasklib.shift(rle, (1, 0))  # shift down by 1
        decoded = rlemasklib.decode(shifted)
        # First row should be zeros, rest shifted down
        assert decoded[0, :].sum() == 0
        np.testing.assert_array_equal(decoded[1:, :], eye3_mask[:-1, :])

    def test_to_bbox(self, sample_5x5):
        """to_bbox() should return bounding box."""
        rle = rlemasklib.encode(sample_5x5)
        bbox = rlemasklib.to_bbox(rle)
        # Cross pattern spans full width at row 2, full height at col 2
        # bbox format: [x, y, width, height]
        assert bbox[0] == 0  # x starts at 0
        assert bbox[1] == 0  # y starts at 0
        assert bbox[2] == 5  # width
        assert bbox[3] == 5  # height

    def test_to_bbox_small_region(self):
        """to_bbox() for small region."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:5, 3:7] = 1
        rle = rlemasklib.encode(mask)
        bbox = rlemasklib.to_bbox(rle)
        np.testing.assert_array_equal(bbox, [3, 2, 4, 3])  # x, y, w, h


# =============================================================================
# Morphological Operations Tests
# =============================================================================

class TestMorphologicalOperations:
    def test_dilate_single_pixel(self):
        """Dilating single pixel should create cross (4-connectivity)."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = rlemasklib.encode(mask)
        dilated = rlemasklib.dilate(rle, connectivity=4)
        decoded = rlemasklib.decode(dilated)
        # Should have cross pattern
        assert decoded[2, 2] == 1  # center
        assert decoded[1, 2] == 1  # top
        assert decoded[3, 2] == 1  # bottom
        assert decoded[2, 1] == 1  # left
        assert decoded[2, 3] == 1  # right
        # Corners should be 0 with 4-connectivity
        assert decoded[1, 1] == 0
        assert decoded[1, 3] == 0

    def test_dilate_8_connectivity(self):
        """Dilating with 8-connectivity includes diagonals."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = rlemasklib.encode(mask)
        dilated = rlemasklib.dilate(rle, connectivity=8)
        decoded = rlemasklib.decode(dilated)
        # All 8 neighbors should be 1
        assert decoded[1:4, 1:4].sum() == 9

    def test_erode(self):
        """Erosion should shrink mask."""
        # Use a mask with 0 border so erosion has something to erode
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1  # 5x5 block inside 7x7
        rle = rlemasklib.encode(mask)
        eroded = rlemasklib.erode(rle, connectivity=4)
        decoded = rlemasklib.decode(eroded)
        # The 5x5 block should shrink to 3x3
        assert decoded[2:5, 2:5].sum() == 9
        # The border of the original block should be eroded
        assert decoded[1, 1:6].sum() == 0
        assert decoded[5, 1:6].sum() == 0

    def test_opening_removes_noise(self):
        """Opening should remove small isolated pixels."""
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[2:5, 2:5] = 1  # 3x3 block
        mask[0, 0] = 1  # isolated pixel (noise)
        rle = rlemasklib.encode(mask)
        opened = rlemasklib.opening(rle, connectivity=4)
        decoded = rlemasklib.decode(opened)
        # Isolated pixel should be removed
        assert decoded[0, 0] == 0
        # Main block should mostly remain (may shrink slightly)
        assert decoded[2:5, 2:5].sum() > 0

    def test_closing_fills_holes(self):
        """Closing should fill small holes."""
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[2, 2] = 0  # hole in center
        rle = rlemasklib.encode(mask)
        closed = rlemasklib.closing(rle, connectivity=4)
        decoded = rlemasklib.decode(closed)
        # Hole should be filled
        assert decoded[2, 2] == 1


# =============================================================================
# Connected Components Tests
# =============================================================================

class TestConnectedComponents:
    def test_connected_components_two_regions(self):
        """Should find separate connected regions."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:3, 0:3] = 1  # top-left region
        mask[7:10, 7:10] = 1  # bottom-right region
        rle = rlemasklib.encode(mask)
        components = rlemasklib.connected_components(rle, connectivity=4)
        assert len(components) == 2
        # Each component should have area 9
        areas = [rlemasklib.area(c) for c in components]
        assert sorted(areas) == [9, 9]

    def test_largest_connected_component(self):
        """Should return the largest region."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # small region (4 pixels)
        mask[5:10, 5:10] = 1  # large region (25 pixels)
        rle = rlemasklib.encode(mask)
        largest = rlemasklib.largest_connected_component(rle, connectivity=4)
        assert rlemasklib.area(largest) == 25

    def test_remove_small_components(self):
        """Should remove components below min_size."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels
        mask[5:10, 5:10] = 1  # 25 pixels
        rle = rlemasklib.encode(mask)
        filtered = rlemasklib.remove_small_components(rle, connectivity=4, min_size=10)
        decoded = rlemasklib.decode(filtered)
        # Small component should be gone
        assert decoded[0:2, 0:2].sum() == 0
        # Large component remains
        assert decoded[5:10, 5:10].sum() == 25

    def test_fill_small_holes(self):
        """Should fill small holes in mask."""
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[2, 2] = 0  # small hole (1 pixel)
        mask[5:8, 5:8] = 0  # larger hole (9 pixels)
        rle = rlemasklib.encode(mask)
        filled = rlemasklib.fill_small_holes(rle, connectivity=4, min_size=5)
        decoded = rlemasklib.decode(filled)
        # Small hole should be filled
        assert decoded[2, 2] == 1
        # Large hole should remain
        assert decoded[5:8, 5:8].sum() == 0


# =============================================================================
# Mask Analysis Tests
# =============================================================================

class TestMaskAnalysis:
    def test_area(self, eye3_mask):
        """area() should count foreground pixels."""
        rle = rlemasklib.encode(eye3_mask)
        assert rlemasklib.area(rle) == 3

    def test_area_full_mask(self):
        """area() of full mask should equal total pixels."""
        mask = np.ones((10, 15), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        assert rlemasklib.area(rle) == 150

    def test_area_empty_mask(self):
        """area() of empty mask should be 0."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        assert rlemasklib.area(rle) == 0

    def test_centroid(self):
        """centroid() should return center of mass."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1  # single pixel at center
        rle = rlemasklib.encode(mask)
        cx, cy = rlemasklib.centroid(rle)
        assert cx == 2.0
        assert cy == 2.0

    def test_centroid_rectangle(self):
        """centroid() of rectangle should be at center."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 3:7] = 1  # 6x4 rectangle
        rle = rlemasklib.encode(mask)
        cx, cy = rlemasklib.centroid(rle)
        # Center should be at (4.5, 4.5) - middle of rectangle
        assert abs(cx - 4.5) < 0.1
        assert abs(cy - 4.5) < 0.1

    def test_iou_identical(self, eye3_mask):
        """IoU of identical masks should be 1.0."""
        rle = rlemasklib.encode(eye3_mask)
        iou = rlemasklib.iou([rle, rle])
        assert abs(iou - 1.0) < 1e-6

    def test_iou_no_overlap(self):
        """IoU of non-overlapping masks should be 0.0."""
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[0:3, 0:3] = 1
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[7:10, 7:10] = 1
        r1 = rlemasklib.encode(m1)
        r2 = rlemasklib.encode(m2)
        iou = rlemasklib.iou([r1, r2])
        assert iou == 0.0

    def test_iou_partial_overlap(self):
        """IoU of partially overlapping masks."""
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[0:5, 0:5] = 1  # 25 pixels
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[2:7, 2:7] = 1  # 25 pixels
        # Overlap is [2:5, 2:5] = 9 pixels
        # Union is 25 + 25 - 9 = 41 pixels
        r1 = rlemasklib.encode(m1)
        r2 = rlemasklib.encode(m2)
        iou = rlemasklib.iou([r1, r2])
        expected_iou = 9 / 41
        assert abs(iou - expected_iou) < 1e-6


# =============================================================================
# Compression Tests
# =============================================================================

class TestCompression:
    def test_compress_decompress_roundtrip(self, sample_5x5):
        """Compression then decompression should preserve mask."""
        rle = rlemasklib.encode(sample_5x5)
        compressed = rlemasklib.compress(rle)
        decompressed = rlemasklib.decompress(compressed)
        decoded = rlemasklib.decode(decompressed)
        np.testing.assert_array_equal(decoded, sample_5x5)

    def test_encode_with_compression(self, sample_5x5):
        """encode() with compressed=True should work."""
        rle = rlemasklib.encode(sample_5x5, compressed=True)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, sample_5x5)

    def test_compressed_smaller_for_simple_mask(self):
        """Compressed RLE should be compact for simple masks."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[40:60, 40:60] = 1  # simple rectangle
        rle_uncompressed = rlemasklib.encode(mask, compressed=False)
        rle_compressed = rlemasklib.encode(mask, compressed=True)
        # Compressed counts should be shorter or same
        assert len(rle_compressed['counts']) <= len(str(rle_uncompressed['ucounts'])) + 50


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    def test_single_pixel_mask(self):
        """1x1 mask should work."""
        mask = np.array([[1]], dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_single_row_mask(self):
        """1xN mask should work."""
        mask = np.array([[0, 1, 1, 0, 1]], dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_single_column_mask(self):
        """Nx1 mask should work."""
        mask = np.array([[0], [1], [1], [0], [1]], dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_large_mask(self):
        """Large mask should work efficiently."""
        mask = np.random.randint(0, 2, (500, 500), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_alternating_pattern(self):
        """Highly fragmented mask (worst case for RLE)."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[::2, ::2] = 1  # checkerboard-like
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_operations_on_different_sizes_should_fail(self):
        """Operations on different-sized masks should raise error."""
        m1 = np.eye(3, dtype=np.uint8)
        m2 = np.eye(5, dtype=np.uint8)
        r1 = rlemasklib.encode(m1)
        r2 = rlemasklib.encode(m2)
        with pytest.raises((ValueError, AssertionError)):
            rlemasklib.union([r1, r2])

    def test_crop_outside_bounds(self):
        """Cropping with bbox partially outside should handle gracefully."""
        mask = np.ones((5, 5), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        # This might either clip or raise - just verify it doesn't crash
        try:
            cropped = rlemasklib.crop(rle, [3, 3, 5, 5])  # extends beyond
            # If it works, verify output is valid
            decoded = rlemasklib.decode(cropped)
            assert decoded.shape[0] > 0 and decoded.shape[1] > 0
        except (ValueError, IndexError):
            pass  # Also acceptable behavior

    def test_nonempty_via_area(self):
        """Check if mask is non-empty via area > 0."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = rlemasklib.encode(mask)
        assert rlemasklib.area(rle) > 0

    def test_empty_via_area(self):
        """Check if mask is empty via area == 0."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        assert rlemasklib.area(rle) == 0

    def test_full_via_area(self):
        """Check if mask is full via area == total pixels."""
        mask = np.ones((5, 5), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        assert rlemasklib.area(rle) == 25


# =============================================================================
# Batch Operations Tests
# =============================================================================

class TestBatchOperations:
    def test_area_batch(self):
        """area() should work on list of masks."""
        masks = [
            np.eye(3, dtype=np.uint8),
            np.ones((4, 4), dtype=np.uint8),
            np.zeros((5, 5), dtype=np.uint8)
        ]
        rles = [rlemasklib.encode(m) for m in masks]
        areas = rlemasklib.area(rles)
        np.testing.assert_array_equal(areas, [3, 16, 0])

    def test_to_bbox_batch(self):
        """to_bbox() should work on list of masks."""
        m1 = np.zeros((10, 10), dtype=np.uint8)
        m1[2:5, 3:7] = 1
        m2 = np.zeros((10, 10), dtype=np.uint8)
        m2[0:3, 0:3] = 1
        rles = [rlemasklib.encode(m1), rlemasklib.encode(m2)]
        bboxes = rlemasklib.to_bbox(rles)
        assert len(bboxes) == 2
        np.testing.assert_array_equal(bboxes[0], [3, 2, 4, 3])
        np.testing.assert_array_equal(bboxes[1], [0, 0, 3, 3])
