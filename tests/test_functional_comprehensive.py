"""Comprehensive tests for the functional RLE API."""

import numpy as np
import pytest
import rlemasklib
from rlemasklib.boolfunc import BoolFunc


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def eye3():
    """3x3 identity matrix mask."""
    return np.eye(3, dtype=np.uint8)


@pytest.fixture
def eye3_flipped():
    """3x3 anti-diagonal mask."""
    return np.eye(3, dtype=np.uint8)[::-1]


@pytest.fixture
def rect_5x5():
    """5x5 mask with 3x3 rectangle in center."""
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    return mask


def decode_and_compare(rle, expected):
    """Helper to decode RLE and compare with expected array."""
    decoded = rlemasklib.decode(rle)
    np.testing.assert_array_equal(decoded, expected.astype(np.uint8))


# =============================================================================
# Encode/Decode Tests
# =============================================================================

class TestEncodeDecode:
    def test_encode_compressed(self, eye3):
        """encode with compressed=True should return 'counts' key."""
        rle = rlemasklib.encode(eye3, compressed=True)
        assert 'counts' in rle
        assert 'ucounts' not in rle

    def test_encode_uncompressed(self, eye3):
        """encode with compressed=False should return 'ucounts' key."""
        rle = rlemasklib.encode(eye3, compressed=False)
        assert 'ucounts' in rle
        assert 'counts' not in rle

    def test_encode_with_zlevel(self, eye3):
        """encode with zlevel should return 'zcounts' key."""
        rle = rlemasklib.encode(eye3, compressed=True, zlevel=-1)
        assert 'zcounts' in rle

    def test_encode_c_contiguous(self):
        """encode should handle C-contiguous arrays."""
        mask = np.ascontiguousarray(np.eye(5, dtype=np.uint8))
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_encode_f_contiguous(self):
        """encode should handle F-contiguous arrays."""
        mask = np.asfortranarray(np.eye(5, dtype=np.uint8))
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_encode_bool_array(self):
        """encode should handle boolean arrays."""
        mask = np.array([[True, False], [False, True]])
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask.astype(np.uint8))

    def test_encode_int_array(self):
        """encode should handle int arrays (nonzero = foreground)."""
        mask = np.array([[0, 5], [3, 0]], dtype=np.int32)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        expected = (mask != 0).astype(np.uint8)
        np.testing.assert_array_equal(decoded, expected)

    def test_decode_ucounts(self):
        """decode should handle 'ucounts' format."""
        rle = {'size': [2, 2], 'ucounts': [0, 1, 2, 1]}
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, np.eye(2, dtype=np.uint8))

    def test_decode_zcounts(self, eye3):
        """decode should handle 'zcounts' format."""
        rle = rlemasklib.encode(eye3, zlevel=-1)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, eye3)

    def test_encode_batch(self):
        """encode should handle batch of masks."""
        masks = np.stack([np.eye(3), np.eye(3)[::-1]], axis=2).astype(np.uint8)
        rles = rlemasklib.encode(masks)
        assert len(rles) == 2
        np.testing.assert_array_equal(rlemasklib.decode(rles[0]), np.eye(3))

    def test_encode_batch_first(self):
        """encode with batch_first=True should handle (N,H,W) arrays."""
        masks = np.stack([np.eye(3), np.eye(3)[::-1]], axis=0).astype(np.uint8)
        rles = rlemasklib.encode(masks, batch_first=True)
        assert len(rles) == 2


# =============================================================================
# Compression Tests
# =============================================================================

class TestCompression:
    def test_compress_ucounts(self, eye3):
        """compress should convert ucounts to counts."""
        rle = rlemasklib.encode(eye3, compressed=False)
        compressed = rlemasklib.compress(rle)
        assert 'counts' in compressed
        assert 'ucounts' not in compressed

    def test_compress_with_zlevel(self, eye3):
        """compress with zlevel should add zlib compression."""
        rle = rlemasklib.encode(eye3, compressed=True)
        compressed = rlemasklib.compress(rle, zlevel=-1)
        assert 'zcounts' in compressed

    def test_decompress_counts(self, eye3):
        """decompress should convert counts to ucounts."""
        rle = rlemasklib.encode(eye3, compressed=True)
        decompressed = rlemasklib.decompress(rle)
        assert 'ucounts' in decompressed

    def test_decompress_zcounts(self, eye3):
        """decompress should handle zcounts."""
        rle = rlemasklib.encode(eye3, zlevel=-1)
        decompressed = rlemasklib.decompress(rle)
        assert 'ucounts' in decompressed

    def test_decompress_only_gzip(self, eye3):
        """decompress with only_gzip=True should only remove zlib."""
        rle = rlemasklib.encode(eye3, zlevel=-1)
        decompressed = rlemasklib.decompress(rle, only_gzip=True)
        assert 'counts' in decompressed
        assert 'zcounts' not in decompressed


# =============================================================================
# Mask Creation Tests
# =============================================================================

class TestMaskCreation:
    def test_zeros(self):
        """zeros should create all-background mask."""
        rle = rlemasklib.zeros(imshape=(5, 7))
        assert rle['size'] == [5, 7]
        assert rlemasklib.area(rle) == 0

    def test_ones(self):
        """ones should create all-foreground mask."""
        rle = rlemasklib.ones(imshape=(5, 7))
        assert rle['size'] == [5, 7]
        assert rlemasklib.area(rle) == 35

    def test_zeros_imsize(self):
        """zeros with imsize should use (width, height) order."""
        rle = rlemasklib.zeros(imsize=(7, 5))  # width=7, height=5
        assert rle['size'] == [5, 7]

    def test_ones_imsize(self):
        """ones with imsize should use (width, height) order."""
        rle = rlemasklib.ones(imsize=(7, 5))
        assert rle['size'] == [5, 7]

    def test_zeros_like(self, eye3):
        """zeros_like should create zeros with same size."""
        rle = rlemasklib.encode(eye3)
        zeros = rlemasklib.zeros_like(rle)
        assert zeros['size'] == rle['size']
        assert rlemasklib.area(zeros) == 0

    def test_ones_like(self, eye3):
        """ones_like should create ones with same size."""
        rle = rlemasklib.encode(eye3)
        ones = rlemasklib.ones_like(rle)
        assert ones['size'] == rle['size']
        assert rlemasklib.area(ones) == 9

    def test_from_bbox(self):
        """from_bbox should create rectangular mask."""
        rle = rlemasklib.from_bbox([1, 2, 3, 4], imshape=(10, 10))
        decoded = rlemasklib.decode(rle)
        expected = np.zeros((10, 10), dtype=np.uint8)
        expected[2:6, 1:4] = 1
        np.testing.assert_array_equal(decoded, expected)

    def test_from_bbox_batch(self):
        """from_bbox should handle batch of bboxes."""
        bboxes = np.array([[0, 0, 2, 2], [3, 3, 2, 2]], dtype=np.float64)
        rles = rlemasklib.from_bbox(bboxes, imshape=(5, 5))
        assert len(rles) == 2

    def test_from_polygon(self):
        """from_polygon should create polygon mask."""
        # Simple triangle
        poly = np.array([0, 0, 4, 0, 2, 4], dtype=np.float64)
        rle = rlemasklib.from_polygon(poly, imshape=(5, 5))
        decoded = rlemasklib.decode(rle)
        assert decoded.sum() > 0


# =============================================================================
# Set Operations Tests
# =============================================================================

class TestSetOperations:
    def test_complement(self, eye3):
        """complement should invert mask."""
        rle = rlemasklib.encode(eye3)
        comp = rlemasklib.complement(rle)
        decoded = rlemasklib.decode(comp)
        np.testing.assert_array_equal(decoded, 1 - eye3)

    def test_complement_batch(self):
        """complement should handle list of masks."""
        masks = [np.eye(3, dtype=np.uint8), np.ones((3, 3), dtype=np.uint8)]
        rles = [rlemasklib.encode(m) for m in masks]
        comps = rlemasklib.complement(rles)
        assert len(comps) == 2

    def test_union(self, eye3, eye3_flipped):
        """union should combine masks."""
        rle1 = rlemasklib.encode(eye3)
        rle2 = rlemasklib.encode(eye3_flipped)
        result = rlemasklib.union([rle1, rle2])
        expected = eye3 | eye3_flipped
        decode_and_compare(result, expected)

    def test_intersection(self, eye3, eye3_flipped):
        """intersection should find common area."""
        rle1 = rlemasklib.encode(eye3)
        rle2 = rlemasklib.encode(eye3_flipped)
        result = rlemasklib.intersection([rle1, rle2])
        expected = eye3 & eye3_flipped
        decode_and_compare(result, expected)

    def test_difference(self, eye3, eye3_flipped):
        """difference should subtract second from first."""
        rle1 = rlemasklib.encode(eye3)
        rle2 = rlemasklib.encode(eye3_flipped)
        result = rlemasklib.difference(rle1, rle2)
        expected = eye3 & ~eye3_flipped
        decode_and_compare(result, expected)

    def test_symmetric_difference(self, eye3, eye3_flipped):
        """symmetric_difference should find XOR."""
        rle1 = rlemasklib.encode(eye3)
        rle2 = rlemasklib.encode(eye3_flipped)
        result = rlemasklib.symmetric_difference(rle1, rle2)
        expected = eye3 ^ eye3_flipped
        decode_and_compare(result, expected)

    def test_merge_with_boolfunc(self, eye3, eye3_flipped):
        """merge should apply custom BoolFunc."""
        rle1 = rlemasklib.encode(eye3)
        rle2 = rlemasklib.encode(eye3_flipped)
        # Test NAND
        result = rlemasklib.merge([rle1, rle2], BoolFunc.NAND)
        expected = ~(eye3 & eye3_flipped) & 1
        decode_and_compare(result, expected)

    def test_merge_multiple(self):
        """merge should handle more than 2 masks."""
        m1 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        m2 = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        m3 = np.array([[0, 0], [1, 0]], dtype=np.uint8)
        rles = [rlemasklib.encode(m) for m in [m1, m2, m3]]
        result = rlemasklib.merge(rles, BoolFunc.OR)
        expected = m1 | m2 | m3
        decode_and_compare(result, expected)


# =============================================================================
# BoolFunc Tests
# =============================================================================

class TestBoolFunc:
    def test_boolfunc_a(self):
        """BoolFunc.A should return first argument."""
        m1 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        m2 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.A)
        decode_and_compare(result, m1)

    def test_boolfunc_b(self):
        """BoolFunc.B should return second argument."""
        m1 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        m2 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.B)
        decode_and_compare(result, m2)

    def test_boolfunc_or(self):
        """BoolFunc.OR should compute union."""
        m1 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        m2 = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.OR)
        decode_and_compare(result, m1 | m2)

    def test_boolfunc_and(self):
        """BoolFunc.AND should compute intersection."""
        m1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        m2 = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.AND)
        decode_and_compare(result, m1 & m2)

    def test_boolfunc_xor(self):
        """BoolFunc.XOR should compute symmetric difference."""
        m1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        m2 = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.XOR)
        decode_and_compare(result, m1 ^ m2)

    def test_boolfunc_difference(self):
        """BoolFunc.DIFFERENCE should compute A & ~B."""
        m1 = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        m2 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.DIFFERENCE)
        decode_and_compare(result, m1 & (~m2 & 1))

    def test_boolfunc_nor(self):
        """BoolFunc.NOR should compute ~(A | B)."""
        m1 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        m2 = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.NOR)
        decode_and_compare(result, ~(m1 | m2) & 1)

    def test_boolfunc_nand(self):
        """BoolFunc.NAND should compute ~(A & B)."""
        m1 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        m2 = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.NAND)
        decode_and_compare(result, ~(m1 & m2) & 1)

    def test_boolfunc_implication(self):
        """BoolFunc.IMPLICATION should compute ~A | B."""
        m1 = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        m2 = np.array([[0, 1], [0, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.IMPLICATION)
        decode_and_compare(result, (~m1 | m2) & 1)

    def test_boolfunc_equivalence(self):
        """BoolFunc.EQUIVALENCE should compute ~(A ^ B)."""
        m1 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        m2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], BoolFunc.EQUIVALENCE)
        decode_and_compare(result, ~(m1 ^ m2) & 1)

    def test_boolfunc_custom_combination(self):
        """Custom BoolFunc combination should work."""
        # ~A & B using BoolFunc operators
        custom_func = ~BoolFunc.A & BoolFunc.B
        m1 = np.array([[1, 0], [1, 0]], dtype=np.uint8)
        m2 = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        result = rlemasklib.merge([rle1, rle2], custom_func)
        expected = (~m1 & m2) & 1
        decode_and_compare(result, expected)


# =============================================================================
# Geometric Operations Tests
# =============================================================================

class TestGeometricOperations:
    def test_crop(self, rect_5x5):
        """crop should extract rectangular region."""
        rle = rlemasklib.encode(rect_5x5)
        cropped = rlemasklib.crop(rle, [1, 1, 3, 3])
        decoded = rlemasklib.decode(cropped)
        expected = rect_5x5[1:4, 1:4]
        np.testing.assert_array_equal(decoded, expected)

    def test_crop_batch(self):
        """crop should handle batch of masks and bboxes."""
        m1 = np.ones((5, 5), dtype=np.uint8)
        m2 = np.zeros((5, 5), dtype=np.uint8)
        m2[2, 2] = 1
        rles = [rlemasklib.encode(m) for m in [m1, m2]]
        bboxes = np.array([[1, 1, 2, 2], [1, 1, 3, 3]], dtype=np.uint32)
        cropped = rlemasklib.crop(rles, bboxes)
        assert len(cropped) == 2

    def test_pad(self, eye3):
        """pad should add border around mask."""
        rle = rlemasklib.encode(eye3)
        padded = rlemasklib.pad(rle, (1, 2, 3, 4))  # left, right, top, bottom
        decoded = rlemasklib.decode(padded)
        assert decoded.shape == (10, 6)  # 3+3+4, 3+1+2

    def test_pad_with_value(self, eye3):
        """pad with value=1 should add foreground border."""
        rle = rlemasklib.encode(eye3)
        padded = rlemasklib.pad(rle, (1, 1, 1, 1), value=1)
        decoded = rlemasklib.decode(padded)
        assert decoded[0, :].sum() == 5  # top row all ones

    def test_shift(self, eye3):
        """shift should translate mask."""
        rle = rlemasklib.encode(eye3)
        shifted = rlemasklib.shift(rle, (1, 1))  # dy, dx
        decoded = rlemasklib.decode(shifted)
        assert decoded[0, :].sum() == 0
        assert decoded[:, 0].sum() == 0

    def test_shift_zero(self, eye3):
        """shift by (0, 0) should return same mask."""
        rle = rlemasklib.encode(eye3)
        shifted = rlemasklib.shift(rle, (0, 0))
        decode_and_compare(shifted, eye3)

    def test_to_bbox(self, rect_5x5):
        """to_bbox should return bounding box."""
        rle = rlemasklib.encode(rect_5x5)
        bbox = rlemasklib.to_bbox(rle)
        # rect_5x5 has 1s at [1:4, 1:4]
        np.testing.assert_array_equal(bbox, [1, 1, 3, 3])

    def test_to_bbox_batch(self):
        """to_bbox should handle batch of masks."""
        m1 = np.zeros((5, 5), dtype=np.uint8)
        m1[0:2, 0:2] = 1
        m2 = np.zeros((5, 5), dtype=np.uint8)
        m2[3:5, 3:5] = 1
        rles = [rlemasklib.encode(m1), rlemasklib.encode(m2)]
        bboxes = rlemasklib.to_bbox(rles)
        assert bboxes.shape == (2, 4)


# =============================================================================
# Morphological Operations Tests
# =============================================================================

class TestMorphology:
    def test_dilate(self):
        """dilate should expand mask."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = rlemasklib.encode(mask)
        dilated = rlemasklib.dilate(rle, connectivity=4)
        decoded = rlemasklib.decode(dilated)
        # Should form cross
        assert decoded.sum() == 5

    def test_dilate_8_connectivity(self):
        """dilate with connectivity=8 should include diagonals."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = rlemasklib.encode(mask)
        dilated = rlemasklib.dilate(rle, connectivity=8)
        decoded = rlemasklib.decode(dilated)
        # Should form 3x3 square
        assert decoded[1:4, 1:4].sum() == 9

    def test_erode(self):
        """erode should shrink mask."""
        # Use a mask with explicit zeros around it to avoid boundary handling issues
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1  # 5x5 block of ones
        rle = rlemasklib.encode(mask)
        eroded = rlemasklib.erode(rle, connectivity=4)
        decoded = rlemasklib.decode(eroded)
        # 5x5 block should shrink to 3x3
        assert decoded[2:5, 2:5].sum() == 9
        assert decoded.sum() == 9

    def test_opening(self):
        """opening should remove small protrusions."""
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[2:5, 2:5] = 1  # 3x3 block
        mask[0, 0] = 1  # isolated pixel
        rle = rlemasklib.encode(mask)
        opened = rlemasklib.opening(rle)
        decoded = rlemasklib.decode(opened)
        assert decoded[0, 0] == 0

    def test_closing(self):
        """closing should fill small holes."""
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[2, 2] = 0  # hole
        rle = rlemasklib.encode(mask)
        closed = rlemasklib.closing(rle)
        decoded = rlemasklib.decode(closed)
        assert decoded[2, 2] == 1

    def test_dilate2(self):
        """dilate2 should use 5x5 kernel."""
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 1
        rle = rlemasklib.encode(mask)
        dilated = rlemasklib.dilate2(rle)
        decoded = rlemasklib.decode(dilated)
        # Should expand by 2 pixels in each direction (with rounded corners)
        assert decoded[2, 4] == 1
        assert decoded[4, 2] == 1

    def test_erode2(self):
        """erode2 should use 5x5 kernel."""
        # Use a mask with explicit zeros around it to avoid boundary handling issues
        mask = np.zeros((13, 13), dtype=np.uint8)
        mask[2:11, 2:11] = 1  # 9x9 block of ones
        rle = rlemasklib.encode(mask)
        eroded = rlemasklib.erode2(rle)
        decoded = rlemasklib.decode(eroded)
        # 9x9 block should shrink to 5x5 (erode by 2 pixels on each side)
        assert decoded[4:9, 4:9].sum() == 25
        assert decoded.sum() == 25

    def test_opening2(self):
        """opening2 should use 5x5 kernel."""
        mask = np.zeros((11, 11), dtype=np.uint8)
        mask[3:8, 3:8] = 1  # 5x5 block
        mask[0, 0] = 1  # isolated pixel
        rle = rlemasklib.encode(mask)
        opened = rlemasklib.opening2(rle)
        decoded = rlemasklib.decode(opened)
        assert decoded[0, 0] == 0

    def test_closing2(self):
        """closing2 should use 5x5 kernel."""
        mask = np.ones((11, 11), dtype=np.uint8)
        mask[5, 5] = 0  # small hole
        rle = rlemasklib.encode(mask)
        closed = rlemasklib.closing2(rle)
        decoded = rlemasklib.decode(closed)
        assert decoded[5, 5] == 1


# =============================================================================
# Connected Components Tests
# =============================================================================

class TestConnectedComponents:
    def test_connected_components(self):
        """connected_components should find separate regions."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        mask[4, 4] = 1
        rle = rlemasklib.encode(mask)
        components = rlemasklib.connected_components(rle, connectivity=4)
        assert len(components) == 2

    def test_connected_components_8_connectivity(self):
        """connected_components with 8-connectivity should connect diagonals."""
        mask = np.eye(3, dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        comp4 = rlemasklib.connected_components(rle, connectivity=4)
        comp8 = rlemasklib.connected_components(rle, connectivity=8)
        assert len(comp4) == 3
        assert len(comp8) == 1

    def test_connected_components_min_size(self):
        """connected_components should filter by min_size."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels
        mask[5:10, 5:10] = 1  # 25 pixels
        rle = rlemasklib.encode(mask)
        components = rlemasklib.connected_components(rle, connectivity=4, min_size=10)
        assert len(components) == 1

    def test_largest_connected_component(self):
        """largest_connected_component should return biggest region."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels
        mask[5:10, 5:10] = 1  # 25 pixels
        rle = rlemasklib.encode(mask)
        largest = rlemasklib.largest_connected_component(rle)
        assert rlemasklib.area(largest) == 25

    def test_remove_small_components(self):
        """remove_small_components should remove small regions."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels
        mask[5:10, 5:10] = 1  # 25 pixels
        rle = rlemasklib.encode(mask)
        cleaned = rlemasklib.remove_small_components(rle, connectivity=4, min_size=10)
        decoded = rlemasklib.decode(cleaned)
        assert decoded[0:2, 0:2].sum() == 0
        assert decoded[5:10, 5:10].sum() == 25

    def test_fill_small_holes(self):
        """fill_small_holes should fill small background regions."""
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[2, 2] = 0  # small hole
        rle = rlemasklib.encode(mask)
        filled = rlemasklib.fill_small_holes(rle, connectivity=4, min_size=2)
        decoded = rlemasklib.decode(filled)
        assert decoded[2, 2] == 1


# =============================================================================
# Analysis Tests
# =============================================================================

class TestAnalysis:
    def test_area(self, eye3):
        """area should count foreground pixels."""
        rle = rlemasklib.encode(eye3)
        assert rlemasklib.area(rle) == 3

    def test_area_batch(self):
        """area should handle list of masks."""
        masks = [np.eye(3, dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)]
        rles = [rlemasklib.encode(m) for m in masks]
        areas = rlemasklib.area(rles)
        np.testing.assert_array_equal(areas, [3, 16])

    def test_centroid(self):
        """centroid should return center of mass."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = rlemasklib.encode(mask)
        cx, cy = rlemasklib.centroid(rle)
        assert cx == 2.0
        assert cy == 2.0

    def test_centroid_batch(self):
        """centroid should handle list of masks."""
        m1 = np.zeros((5, 5), dtype=np.uint8)
        m1[2, 2] = 1
        m2 = np.zeros((5, 5), dtype=np.uint8)
        m2[0, 0] = 1
        rles = [rlemasklib.encode(m) for m in [m1, m2]]
        centroids = rlemasklib.centroid(rles)
        assert centroids.shape == (2, 2)

    def test_iou(self, eye3):
        """iou should compute intersection over union."""
        rle = rlemasklib.encode(eye3)
        assert rlemasklib.iou([rle, rle]) == 1.0

    def test_iou_no_overlap(self):
        """iou of non-overlapping masks should be 0."""
        m1 = np.zeros((5, 5), dtype=np.uint8)
        m1[0, 0] = 1
        m2 = np.zeros((5, 5), dtype=np.uint8)
        m2[4, 4] = 1
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        assert rlemasklib.iou([rle1, rle2]) == 0.0


# =============================================================================
# Edge Cases Tests
# =============================================================================

class TestEdgeCases:
    def test_empty_mask(self):
        """Operations on empty mask should work."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        assert rlemasklib.area(rle) == 0
        assert rlemasklib.largest_connected_component(rle) is None

    def test_full_mask(self):
        """Operations on full mask should work."""
        mask = np.ones((5, 5), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        assert rlemasklib.area(rle) == 25

    def test_single_pixel(self):
        """Operations on 1x1 mask should work."""
        mask = np.array([[1]], dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_single_row(self):
        """Operations on 1xN mask should work."""
        mask = np.array([[0, 1, 0, 1, 1]], dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_single_column(self):
        """Operations on Nx1 mask should work."""
        mask = np.array([[0], [1], [0], [1], [1]], dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_large_mask(self):
        """Operations on large mask should work."""
        mask = np.random.randint(0, 2, (500, 500), dtype=np.uint8)
        rle = rlemasklib.encode(mask)
        decoded = rlemasklib.decode(rle)
        np.testing.assert_array_equal(decoded, mask)

    def test_different_size_error(self):
        """Operations on different-sized masks should raise error."""
        m1 = np.eye(3, dtype=np.uint8)
        m2 = np.eye(5, dtype=np.uint8)
        rle1 = rlemasklib.encode(m1)
        rle2 = rlemasklib.encode(m2)
        with pytest.raises((ValueError, AssertionError)):
            rlemasklib.union([rle1, rle2])

    def test_decode_invalid_size(self):
        """decode with invalid size should raise error."""
        rle = rlemasklib.encode(np.eye(3, dtype=np.uint8))
        rle['size'] = [2, 2]  # wrong size
        with pytest.raises(ValueError):
            rlemasklib.decode(rle)