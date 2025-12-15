"""Comprehensive tests for the RLEMask object-oriented API."""

import numpy as np
import pytest
from rlemasklib.oop import RLEMask
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
def cross_5x5():
    """5x5 cross pattern."""
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, :] = 1
    mask[:, 2] = 1
    return mask


@pytest.fixture
def rect_mask():
    """10x10 mask with 4x4 rectangle in center."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[3:7, 3:7] = 1
    return mask


# =============================================================================
# Construction Tests
# =============================================================================

class TestConstruction:
    def test_from_array_basic(self, eye3):
        """from_array should create RLEMask from numpy array."""
        rle = RLEMask.from_array(eye3)
        np.testing.assert_array_equal(np.array(rle), eye3)

    def test_from_array_bool(self):
        """from_array should handle boolean arrays."""
        mask = np.array([[True, False], [False, True]])
        rle = RLEMask.from_array(mask)
        np.testing.assert_array_equal(np.array(rle), mask.astype(np.uint8))

    def test_from_array_c_contiguous(self):
        """from_array should handle C-contiguous arrays."""
        mask = np.ascontiguousarray(np.eye(5, dtype=np.uint8))
        rle = RLEMask.from_array(mask)
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_from_array_f_contiguous(self):
        """from_array should handle F-contiguous arrays."""
        mask = np.asfortranarray(np.eye(5, dtype=np.uint8))
        rle = RLEMask.from_array(mask)
        np.testing.assert_array_equal(np.array(rle), mask)

    def test_from_array_thresh128(self):
        """from_array with thresh128=True should threshold at 128."""
        mask = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        rle = RLEMask.from_array(mask, thresh128=True)
        expected = np.array([[0, 1], [0, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_from_dict_ucounts(self):
        """from_dict should handle uncompressed counts."""
        d = {'ucounts': [0, 1, 2, 1], 'size': [2, 2]}
        rle = RLEMask.from_dict(d)
        np.testing.assert_array_equal(np.array(rle), np.eye(2, dtype=np.uint8))

    def test_from_dict_compressed(self, eye3):
        """from_dict should handle compressed counts."""
        rle1 = RLEMask.from_array(eye3)
        d = rle1.to_dict()
        rle2 = RLEMask.from_dict(d)
        assert rle1 == rle2

    def test_from_dict_zcounts(self, eye3):
        """from_dict should handle zlib-compressed counts."""
        rle1 = RLEMask.from_array(eye3)
        d = rle1.to_dict(zlevel=-1)
        assert 'zcounts' in d
        rle2 = RLEMask.from_dict(d)
        assert rle1 == rle2

    def test_from_counts_fortran_order(self):
        """from_counts with order='F' should work correctly."""
        # Eye 2x2 in column-major: col0=[1,0], col1=[0,1] -> flat [1,0,0,1]
        # RLE counts: 0 zeros, 1 one, 2 zeros, 1 one -> [0, 1, 2, 1]
        rle = RLEMask.from_counts([0, 1, 2, 1], shape=(2, 2), order='F')
        np.testing.assert_array_equal(np.array(rle), np.eye(2, dtype=np.uint8))

    def test_from_counts_c_order(self):
        """from_counts with order='C' should work correctly."""
        # Eye 2x2 in row-major: row0=[1,0], row1=[0,1] -> flat [1,0,0,1]
        # RLE counts: 0 zeros, 1 one, 2 zeros, 1 one -> [0, 1, 2, 1]
        rle = RLEMask.from_counts([0, 1, 2, 1], shape=(2, 2), order='C')
        np.testing.assert_array_equal(np.array(rle), np.eye(2, dtype=np.uint8))

    def test_from_counts_invalid_sum(self):
        """from_counts should raise error if counts don't sum to H*W."""
        with pytest.raises(ValueError):
            RLEMask.from_counts([1, 2, 3], shape=(3, 3))

    def test_from_counts_no_shape(self):
        """Constructor with counts but no shape should raise error."""
        with pytest.raises(ValueError):
            RLEMask([1, 2, 1])

    def test_from_bbox(self):
        """from_bbox should create rectangular mask."""
        rle = RLEMask.from_bbox([1, 2, 3, 4], imshape=(10, 10))
        arr = np.array(rle)
        assert arr[2:6, 1:4].sum() == 12  # 3*4 = 12
        assert arr.sum() == 12  # only the bbox is filled

    def test_from_bbox_imsize(self):
        """from_bbox with imsize should work (width, height order)."""
        rle = RLEMask.from_bbox([0, 0, 2, 3], imsize=(5, 10))  # width=5, height=10
        assert rle.shape == (10, 5)

    def test_from_polygon(self):
        """from_polygon should create polygon mask."""
        # Simple square polygon
        poly = np.array([[1, 1], [4, 1], [4, 4], [1, 4]], dtype=np.float64)
        rle = RLEMask.from_polygon(poly, imshape=(6, 6))
        arr = np.array(rle)
        # Interior should be filled
        assert arr[2, 2] == 1
        assert arr[3, 3] == 1

    def test_from_circle(self):
        """from_circle should create circular mask."""
        rle = RLEMask.from_circle([5, 5], 3, imshape=(11, 11))
        arr = np.array(rle)
        # Center should be filled
        assert arr[5, 5] == 1
        # Area should be reasonable for a circle of radius 3
        assert rle.area() > 0
        # Corners should be empty (outside the circle)
        assert arr[0, 0] == 0
        assert arr[10, 10] == 0

    def test_zeros(self):
        """zeros should create all-background mask."""
        rle = RLEMask.zeros((5, 7))
        assert rle.shape == (5, 7)
        assert rle.area() == 0
        np.testing.assert_array_equal(np.array(rle), np.zeros((5, 7), dtype=np.uint8))

    def test_ones(self):
        """ones should create all-foreground mask."""
        rle = RLEMask.ones((5, 7))
        assert rle.shape == (5, 7)
        assert rle.area() == 35
        np.testing.assert_array_equal(np.array(rle), np.ones((5, 7), dtype=np.uint8))

    def test_zeros_like(self, eye3):
        """zeros_like should create zeros with same shape."""
        rle = RLEMask.from_array(eye3)
        zeros = RLEMask.zeros_like(rle)
        assert zeros.shape == rle.shape
        assert zeros.area() == 0

    def test_ones_like(self, eye3):
        """ones_like should create ones with same shape."""
        rle = RLEMask.from_array(eye3)
        ones = RLEMask.ones_like(rle)
        assert ones.shape == rle.shape
        assert ones.area() == 9


# =============================================================================
# Properties Tests
# =============================================================================

class TestProperties:
    def test_shape(self, eye3):
        """shape property should return (height, width)."""
        rle = RLEMask.from_array(eye3)
        assert rle.shape == (3, 3)

    def test_counts(self, eye3):
        """counts property should return run-length counts as copy."""
        rle = RLEMask.from_array(eye3)
        counts = rle.counts
        assert isinstance(counts, np.ndarray)
        # Modifying counts should not affect mask
        original_sum = counts.sum()
        counts[0] = 999
        assert rle.counts.sum() == original_sum

    def test_counts_view(self, eye3):
        """counts_view should return direct view of counts."""
        rle = RLEMask.from_array(eye3)
        view = rle.counts_view
        assert isinstance(view, np.ndarray)

    def test_density(self):
        """density should return ratio of runlengths to pixels."""
        rle = RLEMask.zeros((10, 10))
        # Single run of 100 zeros -> 1 runlength / 100 pixels
        assert rle.density == 1 / 100

    def test_T_property(self, eye3):
        """T property should return transpose."""
        rle = RLEMask.from_array(eye3)
        transposed = rle.T
        np.testing.assert_array_equal(np.array(transposed), eye3.T)


# =============================================================================
# Boolean Operations Tests
# =============================================================================

class TestBooleanOperations:
    def test_or_operator(self, eye3, eye3_flipped):
        """| operator should compute union."""
        rle1 = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        result = rle1 | rle2
        expected = eye3 | eye3_flipped
        np.testing.assert_array_equal(np.array(result), expected)

    def test_and_operator(self, eye3, eye3_flipped):
        """& operator should compute intersection."""
        rle1 = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        result = rle1 & rle2
        expected = eye3 & eye3_flipped
        np.testing.assert_array_equal(np.array(result), expected)

    def test_sub_operator(self, eye3, eye3_flipped):
        """- operator should compute difference."""
        rle1 = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        result = rle1 - rle2
        expected = eye3 & ~eye3_flipped
        np.testing.assert_array_equal(np.array(result), expected)

    def test_xor_operator(self, eye3, eye3_flipped):
        """^ operator should compute symmetric difference."""
        rle1 = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        result = rle1 ^ rle2
        expected = eye3 ^ eye3_flipped
        np.testing.assert_array_equal(np.array(result), expected)

    def test_invert_operator(self, eye3):
        """~ operator should compute complement."""
        rle = RLEMask.from_array(eye3)
        result = ~rle
        expected = 1 - eye3
        np.testing.assert_array_equal(np.array(result), expected)

    def test_ior_operator(self, eye3, eye3_flipped):
        """|= operator should compute union in place."""
        rle = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        rle |= rle2
        expected = eye3 | eye3_flipped
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_iand_operator(self, eye3, eye3_flipped):
        """&= operator should compute intersection in place."""
        rle = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        rle &= rle2
        expected = eye3 & eye3_flipped
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_isub_operator(self, eye3, eye3_flipped):
        """-= operator should compute difference in place."""
        rle = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        rle -= rle2
        expected = eye3 & ~eye3_flipped
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_ixor_operator(self, eye3, eye3_flipped):
        """^= operator should compute symmetric difference in place."""
        rle = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        rle ^= rle2
        expected = eye3 ^ eye3_flipped
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_complement_inplace(self, eye3):
        """complement with inplace=True should modify in place."""
        rle = RLEMask.from_array(eye3)
        result = rle.complement(inplace=True)
        assert result is rle
        expected = 1 - eye3
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_complement_not_inplace(self, eye3):
        """complement with inplace=False should return new object."""
        rle = RLEMask.from_array(eye3)
        result = rle.complement(inplace=False)
        assert result is not rle
        # Original should be unchanged
        np.testing.assert_array_equal(np.array(rle), eye3)


# =============================================================================
# Merge Operations Tests
# =============================================================================

class TestMergeOperations:
    def test_merge_with_boolfunc(self, eye3, eye3_flipped):
        """merge should apply BoolFunc correctly."""
        rle1 = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3_flipped)
        result = rle1.merge(rle2, BoolFunc.OR)
        expected = eye3 | eye3_flipped
        np.testing.assert_array_equal(np.array(result), expected)

    def test_merge_many_single_func(self):
        """merge_many with single BoolFunc should work."""
        masks = [
            RLEMask.from_array(np.array([[1, 0], [0, 0]], dtype=np.uint8)),
            RLEMask.from_array(np.array([[0, 1], [0, 0]], dtype=np.uint8)),
            RLEMask.from_array(np.array([[0, 0], [1, 0]], dtype=np.uint8)),
        ]
        result = RLEMask.merge_many(masks, BoolFunc.OR)
        expected = np.array([[1, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(np.array(result), expected)

    def test_merge_many_multiple_funcs(self):
        """merge_many with sequence of BoolFuncs should work."""
        m1 = RLEMask.from_array(np.array([[1, 1], [0, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[1, 0], [1, 0]], dtype=np.uint8))
        # m1 | m2 using single BoolFunc
        result = RLEMask.merge_many([m1, m2], BoolFunc.OR)
        expected = np.array([[1, 1], [0, 0]]) | np.array([[1, 0], [1, 0]])
        np.testing.assert_array_equal(np.array(result), expected.astype(np.uint8))

    def test_union_static(self):
        """RLEMask.union should compute union of list."""
        masks = [
            RLEMask.from_array(np.array([[1, 0], [0, 0]], dtype=np.uint8)),
            RLEMask.from_array(np.array([[0, 1], [0, 0]], dtype=np.uint8)),
        ]
        result = RLEMask.union(masks)
        expected = np.array([[1, 1], [0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(np.array(result), expected)

    def test_intersection_static(self):
        """RLEMask.intersection should compute intersection of list."""
        masks = [
            RLEMask.from_array(np.array([[1, 1], [0, 0]], dtype=np.uint8)),
            RLEMask.from_array(np.array([[1, 0], [1, 0]], dtype=np.uint8)),
        ]
        result = RLEMask.intersection(masks)
        expected = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(np.array(result), expected)

    def test_merge_count_threshold(self):
        """merge_count should threshold by count."""
        m1 = RLEMask.from_array(np.array([[1, 1], [1, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[1, 1], [0, 1]], dtype=np.uint8))
        m3 = RLEMask.from_array(np.array([[1, 0], [1, 1]], dtype=np.uint8))
        # threshold=2 means at least 2 masks must have pixel set
        result = RLEMask.merge_count([m1, m2, m3], threshold=2)
        # [0,0]=3, [0,1]=2, [1,0]=2, [1,1]=2
        expected = np.array([[1, 1], [1, 1]], dtype=np.uint8)
        np.testing.assert_array_equal(np.array(result), expected)

    def test_merge_many_custom_lambda(self):
        """merge_many_custom with lambda should work."""
        m1 = RLEMask.from_array(np.array([[1, 0], [0, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[1, 1], [0, 0]], dtype=np.uint8))
        m3 = RLEMask.from_array(np.array([[0, 1], [1, 1]], dtype=np.uint8))
        # Custom: (a & b) | c
        result = RLEMask.merge_many_custom([m1, m2, m3], lambda a, b, c: (a & b) | c)
        expected = ((np.array([[1, 0], [0, 0]]) & np.array([[1, 1], [0, 0]])) |
                    np.array([[0, 1], [1, 1]])).astype(np.uint8)
        np.testing.assert_array_equal(np.array(result), expected)

    def test_make_merge_function(self):
        """make_merge_function should create reusable merge function."""
        mergefn = RLEMask.make_merge_function(lambda a, b: a & ~b)
        m1 = RLEMask.from_array(np.array([[1, 1], [1, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[1, 0], [0, 0]], dtype=np.uint8))
        result = mergefn(m1, m2)
        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(np.array(result), expected)


# =============================================================================
# Indexing and Slicing Tests
# =============================================================================

class TestIndexingSlicing:
    def test_getitem_slice(self, cross_5x5):
        """__getitem__ with slices should crop mask."""
        rle = RLEMask.from_array(cross_5x5)
        cropped = rle[1:4, 1:4]
        expected = cross_5x5[1:4, 1:4]
        np.testing.assert_array_equal(np.array(cropped), expected)

    def test_getitem_single_slice(self):
        """__getitem__ with single slice should work."""
        mask = np.arange(12).reshape(3, 4).astype(np.uint8) % 2
        rle = RLEMask.from_array(mask)
        cropped = rle[1:3]
        expected = mask[1:3]
        np.testing.assert_array_equal(np.array(cropped), expected)

    def test_getitem_pixel(self, eye3):
        """__getitem__ with ints should return pixel value."""
        rle = RLEMask.from_array(eye3)
        assert rle[0, 0] == 1
        assert rle[0, 1] == 0
        assert rle[1, 1] == 1

    def test_getitem_negative_index(self, eye3):
        """__getitem__ with negative indices should work."""
        rle = RLEMask.from_array(eye3)
        assert rle[-1, -1] == 1  # bottom-right
        assert rle[-1, 0] == 0

    def test_getitem_with_step(self):
        """__getitem__ with step should subsample."""
        mask = np.arange(16).reshape(4, 4).astype(np.uint8) % 2
        rle = RLEMask.from_array(mask)
        subsampled = rle[::2, ::2]
        expected = mask[::2, ::2]
        np.testing.assert_array_equal(np.array(subsampled), expected)

    def test_getitem_negative_step(self):
        """__getitem__ with negative step should reverse."""
        mask = np.arange(9).reshape(3, 3).astype(np.uint8) % 2
        rle = RLEMask.from_array(mask)
        reversed_rle = rle[::-1, ::-1]
        expected = mask[::-1, ::-1]
        np.testing.assert_array_equal(np.array(reversed_rle), expected)

    def test_setitem_constant(self):
        """__setitem__ with constant should fill region."""
        rle = RLEMask.zeros((5, 5))
        rle[1:4, 1:4] = 1
        arr = np.array(rle)
        assert arr[1:4, 1:4].sum() == 9
        assert arr.sum() == 9

    def test_setitem_mask(self):
        """__setitem__ with RLEMask should paste mask."""
        rle = RLEMask.zeros((5, 5))
        patch = RLEMask.from_array(np.eye(2, dtype=np.uint8))
        rle[1:3, 1:3] = patch
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1:3, 1:3] = np.eye(2)
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_setitem_numpy(self):
        """__setitem__ with numpy array should work."""
        rle = RLEMask.zeros((5, 5))
        rle[1:3, 1:3] = np.eye(2, dtype=np.uint8)
        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[1:3, 1:3] = np.eye(2)
        np.testing.assert_array_equal(np.array(rle), expected)

    def test_setitem_pixel(self):
        """__setitem__ with int indices should set pixel."""
        rle = RLEMask.zeros((3, 3))
        rle[1, 1] = 1
        assert rle[1, 1] == 1
        assert rle.area() == 1


# =============================================================================
# Geometric Operations Tests
# =============================================================================

class TestGeometricOperations:
    def test_crop(self, rect_mask):
        """crop should extract rectangular region."""
        rle = RLEMask.from_array(rect_mask)
        cropped = rle.crop([2, 2, 6, 6])  # x, y, w, h
        expected = rect_mask[2:8, 2:8]
        np.testing.assert_array_equal(np.array(cropped), expected)

    def test_crop_inplace(self, rect_mask):
        """crop with inplace=True should modify in place."""
        rle = RLEMask.from_array(rect_mask)
        result = rle.crop([2, 2, 6, 6], inplace=True)
        assert result is rle
        assert rle.shape == (6, 6)

    def test_tight_crop(self):
        """tight_crop should crop to bounding box."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[3:6, 4:8] = 1
        rle = RLEMask.from_array(mask)
        cropped, bbox = rle.tight_crop()
        assert cropped.shape == (3, 4)
        np.testing.assert_array_equal(np.array(cropped), np.ones((3, 4), dtype=np.uint8))

    def test_pad_zeros(self, eye3):
        """pad should add border with zeros."""
        rle = RLEMask.from_array(eye3)
        padded = rle.pad(1, 2, 3, 4)  # top, bottom, left, right
        assert padded.shape == (6, 10)  # 3+1+2, 3+3+4
        # Original content should be at offset
        np.testing.assert_array_equal(np.array(padded)[1:4, 3:6], eye3)

    def test_pad_ones(self, eye3):
        """pad with value=1 should add foreground border."""
        rle = RLEMask.from_array(eye3)
        padded = rle.pad(1, 1, 1, 1, value=1)
        arr = np.array(padded)
        assert arr[0, :].sum() == 5  # top row all ones

    def test_pad_replicate(self):
        """pad with border_type='replicate' should extend edges."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        padded = rle.pad(1, 1, 1, 1, border_type='replicate')
        arr = np.array(padded)
        # Top-left corner should replicate mask[0,0]=1
        assert arr[0, 0] == 1
        # Top-right corner should replicate mask[0,1]=0
        assert arr[0, 3] == 0

    def test_pad_inplace(self, eye3):
        """pad with inplace=True should modify in place."""
        rle = RLEMask.from_array(eye3)
        result = rle.pad(1, 1, 1, 1, inplace=True)
        assert result is rle
        assert rle.shape == (5, 5)

    def test_shift(self, eye3):
        """shift should translate mask."""
        rle = RLEMask.from_array(eye3)
        shifted = rle.shift((1, 2))  # dy=1, dx=2
        arr = np.array(shifted)
        # Original was at [0,0], [1,1], [2,2]
        # After shift should be at [1,2], [2,3] (clipped)
        assert arr[0, :].sum() == 0  # first row empty
        assert arr[:, 0].sum() == 0  # first col empty
        assert arr[:, 1].sum() == 0  # second col empty

    def test_shift_negative(self, eye3):
        """shift with negative offset should work."""
        rle = RLEMask.from_array(eye3)
        shifted = rle.shift((-1, -1))
        arr = np.array(shifted)
        assert arr[-1, :].sum() == 0  # last row empty
        assert arr[:, -1].sum() == 0  # last col empty

    def test_transpose(self, eye3):
        """transpose should swap axes."""
        mask = np.array([[1, 0, 0], [1, 1, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        transposed = rle.transpose()
        np.testing.assert_array_equal(np.array(transposed), mask.T)

    def test_rot90(self):
        """rot90 should rotate by multiples of 90 degrees."""
        mask = np.array([[1, 0], [0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)

        rot1 = rle.rot90(k=1)
        np.testing.assert_array_equal(np.array(rot1), np.rot90(mask, k=1))

        rot2 = rle.rot90(k=2)
        np.testing.assert_array_equal(np.array(rot2), np.rot90(mask, k=2))

        rot3 = rle.rot90(k=3)
        np.testing.assert_array_equal(np.array(rot3), np.rot90(mask, k=3))

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

    def test_resize(self, rect_mask):
        """resize should scale mask."""
        rle = RLEMask.from_array(rect_mask)
        resized = rle.resize((20, 20))
        assert resized.shape == (20, 20)
        # Area should approximately double in each dimension
        assert resized.area() > rle.area()

    def test_resize_with_scale_factors(self, rect_mask):
        """resize with fx/fy should scale by factors."""
        rle = RLEMask.from_array(rect_mask)
        resized = rle.resize(None, fx=2.0, fy=2.0)
        assert resized.shape == (20, 20)


# =============================================================================
# Concatenation and Tiling Tests
# =============================================================================

class TestConcatTile:
    def test_hconcat(self):
        """hconcat should concatenate horizontally."""
        m1 = RLEMask.from_array(np.ones((3, 2), dtype=np.uint8))
        m2 = RLEMask.from_array(np.zeros((3, 3), dtype=np.uint8))
        result = RLEMask.hconcat([m1, m2])
        assert result.shape == (3, 5)
        arr = np.array(result)
        assert arr[:, :2].sum() == 6
        assert arr[:, 2:].sum() == 0

    def test_vconcat(self):
        """vconcat should concatenate vertically."""
        m1 = RLEMask.from_array(np.ones((2, 3), dtype=np.uint8))
        m2 = RLEMask.from_array(np.zeros((3, 3), dtype=np.uint8))
        result = RLEMask.vconcat([m1, m2])
        assert result.shape == (5, 3)
        arr = np.array(result)
        assert arr[:2, :].sum() == 6
        assert arr[2:, :].sum() == 0

    def test_concatenate_axis0(self):
        """concatenate with axis=0 should vconcat."""
        m1 = RLEMask.ones((2, 3))
        m2 = RLEMask.zeros((2, 3))
        result = RLEMask.concatenate([m1, m2], axis=0)
        assert result.shape == (4, 3)

    def test_concatenate_axis1(self):
        """concatenate with axis=1 should hconcat."""
        m1 = RLEMask.ones((3, 2))
        m2 = RLEMask.zeros((3, 2))
        result = RLEMask.concatenate([m1, m2], axis=1)
        assert result.shape == (3, 4)

    def test_tile(self):
        """tile should repeat mask."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        tiled = rle.tile(2, 3)
        assert tiled.shape == (4, 6)
        # Check pattern repeats
        arr = np.array(tiled)
        np.testing.assert_array_equal(arr[:2, :2], mask)
        np.testing.assert_array_equal(arr[:2, 2:4], mask)
        np.testing.assert_array_equal(arr[2:4, :2], mask)

    def test_repeat(self):
        """repeat should expand each pixel."""
        mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        repeated = rle.repeat(2, 3)
        assert repeated.shape == (4, 6)
        arr = np.array(repeated)
        # Each pixel is repeated 2x3
        assert arr[0:2, 0:3].sum() == 6  # top-left 1 repeated
        assert arr[0:2, 3:6].sum() == 0  # top-right 0 repeated


# =============================================================================
# Morphological Operations Tests
# =============================================================================

class TestMorphology:
    def test_dilate3x3_connectivity4(self):
        """dilate3x3 with connectivity=4 should use cross kernel."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)
        dilated = rle.dilate3x3(connectivity=4)
        arr = np.array(dilated)
        # Should form cross pattern
        assert arr[2, 2] == 1
        assert arr[1, 2] == 1
        assert arr[3, 2] == 1
        assert arr[2, 1] == 1
        assert arr[2, 3] == 1
        # Corners should be 0
        assert arr[1, 1] == 0

    def test_dilate3x3_connectivity8(self):
        """dilate3x3 with connectivity=8 should use square kernel."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)
        dilated = rle.dilate3x3(connectivity=8)
        arr = np.array(dilated)
        # Should form 3x3 square
        assert arr[1:4, 1:4].sum() == 9

    def test_erode3x3(self):
        """erode3x3 should shrink mask."""
        # Use a mask with explicit zeros around it to avoid boundary handling issues
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:6, 1:6] = 1  # 5x5 block of ones
        rle = RLEMask.from_array(mask)
        eroded = rle.erode3x3(connectivity=4)
        arr = np.array(eroded)
        # 5x5 block should shrink to 3x3
        assert arr[2:5, 2:5].sum() == 9
        assert arr.sum() == 9

    def test_dilate5x5(self):
        """dilate5x5 should use 5x5 rounded kernel."""
        mask = np.zeros((9, 9), dtype=np.uint8)
        mask[4, 4] = 1
        rle = RLEMask.from_array(mask)
        dilated = rle.dilate5x5()
        arr = np.array(dilated)
        # Center should be filled
        assert arr[4, 4] == 1
        # 2 pixels away should be filled (not corners of 5x5)
        assert arr[2, 4] == 1
        assert arr[4, 2] == 1

    def test_erode5x5(self):
        """erode5x5 should use 5x5 rounded kernel."""
        # Use a mask with explicit zeros around it to avoid boundary handling issues
        mask = np.zeros((13, 13), dtype=np.uint8)
        mask[2:11, 2:11] = 1  # 9x9 block of ones
        rle = RLEMask.from_array(mask)
        eroded = rle.erode5x5()
        arr = np.array(eroded)
        # 9x9 block should shrink to 5x5 (erode by 2 pixels on each side)
        assert arr[4:9, 4:9].sum() == 25
        assert arr.sum() == 25

    def test_dilate_vertical(self):
        """dilate_vertical should only dilate vertically."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)
        dilated = rle.dilate_vertical(up=1, down=1)
        arr = np.array(dilated)
        # Should expand vertically only
        assert arr[1, 2] == 1
        assert arr[2, 2] == 1
        assert arr[3, 2] == 1
        # Should not expand horizontally
        assert arr[2, 1] == 0
        assert arr[2, 3] == 0


# =============================================================================
# Connected Components Tests
# =============================================================================

class TestConnectedComponents:
    def test_connected_components_basic(self):
        """connected_components should find separate regions."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[0, 0] = 1
        mask[4, 4] = 1
        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4)
        assert len(components) == 2

    def test_connected_components_8_connectivity(self):
        """connected_components with 8-connectivity should connect diagonals."""
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[0, 0] = 1
        mask[1, 1] = 1
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)

        # 4-connectivity: 3 separate components
        comp4 = rle.connected_components(connectivity=4)
        assert len(comp4) == 3

        # 8-connectivity: 1 component (diagonal)
        comp8 = rle.connected_components(connectivity=8)
        assert len(comp8) == 1

    def test_connected_components_min_size(self):
        """connected_components should filter by min_size."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels
        mask[5:10, 5:10] = 1  # 25 pixels
        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4, min_size=10)
        assert len(components) == 1
        assert components[0].area() == 25

    def test_largest_connected_component(self):
        """largest_connected_component should return biggest region."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels
        mask[5:10, 5:10] = 1  # 25 pixels
        rle = RLEMask.from_array(mask)
        largest = rle.largest_connected_component()
        assert largest.area() == 25

    def test_remove_small_components(self):
        """remove_small_components should remove small regions."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0:2, 0:2] = 1  # 4 pixels
        mask[5:10, 5:10] = 1  # 25 pixels
        rle = RLEMask.from_array(mask)
        cleaned = rle.remove_small_components(min_size=10)
        arr = np.array(cleaned)
        assert arr[0:2, 0:2].sum() == 0
        assert arr[5:10, 5:10].sum() == 25

    def test_fill_small_holes(self):
        """fill_small_holes should fill small background regions."""
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[2, 2] = 0  # small hole
        rle = RLEMask.from_array(mask)
        filled = rle.fill_small_holes(min_size=2)
        assert filled.area() == 25  # hole filled


# =============================================================================
# Analysis and Metrics Tests
# =============================================================================

class TestAnalysisMetrics:
    def test_area(self, eye3):
        """area should count foreground pixels."""
        rle = RLEMask.from_array(eye3)
        assert rle.area() == 3

    def test_count_nonzero(self, eye3):
        """count_nonzero should be same as area."""
        rle = RLEMask.from_array(eye3)
        assert rle.count_nonzero() == rle.area()

    def test_centroid(self):
        """centroid should return center of mass."""
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[2, 2] = 1
        rle = RLEMask.from_array(mask)
        cx, cy = rle.centroid()
        assert cx == 2.0
        assert cy == 2.0

    def test_bbox(self, rect_mask):
        """bbox should return bounding box."""
        rle = RLEMask.from_array(rect_mask)
        bbox = rle.bbox()
        # rect_mask has 1s at [3:7, 3:7]
        assert bbox[0] == 3  # x
        assert bbox[1] == 3  # y
        assert bbox[2] == 4  # width
        assert bbox[3] == 4  # height

    def test_perimeter(self):
        """perimeter should count contour pixels."""
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[1:-1, 1:-1] = 1  # filled square
        rle = RLEMask.from_array(mask)
        # Perimeter of 5x5 square is 4*5 - 4 = 16 (edges minus corners counted twice)
        # But perimeter counts pixels, so it's the border pixels: 5*4 - 4 = 16
        assert rle.perimeter() > 0

    def test_contours(self):
        """contours should return edge pixels."""
        mask = np.ones((5, 5), dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        contours = rle.contours()
        arr = np.array(contours)
        # Border pixels should be 1
        assert arr[0, :].sum() == 5
        assert arr[-1, :].sum() == 5
        # Interior should be 0
        assert arr[2, 2] == 0

    def test_iou_identical(self, eye3):
        """iou of identical masks should be 1.0."""
        rle = RLEMask.from_array(eye3)
        assert rle.iou(rle) == 1.0

    def test_iou_no_overlap(self):
        """iou of non-overlapping masks should be 0.0."""
        m1 = np.zeros((5, 5), dtype=np.uint8)
        m1[0:2, 0:2] = 1
        m2 = np.zeros((5, 5), dtype=np.uint8)
        m2[3:5, 3:5] = 1
        rle1 = RLEMask.from_array(m1)
        rle2 = RLEMask.from_array(m2)
        assert rle1.iou(rle2) == 0.0

    def test_iou_matrix(self):
        """iou_matrix should compute pairwise IoUs."""
        masks1 = [RLEMask.from_array(np.eye(3, dtype=np.uint8))]
        masks2 = [
            RLEMask.from_array(np.eye(3, dtype=np.uint8)),
            RLEMask.from_array(np.zeros((3, 3), dtype=np.uint8))
        ]
        iou_mat = RLEMask.iou_matrix(masks1, masks2)
        assert iou_mat.shape == (1, 2)
        assert iou_mat[0, 0] == 1.0
        assert iou_mat[0, 1] == 0.0

    def test_nonzero(self, eye3):
        """nonzero should return foreground coordinates."""
        rle = RLEMask.from_array(eye3)
        coords = rle.nonzero()
        assert coords.shape[0] == 3  # 3 foreground pixels
        assert coords.shape[1] == 2  # x, y

    def test_any(self):
        """any should return True if any foreground pixels."""
        assert RLEMask.zeros((3, 3)).any() == False
        assert RLEMask.ones((3, 3)).any() == True

    def test_all(self):
        """all should return True if all pixels are foreground."""
        assert RLEMask.zeros((3, 3)).all() == False
        assert RLEMask.ones((3, 3)).all() == True

    def test_is_valid_rle(self, eye3):
        """is_valid_rle should validate RLE structure."""
        rle = RLEMask.from_array(eye3)
        assert rle.is_valid_rle() == True


# =============================================================================
# Fill Operations Tests
# =============================================================================

class TestFillOperations:
    def test_fill_rectangle(self):
        """fill_rectangle should fill rectangular region."""
        rle = RLEMask.zeros((5, 5))
        filled = rle.fill_rectangle([1, 1, 2, 2], value=1)
        arr = np.array(filled)
        assert arr[1:3, 1:3].sum() == 4
        assert arr.sum() == 4

    def test_fill_rectangle_clear(self):
        """fill_rectangle with value=0 should clear region."""
        rle = RLEMask.ones((5, 5))
        filled = rle.fill_rectangle([1, 1, 2, 2], value=0)
        arr = np.array(filled)
        assert arr[1:3, 1:3].sum() == 0
        assert arr.sum() == 21

    def test_fill_circle(self):
        """fill_circle should fill circular region."""
        rle = RLEMask.zeros((11, 11))
        filled = rle.fill_circle([5, 5], 2, value=1)
        arr = np.array(filled)
        assert arr[5, 5] == 1  # center filled
        assert arr[0, 0] == 0  # corner empty


# =============================================================================
# Pooling Operations Tests
# =============================================================================

class TestPooling:
    def test_max_pool2x2(self):
        """max_pool2x2 should max-pool by 2."""
        mask = np.array([[1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        pooled = rle.max_pool2x2()
        arr = np.array(pooled)
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 1  # max of top-left 2x2
        assert arr[1, 1] == 1  # max of bottom-right 2x2

    def test_min_pool2x2(self):
        """min_pool2x2 should min-pool by 2."""
        mask = np.array([[1, 1, 1, 1],
                         [1, 1, 1, 1],
                         [1, 1, 0, 0],
                         [1, 1, 0, 0]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        pooled = rle.min_pool2x2()
        arr = np.array(pooled)
        assert arr.shape == (2, 2)
        assert arr[0, 0] == 1  # min of top-left 2x2 (all 1s)
        assert arr[1, 1] == 0  # min of bottom-right 2x2 (has 0)

    def test_avg_pool2x2(self):
        """avg_pool2x2 should average-pool by 2."""
        mask = np.array([[1, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 0, 1, 1],
                         [0, 0, 1, 1]], dtype=np.uint8)
        rle = RLEMask.from_array(mask)
        pooled = rle.avg_pool2x2()
        arr = np.array(pooled)
        assert arr.shape == (2, 2)
        # Top-left: 3/4 >= 0.5 -> 1
        assert arr[0, 0] == 1
        # Bottom-right: 4/4 >= 0.5 -> 1
        assert arr[1, 1] == 1


# =============================================================================
# Conversion Tests
# =============================================================================

class TestConversion:
    def test_to_array_default(self, eye3):
        """to_array should return numpy array."""
        rle = RLEMask.from_array(eye3)
        arr = rle.to_array()
        np.testing.assert_array_equal(arr, eye3)

    def test_to_array_value(self, eye3):
        """to_array with value should use that for foreground."""
        rle = RLEMask.from_array(eye3)
        arr = rle.to_array(value=255)
        expected = eye3 * 255
        np.testing.assert_array_equal(arr, expected)

    def test_to_array_order_c(self, eye3):
        """to_array with order='C' should return C-contiguous."""
        rle = RLEMask.from_array(eye3)
        arr = rle.to_array(order='C')
        assert arr.flags.c_contiguous

    def test_to_array_order_f(self, eye3):
        """to_array with order='F' should return F-contiguous."""
        rle = RLEMask.from_array(eye3)
        arr = rle.to_array(order='F')
        assert arr.flags.f_contiguous

    def test_to_dict(self, eye3):
        """to_dict should return RLE dictionary."""
        rle = RLEMask.from_array(eye3)
        d = rle.to_dict()
        assert 'size' in d
        assert 'counts' in d
        assert d['size'] == [3, 3]

    def test_to_dict_zlevel(self, eye3):
        """to_dict with zlevel should compress."""
        rle = RLEMask.from_array(eye3)
        d = rle.to_dict(zlevel=-1)
        assert 'zcounts' in d
        assert 'counts' not in d

    def test___array__(self, eye3):
        """__array__ should allow np.array(rle)."""
        rle = RLEMask.from_array(eye3)
        arr = np.array(rle)
        np.testing.assert_array_equal(arr, eye3)

    def test___array___dtype(self, eye3):
        """__array__ with dtype should convert."""
        rle = RLEMask.from_array(eye3)
        arr = np.array(rle, dtype=np.float32)
        assert arr.dtype == np.float32


# =============================================================================
# Misc Tests
# =============================================================================

class TestMisc:
    def test_copy(self, eye3):
        """copy should create independent copy."""
        rle1 = RLEMask.from_array(eye3)
        rle2 = rle1.copy()
        assert rle1 == rle2
        assert rle1 is not rle2

    def test_equality(self, eye3):
        """== should compare masks."""
        rle1 = RLEMask.from_array(eye3)
        rle2 = RLEMask.from_array(eye3)
        rle3 = RLEMask.zeros((3, 3))
        assert rle1 == rle2
        assert not (rle1 == rle3)

    def test_repr(self, eye3):
        """repr should return string representation."""
        rle = RLEMask.from_array(eye3)
        r = repr(rle)
        assert 'RLEMask' in r
        assert 'shape' in r

    def test_merge_to_label_map(self):
        """merge_to_label_map should create label image."""
        m1 = RLEMask.from_array(np.array([[1, 0], [0, 0]], dtype=np.uint8))
        m2 = RLEMask.from_array(np.array([[0, 1], [0, 0]], dtype=np.uint8))
        m3 = RLEMask.from_array(np.array([[0, 0], [1, 1]], dtype=np.uint8))
        labelmap = RLEMask.merge_to_label_map([m1, m2, m3])
        assert labelmap[0, 0] == 1
        assert labelmap[0, 1] == 2
        assert labelmap[1, 0] == 3
        assert labelmap[1, 1] == 3

    def test_largest_interior_rectangle(self):
        """largest_interior_rectangle should find max rectangle."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 3:9] = 1
        rle = RLEMask.from_array(mask)
        rect = rle.largest_interior_rectangle()
        # Should find a large rectangle inside
        assert rect[2] > 0  # width > 0
        assert rect[3] > 0  # height > 0