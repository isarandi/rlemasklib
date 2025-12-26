"""Hypothesis-based property tests for RLE operations.

Uses hypothesis to find edge cases that might cause memory errors or incorrect results.
"""

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck
from rlemasklib.oop import RLEMask

# Suppress health checks when running under valgrind (very slow)
slow_settings = settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    deadline=None
)

# Strategies for generating test data
@st.composite
def mask_array(draw, max_h=30, max_w=30, min_h=0, min_w=0):
    """Generate a random binary mask array."""
    h = draw(st.integers(min_value=min_h, max_value=max_h))
    w = draw(st.integers(min_value=min_w, max_value=max_w))
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.uint8)
    data = draw(st.lists(st.integers(0, 1), min_size=h*w, max_size=h*w))
    return np.array(data, dtype=np.uint8).reshape(h, w)


@st.composite
def mask_pair(draw, max_h=30, max_w=30, min_h=0, min_w=0):
    """Generate two masks of the same shape."""
    h = draw(st.integers(min_value=min_h, max_value=max_h))
    w = draw(st.integers(min_value=min_w, max_value=max_w))
    if h == 0 or w == 0:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)
    data1 = draw(st.lists(st.integers(0, 1), min_size=h*w, max_size=h*w))
    data2 = draw(st.lists(st.integers(0, 1), min_size=h*w, max_size=h*w))
    return (np.array(data1, dtype=np.uint8).reshape(h, w),
            np.array(data2, dtype=np.uint8).reshape(h, w))


class TestEncodeDecodeHypothesis:
    @given(mask=mask_array())
    @slow_settings
    def test_roundtrip(self, mask):
        """Encoding then decoding should return the original mask."""
        rle = RLEMask.from_array(mask)
        result = np.array(rle)
        np.testing.assert_array_equal(result, mask)

    @given(mask=mask_array())
    @slow_settings
    def test_dict_roundtrip(self, mask):
        """Converting to dict and back should preserve the mask."""
        rle = RLEMask.from_array(mask)
        d = rle.to_dict()
        rle2 = RLEMask.from_dict(d)
        result = np.array(rle2)
        np.testing.assert_array_equal(result, mask)


class TestTransposeHypothesis:
    @given(mask=mask_array())
    @slow_settings
    def test_transpose(self, mask):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.transpose())
        np.testing.assert_array_equal(result, mask.T)

    @given(mask=mask_array())
    @slow_settings
    def test_double_transpose(self, mask):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.transpose().transpose())
        np.testing.assert_array_equal(result, mask)


class TestRotationHypothesis:
    @given(mask=mask_array(), k=st.integers(-4, 4))
    @slow_settings
    def test_rot90(self, mask, k):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.rot90(k=k))
        expected = np.rot90(mask, k=k)
        np.testing.assert_array_equal(result, expected)

    @given(mask=mask_array())
    @slow_settings
    def test_rot90_full_circle(self, mask):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.rot90(4))
        np.testing.assert_array_equal(result, mask)


class TestFlipHypothesis:
    @given(mask=mask_array())
    @slow_settings
    def test_flipud(self, mask):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.flipud())
        np.testing.assert_array_equal(result, np.flipud(mask))

    @given(mask=mask_array())
    @slow_settings
    def test_fliplr(self, mask):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.fliplr())
        np.testing.assert_array_equal(result, np.fliplr(mask))

    @given(mask=mask_array())
    @slow_settings
    def test_double_flip(self, mask):
        rle = RLEMask.from_array(mask)
        np.testing.assert_array_equal(np.array(rle.flipud().flipud()), mask)
        np.testing.assert_array_equal(np.array(rle.fliplr().fliplr()), mask)


class TestComplementHypothesis:
    @given(mask=mask_array())
    @slow_settings
    def test_complement(self, mask):
        rle = RLEMask.from_array(mask)
        result = np.array(~rle)
        np.testing.assert_array_equal(result, 1 - mask)

    @given(mask=mask_array())
    @slow_settings
    def test_double_complement(self, mask):
        rle = RLEMask.from_array(mask)
        result = np.array(~~rle)
        np.testing.assert_array_equal(result, mask)


class TestBooleanOpsHypothesis:
    @given(masks=mask_pair())
    @slow_settings
    def test_union(self, masks):
        m1, m2 = masks
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 | rle2)
        np.testing.assert_array_equal(result, m1 | m2)

    @given(masks=mask_pair())
    @slow_settings
    def test_intersection(self, masks):
        m1, m2 = masks
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 & rle2)
        np.testing.assert_array_equal(result, m1 & m2)

    @given(masks=mask_pair())
    @slow_settings
    def test_xor(self, masks):
        m1, m2 = masks
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 ^ rle2)
        np.testing.assert_array_equal(result, m1 ^ m2)

    @given(masks=mask_pair())
    @slow_settings
    def test_difference(self, masks):
        m1, m2 = masks
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 - rle2)
        np.testing.assert_array_equal(result, m1 & ~m2)


class TestCropHypothesis:
    @given(mask=mask_array(min_h=1, min_w=1), data=st.data())
    @slow_settings
    def test_crop_within_bounds(self, mask, data):
        h, w = mask.shape
        x = data.draw(st.integers(0, w - 1))
        y = data.draw(st.integers(0, h - 1))
        cw = data.draw(st.integers(1, w - x))
        ch = data.draw(st.integers(1, h - y))

        rle = RLEMask.from_array(mask)
        result = np.array(rle.crop([x, y, cw, ch]))
        expected = mask[y:y+ch, x:x+cw]
        np.testing.assert_array_equal(result, expected)

    @given(mask=mask_array(), data=st.data())
    @slow_settings
    def test_crop_outside_bounds(self, mask, data):
        """Crop extending outside should still work without crashing."""
        h, w = mask.shape
        x = data.draw(st.integers(-10, w + 10))
        y = data.draw(st.integers(-10, h + 10))
        cw = data.draw(st.integers(0, 20))
        ch = data.draw(st.integers(0, 20))

        rle = RLEMask.from_array(mask)
        result = rle.crop([x, y, cw, ch])
        # Just verify it doesn't crash and returns a valid shape
        assert len(result.shape) == 2


class TestPadHypothesis:
    @given(mask=mask_array(),
           top=st.integers(0, 10), bottom=st.integers(0, 10),
           left=st.integers(0, 10), right=st.integers(0, 10))
    @slow_settings
    def test_pad_zeros(self, mask, top, bottom, left, right):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.pad(top, bottom, left, right, value=0))
        expected = np.pad(mask, ((top, bottom), (left, right)), constant_values=0)
        np.testing.assert_array_equal(result, expected)

    @given(mask=mask_array(),
           top=st.integers(0, 10), bottom=st.integers(0, 10),
           left=st.integers(0, 10), right=st.integers(0, 10))
    @slow_settings
    def test_pad_ones(self, mask, top, bottom, left, right):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.pad(top, bottom, left, right, value=1))
        expected = np.pad(mask, ((top, bottom), (left, right)), constant_values=1)
        np.testing.assert_array_equal(result, expected)


class TestShiftHypothesis:
    @given(mask=mask_array(), data=st.data())
    @slow_settings
    def test_shift(self, mask, data):
        h, w = mask.shape
        if h == 0 or w == 0:
            return
        dy = data.draw(st.integers(-h, h))
        dx = data.draw(st.integers(-w, w))

        rle = RLEMask.from_array(mask)
        result = np.array(rle.shift((dy, dx)))

        expected = np.zeros_like(mask)
        src_y = slice(max(0, -dy), min(h, h - dy))
        src_x = slice(max(0, -dx), min(w, w - dx))
        dst_y = slice(max(0, dy), min(h, h + dy))
        dst_x = slice(max(0, dx), min(w, w + dx))
        expected[dst_y, dst_x] = mask[src_y, src_x]

        np.testing.assert_array_equal(result, expected)


class TestTileHypothesis:
    @given(mask=mask_array(max_h=10, max_w=10),
           nh=st.integers(1, 5), nw=st.integers(1, 5))
    @slow_settings
    def test_tile(self, mask, nh, nw):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.tile(nh, nw))
        expected = np.tile(mask, (nh, nw))
        np.testing.assert_array_equal(result, expected)


class TestRepeatHypothesis:
    @given(mask=mask_array(max_h=10, max_w=10),
           rh=st.integers(1, 5), rw=st.integers(1, 5))
    @slow_settings
    def test_repeat(self, mask, rh, rw):
        rle = RLEMask.from_array(mask)
        result = np.array(rle.repeat(rh, rw))
        expected = np.repeat(np.repeat(mask, rh, axis=0), rw, axis=1)
        np.testing.assert_array_equal(result, expected)


class TestAreaHypothesis:
    @given(mask=mask_array())
    @slow_settings
    def test_area(self, mask):
        rle = RLEMask.from_array(mask)
        assert rle.area() == np.sum(mask)


class TestBboxHypothesis:
    @given(mask=mask_array())
    @slow_settings
    def test_bbox(self, mask):
        assume(mask.sum() > 0)

        rle = RLEMask.from_array(mask)
        x, y, w, h = rle.bbox()

        ys, xs = np.where(mask)
        assert x == xs.min()
        assert y == ys.min()
        assert x + w == xs.max() + 1
        assert y + h == ys.max() + 1


class TestResizeHypothesis:
    @given(mask=mask_array(min_h=1, min_w=1, max_h=20, max_w=20),
           new_h=st.integers(1, 30), new_w=st.integers(1, 30))
    @slow_settings
    def test_resize_shape(self, mask, new_h, new_w):
        rle = RLEMask.from_array(mask)
        result = rle.resize((new_h, new_w))
        assert result.shape == (new_h, new_w)


class TestMorphologyHypothesis:
    @given(mask=mask_array(min_h=1, min_w=1))
    @slow_settings
    def test_dilate3x3_increases_area(self, mask):
        rle = RLEMask.from_array(mask)
        dilated = rle.dilate3x3()
        assert dilated.area() >= rle.area()

    @given(mask=mask_array(min_h=1, min_w=1))
    @slow_settings
    def test_erode3x3_decreases_area(self, mask):
        rle = RLEMask.from_array(mask)
        eroded = rle.erode3x3()
        assert eroded.area() <= rle.area()

    @given(mask=mask_array(min_h=1, min_w=1))
    @slow_settings
    def test_dilate_erode_idempotent_on_full(self, mask):
        """Dilating a full mask should keep it full."""
        full = RLEMask.ones(mask.shape)
        dilated = full.dilate3x3()
        np.testing.assert_array_equal(np.array(dilated), np.array(full))


class TestConnectedComponentsHypothesis:
    @given(mask=mask_array(min_h=1, min_w=1))
    @slow_settings
    def test_components_union_equals_original(self, mask):
        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4)

        if len(components) == 0:
            assert mask.sum() == 0
        else:
            union = components[0]
            for comp in components[1:]:
                union = union | comp
            np.testing.assert_array_equal(np.array(union), mask)

    @given(mask=mask_array(min_h=1, min_w=1))
    @slow_settings
    def test_components_no_overlap(self, mask):
        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4)

        if len(components) > 1:
            for i, c1 in enumerate(components):
                for c2 in components[i+1:]:
                    assert (c1 & c2).area() == 0


class TestIoUHypothesis:
    @given(masks=mask_pair(min_h=1, min_w=1))
    @slow_settings
    def test_iou_range(self, masks):
        m1, m2 = masks
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        iou = rle1.iou(rle2)
        assert 0 <= iou <= 1

    @given(mask=mask_array(min_h=1, min_w=1))
    @slow_settings
    def test_iou_self_is_one(self, mask):
        assume(mask.sum() > 0)
        rle = RLEMask.from_array(mask)
        assert rle.iou(rle) == pytest.approx(1.0)


class TestLargestInteriorRectangleHypothesis:
    @given(mask=mask_array(min_h=1, min_w=1))
    @slow_settings
    def test_largest_interior_rectangle_inside_mask(self, mask):
        assume(mask.sum() > 0)
        rle = RLEMask.from_array(mask)
        x, y, w, h = rle.largest_interior_rectangle()

        if w > 0 and h > 0:
            # The rectangle should be inside the mask
            rect_area = mask[y:y+h, x:x+w]
            assert rect_area.sum() == w * h  # All ones inside rectangle
