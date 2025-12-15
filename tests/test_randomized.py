"""Randomized tests verifying RLE operations against numpy/cv2 ground truth."""

import numpy as np
import pytest
from rlemasklib.oop import RLEMask

# Try to import cv2 for morphology ground truth
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def random_mask(max_h=20, max_w=20, min_h=0, min_w=0):
    """Generate a random binary mask."""
    h = np.random.randint(min_h, max_h + 1)
    w = np.random.randint(min_w, max_w + 1)
    return np.random.randint(0, 2, (h, w), dtype=np.uint8)


def random_mask_pair(max_h=20, max_w=20, min_h=0, min_w=0):
    """Generate two random binary masks of the same shape."""
    h = np.random.randint(min_h, max_h + 1)
    w = np.random.randint(min_w, max_w + 1)
    m1 = np.random.randint(0, 2, (h, w), dtype=np.uint8)
    m2 = np.random.randint(0, 2, (h, w), dtype=np.uint8)
    return m1, m2


class TestTransposeRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_transpose(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.transpose())
        expected = mask.T
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_double_transpose(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.transpose().transpose())
        np.testing.assert_array_equal(result, mask)


class TestRotationRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_rot90_k1(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.rot90(k=1))
        expected = np.rot90(mask, k=1)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_rot90_k2(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.rot90(k=2))
        expected = np.rot90(mask, k=2)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_rot90_k3(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.rot90(k=3))
        expected = np.rot90(mask, k=3)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_rot90_negative(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        for k in [-1, -2, -3]:
            result = np.array(rle.rot90(k=k))
            expected = np.rot90(mask, k=k)
            np.testing.assert_array_equal(result, expected)


class TestFlipRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_flipud(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.flipud())
        expected = np.flipud(mask)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_fliplr(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.fliplr())
        expected = np.fliplr(mask)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_double_flip(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle.flipud().flipud())
        np.testing.assert_array_equal(result, mask)
        result = np.array(rle.fliplr().fliplr())
        np.testing.assert_array_equal(result, mask)


class TestComplementRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_complement(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(~rle)
        expected = 1 - mask
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_double_complement(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(~~rle)
        np.testing.assert_array_equal(result, mask)


class TestBooleanOpsRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_union(self, seed):
        np.random.seed(seed)
        m1, m2 = random_mask_pair()
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 | rle2)
        expected = m1 | m2
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_intersection(self, seed):
        np.random.seed(seed)
        m1, m2 = random_mask_pair()
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 & rle2)
        expected = m1 & m2
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_difference(self, seed):
        np.random.seed(seed)
        m1, m2 = random_mask_pair()
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 - rle2)
        expected = m1 & ~m2
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_xor(self, seed):
        np.random.seed(seed)
        m1, m2 = random_mask_pair()
        rle1, rle2 = RLEMask.from_array(m1), RLEMask.from_array(m2)
        result = np.array(rle1 ^ rle2)
        expected = m1 ^ m2
        np.testing.assert_array_equal(result, expected)


class TestEncodeDecodeRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_roundtrip(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        result = np.array(rle)
        np.testing.assert_array_equal(result, mask)

    @pytest.mark.parametrize("seed", range(500))
    def test_roundtrip_with_compression(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        d = rle.to_dict(zlevel=-1)
        rle2 = RLEMask.from_dict(d)
        result = np.array(rle2)
        np.testing.assert_array_equal(result, mask)


class TestCropRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_crop(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=30, max_w=30)
        h, w = mask.shape

        if h == 0 or w == 0:
            # Test that cropping 0-size mask works
            rle = RLEMask.from_array(mask)
            result = np.array(rle.crop([0, 0, 0, 0]))
            assert result.shape == (0, 0)
            return

        # Random crop box within bounds
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)
        cw = np.random.randint(1, w - x + 1)
        ch = np.random.randint(1, h - y + 1)

        rle = RLEMask.from_array(mask)
        result = np.array(rle.crop([x, y, cw, ch]))
        expected = mask[y:y+ch, x:x+cw]
        np.testing.assert_array_equal(result, expected)


class TestPadRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_pad_zeros(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        left = np.random.randint(0, 5)
        right = np.random.randint(0, 5)
        top = np.random.randint(0, 5)
        bottom = np.random.randint(0, 5)

        rle = RLEMask.from_array(mask)
        # pad() signature: (top, bottom, left, right, ...)
        result = np.array(rle.pad(top, bottom, left, right, value=0))
        expected = np.pad(mask, ((top, bottom), (left, right)), constant_values=0)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_pad_ones(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        left = np.random.randint(0, 5)
        right = np.random.randint(0, 5)
        top = np.random.randint(0, 5)
        bottom = np.random.randint(0, 5)

        rle = RLEMask.from_array(mask)
        # pad() signature: (top, bottom, left, right, ...)
        result = np.array(rle.pad(top, bottom, left, right, value=1))
        expected = np.pad(mask, ((top, bottom), (left, right)), constant_values=1)
        np.testing.assert_array_equal(result, expected)


class TestShiftRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_shift(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)
        h, w = mask.shape
        dy = np.random.randint(-h//2, h//2 + 1)
        dx = np.random.randint(-w//2, w//2 + 1)

        rle = RLEMask.from_array(mask)
        result = np.array(rle.shift((dy, dx)))

        # Build expected with numpy
        expected = np.zeros_like(mask)
        src_y = slice(max(0, -dy), min(h, h - dy))
        src_x = slice(max(0, -dx), min(w, w - dx))
        dst_y = slice(max(0, dy), min(h, h + dy))
        dst_x = slice(max(0, dx), min(w, w + dx))
        expected[dst_y, dst_x] = mask[src_y, src_x]

        np.testing.assert_array_equal(result, expected)


class TestTileRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_tile(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=10, max_w=10)

        nh = np.random.randint(1, 4)
        nw = np.random.randint(1, 4)

        rle = RLEMask.from_array(mask)
        result = np.array(rle.tile(nh, nw))
        expected = np.tile(mask, (nh, nw))
        np.testing.assert_array_equal(result, expected)


class TestRepeatRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_repeat(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=10, max_w=10)
        rh = np.random.randint(1, 4)
        rw = np.random.randint(1, 4)

        rle = RLEMask.from_array(mask)
        result = np.array(rle.repeat(rh, rw))
        expected = np.repeat(np.repeat(mask, rh, axis=0), rw, axis=1)
        np.testing.assert_array_equal(result, expected)


class TestHconcatRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_hconcat(self, seed):
        np.random.seed(seed)
        h = np.random.randint(0, 10)
        n_masks = np.random.randint(2, 5)
        masks = [random_mask(min_h=h, max_h=h, max_w=10) for _ in range(n_masks)]

        rles = [RLEMask.from_array(m) for m in masks]
        result = np.array(RLEMask.hconcat(rles))
        expected = np.hstack(masks)
        np.testing.assert_array_equal(result, expected)


class TestVconcatRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_vconcat(self, seed):
        np.random.seed(seed)
        w = np.random.randint(0, 10)
        n_masks = np.random.randint(2, 5)
        masks = [random_mask(max_h=10, min_w=w, max_w=w) for _ in range(n_masks)]

        rles = [RLEMask.from_array(m) for m in masks]
        result = np.array(RLEMask.vconcat(rles))
        expected = np.vstack(masks)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not HAS_CV2, reason="cv2 not available")
class TestMorphologyRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_dilate_3x3_connectivity4(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)
        h, w = mask.shape

        # cv2 morphology fails on empty masks
        if h == 0 or w == 0:
            rle = RLEMask.from_array(mask)
            result = np.array(rle.dilate3x3(connectivity=4))
            assert result.shape == mask.shape
            return

        rle = RLEMask.from_array(mask)
        result = np.array(rle.dilate3x3(connectivity=4))

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        expected = cv2.dilate(mask, kernel, iterations=1)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_dilate_3x3_connectivity8(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)
        h, w = mask.shape

        # cv2 morphology fails on empty masks
        if h == 0 or w == 0:
            rle = RLEMask.from_array(mask)
            result = np.array(rle.dilate3x3(connectivity=8))
            assert result.shape == mask.shape
            return

        rle = RLEMask.from_array(mask)
        result = np.array(rle.dilate3x3(connectivity=8))

        kernel = np.ones((3, 3), dtype=np.uint8)
        expected = cv2.dilate(mask, kernel, iterations=1)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_erode_3x3_connectivity4(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)
        h, w = mask.shape

        # cv2 morphology fails on empty masks
        if h == 0 or w == 0:
            rle = RLEMask.from_array(mask)
            result = np.array(rle.erode3x3(connectivity=4))
            assert result.shape == mask.shape
            return

        rle = RLEMask.from_array(mask)
        result = np.array(rle.erode3x3(connectivity=4))

        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        expected = cv2.erode(mask, kernel, iterations=1)
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("seed", range(500))
    def test_erode_3x3_connectivity8(self, seed):
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)
        h, w = mask.shape

        # cv2 morphology fails on empty masks
        if h == 0 or w == 0:
            rle = RLEMask.from_array(mask)
            result = np.array(rle.erode3x3(connectivity=8))
            assert result.shape == mask.shape
            return

        rle = RLEMask.from_array(mask)
        result = np.array(rle.erode3x3(connectivity=8))

        kernel = np.ones((3, 3), dtype=np.uint8)
        expected = cv2.erode(mask, kernel, iterations=1)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.skipif(not HAS_CV2, reason="cv2 not available")
class TestConnectedComponentsRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_connected_components_count_4(self, seed):
        """Verify number of components matches cv2."""
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)
        h, w = mask.shape

        # cv2.connectedComponents crashes on 0-size masks
        if h == 0 or w == 0:
            rle = RLEMask.from_array(mask)
            components = rle.connected_components(connectivity=4)
            assert len(components) == 0
            return

        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4)

        num_labels, _ = cv2.connectedComponents(mask, connectivity=4)
        expected_count = num_labels - 1  # cv2 includes background as label 0
        assert len(components) == expected_count

    @pytest.mark.parametrize("seed", range(500))
    def test_connected_components_count_8(self, seed):
        """Verify number of components matches cv2."""
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)
        h, w = mask.shape

        # cv2.connectedComponents crashes on 0-size masks
        if h == 0 or w == 0:
            rle = RLEMask.from_array(mask)
            components = rle.connected_components(connectivity=8)
            assert len(components) == 0
            return

        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=8)

        num_labels, _ = cv2.connectedComponents(mask, connectivity=8)
        expected_count = num_labels - 1  # cv2 includes background as label 0
        assert len(components) == expected_count

    @pytest.mark.parametrize("seed", range(500))
    def test_connected_components_union_equals_original(self, seed):
        """Verify union of all components equals original mask."""
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)

        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4)

        if len(components) == 0:
            assert mask.sum() == 0
        else:
            union = components[0]
            for comp in components[1:]:
                union = union | comp
            np.testing.assert_array_equal(np.array(union), mask)

    @pytest.mark.parametrize("seed", range(500))
    def test_connected_components_no_overlap(self, seed):
        """Verify components don't overlap."""
        np.random.seed(seed)
        mask = random_mask(max_h=20, max_w=20)

        rle = RLEMask.from_array(mask)
        components = rle.connected_components(connectivity=4)

        if len(components) > 1:
            for i, c1 in enumerate(components):
                for c2 in components[i+1:]:
                    intersection = c1 & c2
                    assert intersection.area() == 0


class TestAreaRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_area(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        rle = RLEMask.from_array(mask)
        assert rle.area() == np.sum(mask)


class TestBboxRandomized:
    @pytest.mark.parametrize("seed", range(500))
    def test_bbox(self, seed):
        np.random.seed(seed)
        mask = random_mask()
        if mask.sum() == 0:
            return  # Skip empty masks

        rle = RLEMask.from_array(mask)
        x, y, w, h = rle.bbox()

        # Verify bbox contains all foreground pixels
        ys, xs = np.where(mask)
        assert x <= xs.min()
        assert y <= ys.min()
        assert x + w > xs.max()
        assert y + h > ys.max()

        # Verify bbox is tight
        assert x == xs.min()
        assert y == ys.min()
        assert x + w == xs.max() + 1
        assert y + h == ys.max() + 1
