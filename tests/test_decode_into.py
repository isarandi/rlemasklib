"""Tests for decode_into with various array layouts and value types."""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

from rlemasklib import RLEMask


# --- Fixtures ---

@pytest.fixture
def simple_mask():
    """A simple rectangular mask for basic tests."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[20:50, 30:70] = 1
    return RLEMask(arr)


@pytest.fixture
def sparse_mask():
    """A sparse mask with scattered pixels."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[10, 10] = 1
    arr[50, 50] = 1
    arr[90, 90] = 1
    return RLEMask(arr)


@pytest.fixture
def empty_mask():
    """An empty mask (all zeros)."""
    return RLEMask(np.zeros((100, 100), dtype=np.uint8))


@pytest.fixture
def full_mask():
    """A full mask (all ones)."""
    return RLEMask(np.ones((100, 100), dtype=np.uint8))


# --- Helper functions ---

def reference_decode_into_2d(mask: RLEMask, arr: np.ndarray, value: int):
    """Reference implementation using numpy for verification."""
    bool_mask = mask.to_array().astype(bool)
    arr[bool_mask] = value


def reference_decode_into_3d_broadcast(mask: RLEMask, arr: np.ndarray, value: int):
    """Reference implementation for 3D broadcast."""
    bool_mask = mask.to_array().astype(bool)
    arr[bool_mask] = value


def reference_decode_into_3d_values(mask: RLEMask, arr: np.ndarray, values):
    """Reference implementation for 3D per-channel values."""
    bool_mask = mask.to_array().astype(bool)
    for c, v in enumerate(values):
        arr[:, :, c][bool_mask] = v


# --- Basic 2D tests ---

class TestDecodeInto2D:
    """Tests for 2D array decode_into."""

    def test_c_contiguous(self, simple_mask):
        """Test with C-contiguous array."""
        arr = np.zeros((100, 100), dtype=np.uint8, order='C')
        simple_mask.decode_into(arr, value=255)
        assert np.sum(arr == 255) == simple_mask.area()
        assert np.sum(arr == 0) == 100 * 100 - simple_mask.area()

    def test_f_contiguous(self, simple_mask):
        """Test with F-contiguous (Fortran order) array."""
        arr = np.zeros((100, 100), dtype=np.uint8, order='F')
        simple_mask.decode_into(arr, value=255)
        assert np.sum(arr == 255) == simple_mask.area()

    def test_strided_from_hwc(self, simple_mask):
        """Test with strided array from HWC image channel slice."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        channel = img[:, :, 0]  # This is strided, not contiguous
        assert not channel.flags.c_contiguous
        assert not channel.flags.f_contiguous

        simple_mask.decode_into(channel, value=128)
        assert np.sum(img[:, :, 0] == 128) == simple_mask.area()
        # Other channels should be unchanged
        assert np.sum(img[:, :, 1]) == 0
        assert np.sum(img[:, :, 2]) == 0

    def test_strided_every_other_row(self, simple_mask):
        """Test with strided array from every-other-row slice."""
        arr = np.zeros((200, 100), dtype=np.uint8)
        strided = arr[::2, :]  # Every other row, creates strided view
        assert not strided.flags.c_contiguous
        assert not strided.flags.f_contiguous

        simple_mask.decode_into(strided, value=200)
        assert np.sum(strided == 200) == simple_mask.area()

    def test_empty_mask(self, empty_mask):
        """Test with empty mask - array should remain unchanged."""
        arr = np.ones((100, 100), dtype=np.uint8) * 50
        empty_mask.decode_into(arr, value=255)
        assert np.all(arr == 50)

    def test_full_mask(self, full_mask):
        """Test with full mask - all pixels should be set."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        full_mask.decode_into(arr, value=255)
        assert np.all(arr == 255)

    def test_value_zero(self, simple_mask):
        """Test with value=0 - should write zeros to foreground pixels."""
        arr = np.ones((100, 100), dtype=np.uint8) * 100
        simple_mask.decode_into(arr, value=0)
        # Foreground pixels should be 0, background should remain 100
        bool_mask = simple_mask.to_array().astype(bool)
        assert np.all(arr[bool_mask] == 0)
        assert np.all(arr[~bool_mask] == 100)

    def test_nonzero_value_on_nonzero_array(self, simple_mask):
        """Test overwriting non-zero values."""
        arr = np.ones((100, 100), dtype=np.uint8) * 100
        simple_mask.decode_into(arr, value=200)
        bool_mask = simple_mask.to_array().astype(bool)
        assert np.all(arr[bool_mask] == 200)
        assert np.all(arr[~bool_mask] == 100)

    def test_overlay_multiple_masks(self, simple_mask, sparse_mask):
        """Test overlaying multiple masks."""
        arr = np.zeros((100, 100), dtype=np.uint8)
        simple_mask.decode_into(arr, value=100)
        sparse_mask.decode_into(arr, value=200)

        # sparse_mask pixels should override simple_mask where they overlap
        sparse_bool = sparse_mask.to_array().astype(bool)
        simple_bool = simple_mask.to_array().astype(bool)

        assert np.all(arr[sparse_bool] == 200)
        assert np.all(arr[simple_bool & ~sparse_bool] == 100)


# --- 3D tests ---

class TestDecodeInto3D:
    """Tests for 3D array decode_into."""

    def test_broadcast_c_contiguous(self, simple_mask):
        """Test broadcast scalar to C-contiguous HWC array."""
        img = np.zeros((100, 100, 3), dtype=np.uint8, order='C')
        simple_mask.decode_into(img, value=128)

        for c in range(3):
            assert np.sum(img[:, :, c] == 128) == simple_mask.area()

    def test_broadcast_f_contiguous(self, simple_mask):
        """Test broadcast scalar to F-contiguous HWC array."""
        img = np.zeros((100, 100, 3), dtype=np.uint8, order='F')
        simple_mask.decode_into(img, value=128)

        for c in range(3):
            assert np.sum(img[:, :, c] == 128) == simple_mask.area()

    def test_rgb_values_c_contiguous(self, simple_mask):
        """Test per-channel RGB values with C-contiguous array."""
        img = np.zeros((100, 100, 3), dtype=np.uint8, order='C')
        simple_mask.decode_into(img, value=(255, 128, 64))

        assert np.sum(img[:, :, 0] == 255) == simple_mask.area()
        assert np.sum(img[:, :, 1] == 128) == simple_mask.area()
        assert np.sum(img[:, :, 2] == 64) == simple_mask.area()

    def test_rgb_values_f_contiguous(self, simple_mask):
        """Test per-channel RGB values with F-contiguous array."""
        img = np.zeros((100, 100, 3), dtype=np.uint8, order='F')
        simple_mask.decode_into(img, value=(255, 128, 64))

        assert np.sum(img[:, :, 0] == 255) == simple_mask.area()
        assert np.sum(img[:, :, 1] == 128) == simple_mask.area()
        assert np.sum(img[:, :, 2] == 64) == simple_mask.area()

    def test_rgba_4_channels(self, simple_mask):
        """Test 4-channel RGBA array."""
        img = np.zeros((100, 100, 4), dtype=np.uint8)
        simple_mask.decode_into(img, value=(255, 128, 64, 200))

        assert np.sum(img[:, :, 0] == 255) == simple_mask.area()
        assert np.sum(img[:, :, 1] == 128) == simple_mask.area()
        assert np.sum(img[:, :, 2] == 64) == simple_mask.area()
        assert np.sum(img[:, :, 3] == 200) == simple_mask.area()

    def test_single_channel_3d(self, simple_mask):
        """Test single-channel 3D array (H, W, 1)."""
        img = np.zeros((100, 100, 1), dtype=np.uint8)
        simple_mask.decode_into(img, value=(128,))
        assert np.sum(img[:, :, 0] == 128) == simple_mask.area()

    def test_many_channels(self, simple_mask):
        """Test array with many channels."""
        img = np.zeros((100, 100, 8), dtype=np.uint8)
        values = tuple(range(10, 90, 10))
        simple_mask.decode_into(img, values)

        for c, v in enumerate(values):
            assert np.sum(img[:, :, c] == v) == simple_mask.area()

    def test_list_values(self, simple_mask):
        """Test with list instead of tuple for values."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        simple_mask.decode_into(img, value=[255, 128, 64])

        assert np.sum(img[:, :, 0] == 255) == simple_mask.area()

    def test_numpy_array_values(self, simple_mask):
        """Test with numpy array for values."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        simple_mask.decode_into(img, value=np.array([255, 128, 64]))

        assert np.sum(img[:, :, 0] == 255) == simple_mask.area()


# --- Error handling tests ---

class TestDecodeIntoErrors:
    """Tests for error conditions."""

    def test_shape_mismatch_2d(self, simple_mask):
        """Test error on shape mismatch for 2D array."""
        arr = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            simple_mask.decode_into(arr, value=255)

    def test_shape_mismatch_3d(self, simple_mask):
        """Test error on shape mismatch for 3D array."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            simple_mask.decode_into(img, value=255)

    def test_wrong_value_length(self, simple_mask):
        """Test error when value length doesn't match channels."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="length"):
            simple_mask.decode_into(img, value=(255, 128))  # Only 2 values for 3 channels

    def test_wrong_ndim(self, simple_mask):
        """Test error for wrong number of dimensions."""
        arr = np.zeros((100,), dtype=np.uint8)
        with pytest.raises(ValueError, match="2D or 3D"):
            simple_mask.decode_into(arr, value=255)

    def test_non_contiguous_3d_error(self, simple_mask):
        """Test error for non-contiguous 3D array."""
        img = np.zeros((100, 100, 6), dtype=np.uint8)
        strided = img[:, :, ::2]  # Every other channel, non-contiguous
        assert not strided.flags.c_contiguous
        assert not strided.flags.f_contiguous

        with pytest.raises(ValueError, match="contiguous"):
            simple_mask.decode_into(strided, value=255)


# --- Hypothesis property-based tests ---

@st.composite
def random_mask_and_shape(draw):
    """Generate a random mask and matching shape."""
    h = draw(st.integers(min_value=1, max_value=200))
    w = draw(st.integers(min_value=1, max_value=200))

    # Generate random mask
    density = draw(st.floats(min_value=0.0, max_value=1.0))
    arr = (np.random.random((h, w)) < density).astype(np.uint8)
    mask = RLEMask(arr)

    return mask, (h, w)


class TestDecodeIntoHypothesis:
    """Property-based tests using hypothesis."""

    @given(
        hw=st.tuples(st.integers(1, 100), st.integers(1, 100)),
        density=st.floats(0.0, 1.0),
        value=st.integers(0, 255),
        order=st.sampled_from(['C', 'F'])
    )
    @settings(max_examples=100)
    def test_2d_matches_reference(self, hw, density, value, order):
        """Test that decode_into matches reference implementation."""
        h, w = hw
        np.random.seed(42)  # For reproducibility within each test
        mask_arr = (np.random.random((h, w)) < density).astype(np.uint8)
        mask = RLEMask(mask_arr)

        # Test array
        arr = np.zeros((h, w), dtype=np.uint8, order=order)
        mask.decode_into(arr, value=value)

        # Reference
        ref = np.zeros((h, w), dtype=np.uint8, order=order)
        reference_decode_into_2d(mask, ref, value)

        np.testing.assert_array_equal(arr, ref)

    @given(
        hw=st.tuples(st.integers(1, 50), st.integers(1, 50)),
        density=st.floats(0.0, 1.0),
        value=st.integers(0, 255),
    )
    @settings(max_examples=50)
    def test_strided_matches_reference(self, hw, density, value):
        """Test strided decode matches reference."""
        h, w = hw
        np.random.seed(42)
        mask_arr = (np.random.random((h, w)) < density).astype(np.uint8)
        mask = RLEMask(mask_arr)

        # Create strided array from HWC image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        channel = img[:, :, 1]  # Middle channel, strided

        mask.decode_into(channel, value=value)

        # Reference
        ref = np.zeros((h, w), dtype=np.uint8)
        reference_decode_into_2d(mask, ref, value)

        np.testing.assert_array_equal(channel, ref)

    @given(
        hw=st.tuples(st.integers(1, 50), st.integers(1, 50)),
        density=st.floats(0.0, 1.0),
        value=st.integers(0, 255),
        order=st.sampled_from(['C', 'F'])
    )
    @settings(max_examples=50)
    def test_3d_broadcast_matches_reference(self, hw, density, value, order):
        """Test 3D broadcast matches reference."""
        h, w = hw
        np.random.seed(42)
        mask_arr = (np.random.random((h, w)) < density).astype(np.uint8)
        mask = RLEMask(mask_arr)

        img = np.zeros((h, w, 3), dtype=np.uint8, order=order)
        mask.decode_into(img, value=value)

        ref = np.zeros((h, w, 3), dtype=np.uint8, order=order)
        reference_decode_into_3d_broadcast(mask, ref, value)

        np.testing.assert_array_equal(img, ref)

    @given(
        hw=st.tuples(st.integers(1, 50), st.integers(1, 50)),
        density=st.floats(0.0, 1.0),
        values=st.tuples(st.integers(0, 255), st.integers(0, 255), st.integers(0, 255)),
        order=st.sampled_from(['C', 'F'])
    )
    @settings(max_examples=50)
    def test_3d_values_matches_reference(self, hw, density, values, order):
        """Test 3D per-channel values matches reference."""
        h, w = hw
        np.random.seed(42)
        mask_arr = (np.random.random((h, w)) < density).astype(np.uint8)
        mask = RLEMask(mask_arr)

        img = np.zeros((h, w, 3), dtype=np.uint8, order=order)
        mask.decode_into(img, value=values)

        ref = np.zeros((h, w, 3), dtype=np.uint8, order=order)
        reference_decode_into_3d_values(mask, ref, values)

        np.testing.assert_array_equal(img, ref)

    @given(
        hw=st.tuples(st.integers(1, 50), st.integers(1, 50)),
        density=st.floats(0.0, 1.0),
    )
    @settings(max_examples=30)
    def test_rgba_4channel(self, hw, density):
        """Test 4-channel RGBA."""
        h, w = hw
        np.random.seed(42)
        mask_arr = (np.random.random((h, w)) < density).astype(np.uint8)
        mask = RLEMask(mask_arr)

        values = (255, 128, 64, 200)
        img = np.zeros((h, w, 4), dtype=np.uint8)
        mask.decode_into(img, value=values)

        ref = np.zeros((h, w, 4), dtype=np.uint8)
        reference_decode_into_3d_values(mask, ref, values)

        np.testing.assert_array_equal(img, ref)


# --- Edge case tests ---

class TestDecodeIntoEdgeCases:
    """Edge case tests."""

    def test_1x1_mask(self):
        """Test with 1x1 mask."""
        mask = RLEMask(np.array([[1]], dtype=np.uint8))
        arr = np.zeros((1, 1), dtype=np.uint8)
        mask.decode_into(arr, value=255)
        assert arr[0, 0] == 255

    def test_1xn_mask(self):
        """Test with 1xN mask."""
        mask = RLEMask(np.array([[1, 0, 1, 1, 0]], dtype=np.uint8))
        arr = np.zeros((1, 5), dtype=np.uint8)
        mask.decode_into(arr, value=100)
        np.testing.assert_array_equal(arr, [[100, 0, 100, 100, 0]])

    def test_nx1_mask(self):
        """Test with Nx1 mask."""
        mask = RLEMask(np.array([[1], [0], [1]], dtype=np.uint8))
        arr = np.zeros((3, 1), dtype=np.uint8)
        mask.decode_into(arr, value=100)
        np.testing.assert_array_equal(arr, [[100], [0], [100]])

    def test_alternating_pixels(self):
        """Test with checkerboard-like alternating pattern."""
        h, w = 10, 10
        mask_arr = np.zeros((h, w), dtype=np.uint8)
        mask_arr[::2, ::2] = 1
        mask_arr[1::2, 1::2] = 1
        mask = RLEMask(mask_arr)

        arr = np.zeros((h, w), dtype=np.uint8)
        mask.decode_into(arr, value=255)

        np.testing.assert_array_equal(arr, mask_arr * 255)

    def test_single_pixel_corners(self):
        """Test mask with pixels only in corners."""
        mask_arr = np.zeros((100, 100), dtype=np.uint8)
        mask_arr[0, 0] = 1
        mask_arr[0, 99] = 1
        mask_arr[99, 0] = 1
        mask_arr[99, 99] = 1
        mask = RLEMask(mask_arr)

        arr = np.zeros((100, 100), dtype=np.uint8)
        mask.decode_into(arr, value=255)

        assert arr[0, 0] == 255
        assert arr[0, 99] == 255
        assert arr[99, 0] == 255
        assert arr[99, 99] == 255
        assert np.sum(arr == 255) == 4

    def test_preserves_existing_values(self):
        """Test that background values are preserved."""
        mask_arr = np.zeros((10, 10), dtype=np.uint8)
        mask_arr[2:8, 2:8] = 1
        mask = RLEMask(mask_arr)

        arr = np.full((10, 10), 50, dtype=np.uint8)
        mask.decode_into(arr, value=200)

        bool_mask = mask_arr.astype(bool)
        assert np.all(arr[bool_mask] == 200)
        assert np.all(arr[~bool_mask] == 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])