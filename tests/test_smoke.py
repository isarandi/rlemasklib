"""Quick smoke test for wheel validation."""
import numpy as np
from rlemasklib import RLEMask


def test_basic_operations():
    # Create mask
    arr = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 0]], dtype=np.uint8)
    r = RLEMask.from_array(arr)

    # Properties
    assert r.shape == (3, 3)
    assert r.area() == 5

    # Roundtrip
    assert np.array_equal(np.array(r), arr)

    # Boolean ops
    r2 = RLEMask.from_array(np.ones((3, 3), dtype=np.uint8))
    assert (r | r2).area() == 9
    assert (r & r2).area() == 5
    assert (~r).area() == 4

    # Geometric ops
    assert r.transpose().shape == (3, 3)
    assert r.fliplr().shape == (3, 3)
    assert r.rot90().shape == (3, 3)

    # Concat
    h = RLEMask.hconcat([r, r])
    assert h.shape == (3, 6)
    v = RLEMask.vconcat([r, r])
    assert v.shape == (6, 3)

    # Zero-size edge cases
    empty = RLEMask.from_array(np.zeros((0, 5), dtype=np.uint8))
    assert empty.shape == (0, 5)
    assert RLEMask.hconcat([empty, empty]).shape == (0, 10)
