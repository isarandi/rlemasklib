import rlemasklib
import numpy as np


def test_union():
    d1 = rlemasklib.encode(np.eye(3))
    d2 = rlemasklib.encode(np.eye(3)[::-1])
    d3 = rlemasklib.union([d1, d2])
    assert np.all(rlemasklib.decode(d3) == np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))


def test_intersection():
    d1 = rlemasklib.encode(np.eye(3))
    d2 = rlemasklib.encode(np.eye(3)[::-1])
    d3 = rlemasklib.intersection([d1, d2])
    assert np.all(rlemasklib.decode(d3) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))


def test_difference():
    d1 = rlemasklib.encode(np.eye(3))
    d2 = rlemasklib.encode(np.eye(3)[::-1])
    d3 = rlemasklib.difference(d1, d2)
    assert np.all(rlemasklib.decode(d3) == np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]))


def test_decode_error():
    d1 = rlemasklib.encode(np.eye(3))
    d1['size'] = [2, 2]
    try:
        rlemasklib.decode(d1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")
