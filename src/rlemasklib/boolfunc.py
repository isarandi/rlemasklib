from enum import Enum

import numpy as np

_X = 0b1100
_Y = 0b1010


class BoolFunc(Enum):
    """Boolean functions for merging masks.

    Any Boolean function can be expressed.

    Examples:
        >>> d1 = encode(np.eye(3))
        >>> d2 = encode(np.eye(3)[::-1])
        >>> d3 = merge([d1, d2], ~BoolFunc.X & BoolFunc.Y)
        array([[0, 0, 1],
               [0, 0, 0],
               [1, 0, 0]], dtype=uint8)

    """

    A = _X
    B = _Y
    UNION = OR = _X | _Y
    INTERSECTION = AND = _X & _Y
    DIFFERENCE = _X & ~_Y
    SYMMETRIC_DIFFERENCE = XOR = _X ^ _Y
    EQUIVALENCE = XNOR = ~(_X ^ _Y)
    IMPLICATION = ~_X | _Y
    NOR = ~(_X | _Y)
    NAND = ~(_X & _Y)

