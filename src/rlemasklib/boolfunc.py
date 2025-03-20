from enum import IntEnum

import numpy as np

_X = 0b1100
_Y = 0b1010


class BoolFunc(IntEnum):
    """Binary Boolean functions for merging masks.

    Any Boolean function can be expressed by combining :attr:`BoolFunc.A` and :attr:`BoolFunc.B`
    using bitwise operators. Several named functions are provided for convenience.

    Examples:
        >>> from rlemasklib import RLEMask, BoolFunc
        >>> m1 = RLEMask.from_array(np.eye(3))
        >>> m2 = RLEMask.from_array(np.eye(3)[::-1])
        >>> m3 = m1.merge(m2, ~BoolFunc.A & BoolFunc.B)
        >>> m3.to_array()
        array([[0, 0, 1],
               [0, 0, 0],
               [1, 0, 0]], dtype=uint8)

        >>> from rlemasklib import encode, merge, decode, BoolFunc
        >>> d1 = encode(np.eye(3))
        >>> d2 = encode(np.eye(3)[::-1])
        >>> d3 = merge([d1, d2], ~BoolFunc.A & BoolFunc.B)
        >>> decode(d3)
        array([[0, 0, 1],
               [0, 0, 0],
               [1, 0, 0]], dtype=uint8)


    """

    A = _X
    """The first argument."""

    B = _Y
    """The second argument."""

    UNION = OR = _X | _Y
    """Union (disjunction) of the two arguments."""

    INTERSECTION = AND = _X & _Y
    """Intersection (conjunction) of the two arguments."""

    DIFFERENCE = _X & ~_Y
    """Difference (subtraction) of the two arguments."""

    SYMMETRIC_DIFFERENCE = _X ^ _Y
    """Symmetric difference (exclusive or) of the two arguments (same as :attr:`XOR`)."""

    XOR = _X ^ _Y
    """Symmetric difference (exclusive or) of the two arguments (same as :attr:`SYMMETRIC_DIFFERENCE`)."""

    EQUIVALENCE = ~(_X ^ _Y)
    """Equivalence (biconditional) of the two arguments (same as :attr:`XNOR`)."""

    XNOR = ~(_X ^ _Y)
    """Equivalence (biconditional) of the two arguments (same as :attr:`EQUIVALENCE`)."""

    IMPLICATION = ~_X | _Y
    """Implication (conditional) of the two arguments."""

    NOR = ~(_X | _Y)
    """NOR (neither) of the two arguments."""

    NAND = ~(_X & _Y)
    """NAND (not both) of the two arguments."""
