import itertools
from collections.abc import Iterable
from typing import Union, Sequence, Optional, Callable

import numpy as np
from rlemasklib.boolfunc import BoolFunc
from rlemasklib.oop_cython import RLECy


class RLEMask:
    """Run-length encoded mask.

    The RLEMask class represents a binary mask using run-length encoding. The mask can be
    created from a dense array, a bounding box, a polygon, or a circle, or from run-length
    counts. The mask can be manipulated using set operations like union, intersection, and
    difference, and can be converted to a dense array. Morphological operations, warping,
    transpose, flipping, cropping, padding, connected components, and other operations are
    also supported.

    The main constructor can take a dense array, a dictionary, or a list of run-length counts.

    It is recommended to use the static factory methods :meth:`from_array`, :meth:`from_dict`,
    :meth:`from_counts`, :meth:`from_bbox`, :meth:`from_polygon`, :meth:`zeros`, and :meth:`ones`
    to create new RLEMask objects, as they are more explicit.

    Args:
        obj: the input object to create the mask from. It can be a dense 2D array, a dictionary, or
            a list/1D-array of run-length counts.
        shape: [height, width] of the mask, in case the input is a list of run-length counts.
    """

    __slots__ = ['cy']

    def __init__(self, obj, *, shape: Optional[Sequence[int]] = None):
        self.cy = RLECy()
        if isinstance(obj, np.ndarray) and obj.ndim == 2:
            self.cy._i_from_array(obj)
        elif isinstance(obj, dict):
            self.cy._i_from_dict(obj)
        elif (isinstance(obj, np.ndarray) and obj.ndim == 1) or isinstance(obj, Sequence):
            counts = np.ascontiguousarray(obj, dtype=np.uint32)
            if shape is None:
                raise ValueError("shape must be provided when counts are given")
            if np.sum(counts) != shape[0] * shape[1]:
                raise ValueError("The sum of the counts must be equal to height * width.")
            self.cy._i_from_counts(shape, counts, 'F')
        else:
            raise ValueError("Unknown input type")

    @staticmethod
    def _init(cy: Optional[RLECy] = None):
        result = RLEMask.__new__(RLEMask)
        result.cy = cy if cy is not None else RLECy()
        return result

    @staticmethod
    def from_counts(
        counts: Sequence[int], shape: Sequence[int], order='F', validate_sum: bool = True
    ) -> "RLEMask":
        """Create an RLEMask object from run-length counts.

        Args:
            counts: the run-length counts of the mask, as a list of integers or a numpy array,
                where odd-indexed elements are runs of 0s and even-indexed elements are runs of 1s.
                The sum of the counts must be equal to height * width.
            shape: [height, width] of the mask
            order: the order of the counts in the list, either 'F' or 'C' for Fortran
                (column major) or C (row major) order
        """
        counts = np.ascontiguousarray(counts, dtype=np.uint32)
        if validate_sum and np.sum(counts) != shape[0] * shape[1]:
            raise ValueError("The sum of the counts must be equal to height * width.")

        if order not in ('F', 'C'):
            raise ValueError("Unknown order, must be 'F' or 'C'")

        result = RLEMask._init()
        result.cy._i_from_counts(shape, counts, order)
        return result

    @staticmethod
    def from_array(
        mask_array: np.ndarray, thresh128: bool = False, is_sparse: bool = True
    ) -> "RLEMask":
        """Create an RLEMask object from a dense mask.

        By default, any nonzero value is considered foreground and zero is considered background.
        If ``thresh128`` is set to True, then values greater than or equal to 128 are considered
        foreground and less than 128 are considered background.

        If `mask_array` is C contiguous, a transpose has to take place since the internal RLE
        format encodes the mask in Fortran order. If `is_sparse` is set to True, the transpose,
        if necessary, will be performed in RLE format, otherwise it will be performed in dense
        array format.

        Args:
            mask_array: a numpy array of numerical type where nonzero means foreground and zero
                means background.
            thresh128: whether to use 128 as the threshold for binarization (default is 1)
            is_sparse: hint that it is more efficient to transpose the mask in RLE form, only
                affects efficiency when the mask is C contiguous.
        """
        result = RLEMask._init()
        result.cy._i_from_array(mask_array, thresh128, is_sparse)
        return result

    @staticmethod
    def from_dict(d: dict) -> "RLEMask":
        """Create an RLEMask object from an RLE dictionary.

        Args:
            d: RLE dictionary with

                - ``"size"`` -- [height, width] of the mask
                - ``"counts"`` -- LEB128-like compressed run-length counts as in pycocotools, or
                - ``"zcounts"`` -- zlib compressed ``"counts"``, or
                - ``"ucounts"`` -- uncompressed ``"counts"``

        Returns:
            An RLEMask object representing the input mask.
        """
        result = RLEMask._init()
        result.cy._i_from_dict(d)
        return result

    @staticmethod
    def from_bbox(bbox, imshape=None, imsize=None) -> "RLEMask":
        """Create an RLEMask object from a bounding box.

        Args:
            bbox: a bounding box, in the format [x_start, y_start, width, height]
            imshape: [height, width] of the desired mask (either this or imsize must be provided)
            imsize: [width, height] of the desired mask (either this or imshape must be provided)

        Returns:
            An RLEMask object where the area of the provided bounding box has the value 1, and \
                the rest is 0.
        """
        result = RLEMask._init()
        result.cy._i_from_bbox(bbox, imshape=_get_imshape(imshape, imsize))
        return result

    @staticmethod
    def from_circle(center, radius, imshape=None, imsize=None) -> "RLEMask":
        """Create an RLEMask object representing a filled circle.

        Args:
            center: the center of the circle, in the format [x, y]
            radius: the radius of the circle
            imshape: [height, width] of the desired mask (either this or imsize must be provided)
            imsize: [width, height] of the desired mask (either this or imshape must be provided)

        Returns:
            An RLEMask object where the area of the provided circle has the value 1, and the \
                rest is 0.
        """
        result = RLEMask._init()
        result.cy._i_from_circle(center, radius, imshape=_get_imshape(imshape, imsize))
        return result

    @staticmethod
    def from_polygon(poly, imshape=None, imsize=None) -> "RLEMask":
        """Create an RLEMask object from a polygon.

        Args:
            poly: a polygon (list of xy coordinates)
            imshape: [height, width] of the desired mask (either this or imsize must be provided)
            imsize: [width, height] of the desired mask (either this or imshape must be provided)

        Returns:
            An RLEMask object representing the input polygon (1 inside the polygon, 0 outside).
        """

        result = RLEMask._init()
        imshape = _get_imshape(imshape, imsize)
        result.cy._i_from_polygon(poly.reshape(-1), imshape)
        return result

    @staticmethod
    def zeros(shape: Sequence[int]) -> "RLEMask":
        """Create a new RLE mask of zeros.

        Args:
            shape: the shape of the mask

        Returns:
            A new RLEMask object representing a mask of zeros.
        """
        result = RLEMask._init()
        result.cy._i_zeros(shape)
        return result

    @staticmethod
    def ones(shape: Sequence[int]) -> "RLEMask":
        """Create a new RLE mask of ones.

        Args:
            shape: the shape of the mask

        Returns:
            A new RLEMask object representing a mask of ones.
        """
        result = RLEMask._init()
        result.cy._i_ones(shape)
        return result

    @staticmethod
    def ones_like(mask) -> "RLEMask":
        """Create a new RLE mask of ones with the same shape as another mask.

        Args:
            mask: any other object with a mask.shape[0] and mask.shape[1]
                (e.g., RLEMask, NumPy array)

        Returns:
            A new RLEMask object representing a mask of ones with the same shape as the input \
                mask.
        """
        return RLEMask.ones(mask.shape)

    @staticmethod
    def zeros_like(mask) -> "RLEMask":
        """Create a new RLE mask of zeros with the same shape as another mask.

        Args:
            mask: any other object with a mask.shape[0] and mask.shape[1]
                (e.g., RLEMask, NumPy array)

        Returns:
            A new RLEMask object representing a mask of zeros with the same shape as the input
                mask.
        """
        return RLEMask.zeros(mask.shape)

    @property
    def shape(self) -> tuple[int, int]:
        """The shape of the mask (height, width).

        Returns:
            A tuple of the shape of the mask (height, width).
        """
        return self.cy.shape

    @property
    def counts(self) -> np.ndarray:
        """The run-length counts of the mask, as a copy of the underlying data.

        Returns:
            A numpy array of the run-length counts as integers.
        """
        return np.array(self.cy._counts_view(), copy=True)

    @property
    def counts_view(self) -> np.ndarray:
        """The run-length counts of the mask, as a direct view of the underlying memory.

        Modifications to the returned array will affect the mask.

        Returns:
            An array view of the run-length counts.
        """
        return self.cy._counts_view()

    @property
    def density(self) -> float:
        """The ratio of the number of runlengths vs number of pixels of the entire mask
        (not just the foreground)."""
        h, w = self.shape
        m = self.counts_view.size
        return m / (h * w)

    @property
    def T(self) -> "RLEMask":
        """The transpose of the mask (i.e., columns become rows and vice versa)

        Returns:
            A new RLEMask object representing the transpose of the mask.

        See Also:
            :meth:`transpose`
        """
        return self.transpose()

    def area(self) -> int:
        """The area of the mask (number of foreground pixels)."""
        return self.cy.area()

    def perimeter(self) -> int:
        """The number of pixels along the contour of the mask.

        See Also:
            :meth:`contours`
        """
        return self.cy._r_contours().area()

    def is_valid_rle(self) -> bool:
        """Check if the RLE mask is valid (no nonfirst zero runs and runs summing to H*W).

        The RLE mask is valid if it has no zero sized runs except perhaps in the first place,
        and the sum of the runs is equal to the number of pixels in the mask, as determined
        by the shape (height, width) of the mask.
        """
        return (
            not np.any(self.counts_view[1:] == 0)
            and np.sum(self.counts_view) == self.shape[0] * self.shape[1]
        )

    def count_nonzero(self) -> int:
        """The number of nonzero pixels in the mask, equivalent to :meth:`area`."""
        return self.cy.area()

    def nonzero(self) -> np.ndarray:
        """The indices of the nonzero elements in the mask as a 2D numpy array.

        The array contains the (x, y) coordinates of the foreground pixels as an Ax2 array, where
        A is the number of foreground pixels (foreground area).
        The coordinate order is (x, y) or in other words, (column, row).
        """
        return self.cy.nonzero_indices()

    def __getitem__(
        self, key: Union[tuple[slice, slice], tuple[int, int]]
    ) -> Union[int, "RLEMask"]:
        """Crop the RLE mask to get a submask, by slicing, or retrieve a single pixel value.

        Args:
            key: a tuple of two slices or two ints, one for height and one for width

        Returns:
            A new RLEMask object representing the submask.

        Examples:
            With slices:

            >>> rle = RLEMask(np.eye(4))
            >>> rle[1:3, 2:4].shape
            (2, 2)

            With integers:

            >>> rle = RLEMask(np.eye(4))
            >>> rle[1, 1]
            1

        """
        if (isinstance(key, tuple) and len(key) == 2) or isinstance(key, slice):
            # Cropping via indexing like rle[1:2, 3:4:2]
            h, w = self.shape

            if isinstance(key, slice):
                key = (key, slice(None))

            if isinstance(key[0], slice) and isinstance(key[1], slice):
                # substitute negative indices and None:
                start_h, span_h, step_h, flip_h = _forward_slice(key[0], h)
                start_w, span_w, step_w, flip_w = _forward_slice(key[1], w)
                cropped_cy = self.cy._r_crop(start_h, start_w, span_h, span_w, step_h, step_w)

                if flip_h and flip_w:
                    cropped_cy._i_rotate_180()
                elif flip_h:
                    cropped_cy = cropped_cy._r_vertical_flip()
                elif flip_w:
                    cropped_cy = cropped_cy._r_vertical_flip()
                    cropped_cy._i_rotate_180()

                return RLEMask._init(cropped_cy)
            # Indexing like rle[1, 2]
            elif isinstance(key[0], int) and isinstance(key[1], int):
                if key[0] < 0:
                    key = (h + key[0], key[1])
                if key[1] < 0:
                    key = (key[0], w + key[1])
                return self.cy._get_int_index(key[0], key[1])
            else:
                raise ValueError("Either slices or integers are supported, not a combination.")
        else:
            raise ValueError("Only 2D slicing is supported with integer indices")

    def __setitem__(
        self, key: Union[tuple[slice], tuple[int]], value: Union[int, "RLEMask", np.ndarray]
    ):
        """Set the value of a submask to either a constant or another RLE or dense mask.

        Args:
            key: a tuple of two slices, one for height and one for width
            value: either a constant (0 or 1) or another RLEMask or a numpy mask with the same size
                as the submask

        Examples:
            >>> rle = RLEMask.ones((4, 4))
            >>> rle[1:3, 2:4] = 0
            >>> np.array(rle)
            array([[1, 1, 1, 1],
                   [1, 1, 0, 0],
                   [1, 1, 0, 0],
                   [1, 1, 1, 1]], dtype=uint8)

        """

        if isinstance(key, tuple) and len(key) == 2:
            if isinstance(value, np.ndarray):
                value = RLEMask.from_array(value)

            h, w = self.shape

            if isinstance(key[0], slice) and isinstance(key[1], slice):
                # substitute negative indices and None:
                start_h, span_h, step_h, flip_h = _forward_slice(key[0], h)
                start_w, span_w, step_w, flip_w = _forward_slice(key[1], w)
                # step must be 1
                if step_h != 1 or step_w != 1:
                    raise ValueError("Only step=1 or -1 is supported for setting values")

                boxmask_cy = RLECy()
                boxmask_cy._i_from_bbox([start_w, start_h, span_w, span_h], (h, w))
                if isinstance(value, RLEMask):
                    value_cy = value.cy
                    if flip_h and flip_w:
                        value_cy = value_cy._r_rotate_180()
                    elif flip_h:
                        value_cy = value_cy._r_vertical_flip()
                    elif flip_w:
                        value_cy = value_cy._r_vertical_flip()
                        value_cy._i_rotate_180()

                    padded_cy = value_cy._r_zeropad(
                        start_w, w - (start_w + span_w), start_h, h - (start_h + span_h), 0
                    )
                    self.cy = self.cy._r_diffor(boxmask_cy, padded_cy)
                elif value == 0:
                    self.cy = self.cy._r_boolfunc(boxmask_cy, BoolFunc.DIFFERENCE.value)
                elif value == 1:
                    self.cy = self.cy._r_boolfunc(boxmask_cy, BoolFunc.OR.value)
                else:
                    raise ValueError("Value must be an RLE or 0 or 1")
            elif isinstance(key[0], int) and isinstance(key[1], int):
                if key[0] < 0:
                    key = (h + key[0], key[1])
                if key[1] < 0:
                    key = (key[0], w + key[1])
                if not isinstance(value, int):
                    raise ValueError("Value must be an integer when indexing with integers")
                self.cy._i_set_int_index(key[0], key[1], value)
        else:
            raise ValueError("Only 2D indexing is supported")

    def __invert__(self) -> "RLEMask":
        """Compute the complement of an RLE mask.

        Returns:
            A new RLEMask object representing the complement of the mask.

        Examples:
            >>> rle = RLEMask(np.eye(3))
            >>> np.array(~rle)
            array([[0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 0]], dtype=uint8)

        See Also:
            :meth:`complement`
        """
        return self.complement(inplace=False)

    def __or__(self, other: "RLEMask") -> "RLEMask":
        """Compute the union of two RLE masks.

        Args:
            other: another RLE mask

        Returns:
            A new RLEMask object representing the union of the two masks.

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> np.array(rle1 | rle2)
            array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]], dtype=uint8)

        See Also:
            :meth:`union`
        """
        return RLEMask._init(self.cy._r_boolfunc(other.cy, BoolFunc.OR.value))

    def __ior__(self, other: "RLEMask") -> "RLEMask":
        """Compute the union with another RLEMask in place.

        Args:
            other: the other RLEMask

        Returns:
            Self

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3, k=-1))
            >>> np.array(rle1 | rle2)
            array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]], dtype=uint8)
        """
        self._raise_if_different_shape(other)
        self.cy = self.cy._r_boolfunc(other.cy, BoolFunc.OR.value)
        return self

    def __and__(self, other: "RLEMask") -> "RLEMask":
        """Compute the intersection of two RLEMasks.

        Args:
            other: another RLEMask

        Returns:
            A new RLEMask object representing the intersection of the two masks.

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> np.array(rle1 & rle2)
            array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=uint8)

        See Also:
            :meth:`intersection`
        """
        self._raise_if_different_shape(other)
        return RLEMask._init(self.cy._r_boolfunc(other.cy, BoolFunc.AND.value))

    def __iand__(self, other: "RLEMask") -> "RLEMask":
        """Compute the intersection with another RLEMask in place.

        Args:
            other: the other RLEMask

        Returns:
            Self

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> np.array(rle1 & rle2)
            array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 0, 0]], dtype=uint8)
        """
        self._raise_if_different_shape(other)
        self.cy = self.cy._r_boolfunc(other.cy, BoolFunc.AND.value)
        return self

    def __xor__(self, other: "RLEMask") -> "RLEMask":
        """Compute the symmetric difference of two RLEMasks.

        Args:
            other: another RLEMask

        Returns:
            A new RLEMask object representing the symmetric difference of the two masks.

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> np.array(rle1 ^ rle2)
            array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]], dtype=uint8)
        """
        self._raise_if_different_shape(other)
        return RLEMask._init(self.cy._r_boolfunc(other.cy, BoolFunc.XOR.value))

    def __ixor__(self, other: "RLEMask") -> "RLEMask":
        """Compute the symmetric difference with another RLEMasks in place.

        Args:
            other: the other RLEMask

        Returns:
            Self

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> np.array(rle1 ^ rle2)
            array([[1, 0, 1],
                   [0, 1, 0],
                   [1, 0, 1]], dtype=uint8)
        """
        self._raise_if_different_shape(other)
        self.cy = self.cy._r_boolfunc(other.cy, BoolFunc.XOR.value)
        return self

    def __sub__(self, other: "RLEMask") -> "RLEMask":
        """Compute the difference of two RLEMasks.

        Args:
            other: another RLE mask

        Returns:
            A new RLEMask object representing the difference of the two masks.

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> np.array(rle1 - rle2)
            array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, 1]], dtype=uint8)
        """
        self._raise_if_different_shape(other)
        return RLEMask._init(self.cy._r_boolfunc(other.cy, BoolFunc.DIFFERENCE.value))

    def __isub__(self, other: "RLEMask") -> "RLEMask":
        """Compute the difference with another RLEMask in place.

        Args:
            other: the other RLEMask

        Returns:
            Self

        Examples:
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> np.array(rle1 - rle2)
            array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, 1]], dtype=uint8)
        """
        self._raise_if_different_shape(other)
        self.cy = self.cy._r_boolfunc(other.cy, BoolFunc.DIFFERENCE.value)
        return self

    def __eq__(self, other):
        """Check if two RLEMasks are equal (same runlengths and shape).

        Returns:
            True if the masks are equal, False otherwise.
        """

        if isinstance(other, RLEMask):
            return self.cy == other.cy
        else:
            return False

    def __repr__(self):
        """A string representation of the RLEMask, containing the shape and the runlengths."""
        return f"RLEMask(shape={self.shape}, counts={repr(self.cy._counts_view().tolist())})"

    def __array__(self, dtype=None, copy=None):
        """Convert the RLEMask to a dense numpy array, used by numpy functions.

        Returns:
            A numpy array of type uint8 representing the mask with 0 and 1 values.

        See Also:
            :meth:`to_array`
        """
        if copy is False:
            raise ValueError("RLEMask cannot be viewed as a numpy array without copying")

        arr = self.to_array()
        if dtype is not None:
            arr = arr.astype(dtype, copy=False)
        return arr

    def any(self) -> bool:
        """Check if any pixel in the mask is foreground."""
        return len(self.counts_view) > 1

    def all(self) -> bool:
        """Check if all pixels in the mask are foreground."""
        counts_view = self.counts_view
        return len(counts_view) == 2 and counts_view[0] == 0

    def dilate_vertical(self, up: int = 0, down: int = 0, inplace: bool = False) -> "RLEMask":
        """Dilate the mask vertically.

        Every foreground pixel causes a given number of its upper and lower neighbors to be set as
        foreground.

        Args:
            up: the number of pixels to dilate upwards
            down: the number of pixels to dilate downwards
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the dilated mask (self if inplace=True)
        """
        result = self if inplace else self.copy()
        result.cy._i_dilate_vertical(up, down)
        return result

    @staticmethod
    def merge_count(masks: Sequence["RLEMask"], threshold: int) -> "RLEMask":
        """Return a mask where each pixel is set if and only if at least `threshold` of the
        input masks have that pixel set.

        For example, if ``threshold`` is set as half the number of masks, then the result will
        be the majority vote of the masks.

        Args:
            masks: a list of RLEMask objects
            threshold: the threshold for merging

        Returns:
            A new RLEMask object representing the merged mask.
        """
        return RLEMask._init(RLECy.merge_many_atleast([m.cy for m in masks], threshold))

    def max_pool2x2(self, inplace=False) -> "RLEMask":
        """Max-pool the mask by a factor of 2.

        Args:
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object representing the max-pooled mask.
        """
        h, w = self.shape
        hr = h - h % 2
        wr = w - w % 2
        dilated = self.dilate_vertical(up=1, inplace=inplace)
        result = dilated[:hr:2, :wr:2] | dilated[:hr:2, 1:wr:2]
        if inplace:
            self.cy = result.cy
            return self
        return result

    def min_pool2x2(self, inplace=False) -> "RLEMask":
        """Min-pool the mask by a factor of 2.

        Args:
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object representing the min-pooled mask.
        """
        result = self.complement(inplace=inplace)
        result.max_pool2x2(inplace=True)
        return result.complement(inplace=True)

    def avg_pool2x2(self) -> "RLEMask":
        """Average-pool the mask by a factor of 2.

        Returns:
            A new RLEMask object representing the average-pooled mask.

        See Also:
            :meth:`avg_pool2d_valid`
        """
        return RLEMask._init(RLECy._r_avg_pool2x2(self.cy))

    def avg_pool2d_valid(
        self, kernel_size: Sequence[int], stride: Sequence[int] = (1, 1), threshold: int = -1
    ) -> "RLEMask":
        """Perform a 2D average pooling with the given kernel size and threshold the result.

        This function does not perform any padding and only returns the "valid" part of the
        pooling, similar to "valid" padding mode in deep learning frameworks as opposed to
        "same" or "full" padding.

        Args:
            kernel_size: the size of the pooling kernel as two integers
            stride: the stride of the pooling as two integers
            threshold: the result is set to 1 if the pooled result is greater than this value

        Returns:
            A new RLEMask object representing the pooled and thresholded mask.

        See Also:
            :meth:`avg_pool2x2`
        """

        return RLEMask._init(
            self.cy._r_avg_pool_valid(
                kernel_size[0], kernel_size[1], threshold, stride[0], stride[1]
            )
        )

    def conv2d_valid(
        self, kernel: np.ndarray, stride: Sequence[int] = (1, 1), threshold: float = 0.5
    ) -> "RLEMask":
        """Perform a 2D convolution with the given weighted kernel and threshold the result.

        This function does not perform any padding and only returns the "valid" part of the
        convolution, similar to "valid" padding mode in deep learning frameworks as opposed to
        "same" or "full" padding.

        Args:
            kernel: the convolution kernel as a 2D numpy array
            stride: the stride of the convolution as two integers
            threshold: the result is set to 1 if the convolution result is greater than this value

        Returns:
            A new RLEMask object representing the convolved and thresholded mask.
        """
        return RLEMask._init(self.cy._r_conv2d_valid(kernel, threshold, stride[0], stride[1]))

    def resize(
        self,
        output_imshape: Optional[Sequence[int]],
        fx: Optional[float] = None,
        fy: Optional[float] = None,
    ) -> "RLEMask":
        """Resize the mask to a new shape.

        It is enough to provide either the `output_imshape` or the scaling factors `fx` and `fy`.

        Internally this is implemented as an affine transformation.

        Args:
            output_imshape: the shape of the output image as (height, width)
            fx: the scaling factor along the horizontal axis
            fy: the scaling factor along the vertical axis

        Returns:
            A new RLEMask object representing the resized mask.
        """

        if output_imshape is None:
            if fx is None or fy is None:
                raise ValueError("Either output_imshape or fx and fy must be provided")
            output_imshape = (int(round(self.shape[0] * fy)), int(round(self.shape[1] * fx)))

        if fx is None:
            fx = output_imshape[1] / self.shape[1]
        if fy is None:
            fy = output_imshape[0] / self.shape[0]

        affine_mat = np.array([[fx, 0, 0], [0, fy, 0]], np.float64)
        return self.warp_affine(affine_mat, output_imshape)

    def warp_affine(self, M: np.ndarray, output_imshape: Sequence[int]) -> "RLEMask":
        """Apply an affine warping transformation to the mask.

        The transformation matrix M should be the forward transformation, i.e. the output
        location of an input pixel is calculated as `x_out = M @ x_in_homogeneous`.

        Args:
            M: the affine transformation matrix as a 2x3 or 3x3 numpy array
            output_imshape: the shape of the output image as (height, width)

        Returns:
            A new RLEMask object representing the warped mask.

        See Also:
            :meth:`warp_perspective`, :meth:`warp_distorted`
        """

        return RLEMask._init(self.cy._r_warp_affine(M, output_imshape[0], output_imshape[1]))

    def warp_perspective(self, H: np.ndarray, output_imshape: Sequence[int]) -> "RLEMask":
        """Apply a perspective warping (homography) transformation to the mask.

        The transformation matrix H should be the forward transformation, i.e. the output
        location of an input pixel is calculated as `x_out_homogeneous = H @ x_in_homogeneous`.

        Args:
            H: the perspective transformation matrix as a 3x3 numpy array
            output_imshape: the shape of the output image as (height, width)

        Returns:
            A new RLEMask object representing the warped mask.

        See Also:
            :meth:`warp_affine`, :meth:`warp_distorted`
        """
        return RLEMask._init(self.cy._r_warp_perspective(H, output_imshape[0], output_imshape[1]))

    def warp_distorted(
        self,
        R1: np.ndarray,
        R2: np.ndarray,
        K1: np.ndarray,
        K2: np.ndarray,
        d1: np.ndarray,
        d2: np.ndarray,
        polar_ud1,
        polar_ud2,
        output_imshape: Sequence[int],
    ) -> "RLEMask":
        """[Experimental] Warp the mask according to changing lens-distorted camera parameters

        This function supports OpenCV-like lens distortion parameters. API design and documentation
        is subject to change.
        """
        return RLEMask._init(
            self.cy._r_warp_distorted(
                R1,
                R2,
                K1,
                K2,
                d1,
                d2,
                polar_ud1,
                polar_ud2,
                output_imshape[0],
                output_imshape[1],
            )
        )

    def pad(
        self, top, bottom, left, right, border_type='constant', value: int = 0, inplace=False
    ) -> "RLEMask":
        """Pad the mask with constant values.

        Args:
            top: the number of pixels to pad on the top
            bottom: the number of pixels to pad on the bottom
            left: the number of pixels to pad on the left
            right: the number of pixels to pad on the right
            border_type: either 'constant', 'replicate', or 'edge'
            value: the value to pad with (0 or 1), only used when border_type='constant'
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object representing the padded mask.

        See Also:
            :meth:`crop`, :meth:`shift`
        """

        if border_type == 'constant':
            if inplace:
                self.cy._i_zeropad(left, right, top, bottom, value)
                return self
            else:
                return RLEMask._init(self.cy._r_zeropad(left, right, top, bottom, value))
        elif border_type in ('replicate', 'edge'):
            res_cy = self.cy._r_pad_replicate(left, right, top, bottom)
            if inplace:
                self.cy = res_cy
                return self
            return RLEMask._init(res_cy)
        else:
            raise ValueError(f"Unknown border type {border_type}")

    def complement(self, inplace: bool = False) -> "RLEMask":
        """Compute the complement of an RLE mask.

        Args:
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the complement of the mask (self if inplace=True)

        See Also:
            :meth:`__invert__`, which provides the complement as the `~` operator.
        """
        if inplace:
            self.cy._i_complement()
            return self
        else:
            return RLEMask._init(self.cy._r_complement())

    def shift(
        self, offset: Sequence[int], border_value: int = 0, inplace: bool = False
    ) -> "RLEMask":
        """Shift (translate) the mask by an offset vector.

        Args:
            offset: the offset vector [dy, dx]
            border_value: the value to pad with (0 or 1)
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the shifted mask (self if inplace=True)
        """

        if np.array_equal(offset, (0, 0)):
            return self if inplace else self.copy()

        h, w = self.shape
        cropbox = np.maximum(0, np.array([-offset[1], -offset[0], w, h]))
        return self.pad(
            max(0, offset[0]),
            max(0, -offset[0]),
            max(0, offset[1]),
            max(0, -offset[1]),
            value=border_value,
            inplace=inplace,
        ).crop(cropbox, inplace=True)

    def dilate(self, kernel_shape='circle', kernel_size=7, inplace=False) -> "RLEMask":
        """Dilate a mask with a kernel of a given shape and size.

        Args:
            kernel_shape: the shape of the kernel, either 'circle' or 'square'
            kernel_size: the size of the kernel
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the dilated mask (self if inplace=True)

        See Also:
            :meth:`dilate3x3` for a 3x3 kernel.
            :meth:`dilate5x5` for a 5x5 kernel.
            :meth:`erode`
        """
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        radius = kernel_size // 2
        x = np.arange(-radius, 1)
        if kernel_shape == 'circle':
            heights = np.sqrt((kernel_size / 2) ** 2 - x**2).astype(np.uint32)
        elif kernel_shape == 'square':
            heights = np.ones_like(x, dtype=np.uint32) * radius
        elif kernel_shape == 'diamond':
            heights = np.abs(x).astype(np.uint32)
        elif kernel_shape == 'cross':
            heights = np.zeros_like(x, dtype=np.uint32)
            heights[-1] = radius
        else:
            raise ValueError("Unknown kernel shape")

        to_merge = []
        vertical = self if inplace else self.copy()
        height_diffs = np.diff(heights, prepend=0)
        for j, d in enumerate(height_diffs, start=-radius):
            if d > 0:
                vertical.cy._i_dilate_vertical(d, d)
            to_merge += [vertical.shift((0, j)), vertical.shift((0, -j))]
        merged = RLEMask.merge_many(to_merge, BoolFunc.OR)
        if inplace:
            self.cy = merged.cy
            return self
        return merged

    def erode(self, kernel_shape='circle', kernel_size=7, inplace=False) -> "RLEMask":
        """Erode a mask with a kernel of a given shape and size.

        Args:
            kernel_shape: the shape of the kernel, either 'circle' or 'square'
            kernel_size: the size of the kernel
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the eroded mask (self if inplace=True)

        See Also:
            :meth:`erode3x3` for a 3x3 kernel.
            :meth:`erode5x5` for a 5x5 kernel.
            :meth:`dilate`
        """
        result = self.complement(inplace=inplace)
        result.dilate(kernel_shape, kernel_size, inplace=True)
        return result.complement(inplace=True)

    def dilate3x3(self, connectivity: int = 4, inplace: bool = False) -> "RLEMask":
        """Dilate a mask with a 3x3 kernel.

        After dilation, all pixels that were foreground before remain foreground and additionally any
        pixel with at least one foreground neighbor (according to the specified connectivity, 4-way or
        8-way) becomes also foreground.

        Args:
            connectivity: either 4 or 8, the connectivity of the dilation. 4 means a cross-shaped
                kernel, 8 means a square kernel.
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the dilated mask (self if inplace=True)

        See Also:
            :meth:`dilate` for arbitrary kernel shapes.
            :meth:`dilate5x5` for a 5x5 kernel.
        """
        left_shifted = self.shift((0, -1))
        right_shifted = self.shift((0, 1))

        if connectivity == 4:
            vertical_dilated = self if inplace else self.copy()
            vertical_dilated.cy._i_dilate_vertical()
            merged = RLEMask.merge_many(
                [vertical_dilated, left_shifted, right_shifted], BoolFunc.OR
            )
        else:
            merged = RLEMask.merge_many([self, left_shifted, right_shifted], BoolFunc.OR)
            merged.cy._i_dilate_vertical()

        if inplace:
            self.cy = merged.cy
            return self
        return merged

    def erode3x3(self, connectivity: int = 4, inplace: bool = False) -> "RLEMask":
        """Erode a mask with a 3x3 kernel.

        After erosion, only those pixels remain foreground that were foreground before and all its
        neighbors (according to the specified connectivity, 4-way or 8-way) are also foreground.


        Args:
            connectivity: either 4 or 8, the connectivity of the erosion. 4 means a cross-shaped
                kernel, 8 means a square kernel.
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the eroded mask (self if inplace=True)

        See Also:
            :meth:`erode` for arbitrary kernel shapes.
            :meth:`erode5x5` for a 5x5 kernel.
        """
        result = self.complement(inplace=inplace)
        result.dilate3x3(connectivity, inplace=True)
        return result.complement(inplace=True)

    def dilate5x5(self, inplace: bool = False) -> "RLEMask":
        """Dilate a mask with a round 5x5 kernel.

        The kernel is 0 in the four corners, otherwise 1.

        ::

            0 1 1 1 0
            1 1 1 1 1
            1 1 1 1 1
            1 1 1 1 1
            0 1 1 1 0

        Args:
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the dilated mask (self if inplace=True)

        See Also:
            :meth:`dilate` for arbitrary kernel shapes.
            :meth:`dilate3x3` for a 3x3 kernel.
        """

        vertical = self if inplace else self.copy()
        vertical.cy._i_dilate_vertical()
        vertical3_left = vertical.shift((0, -2))
        vertical3_right = vertical.shift((0, 2))
        vertical.cy._i_dilate_vertical()
        vertical5_left = vertical.shift((0, -1))
        vertical5_right = vertical.shift((0, 1))

        merged = RLEMask.merge_many(
            [vertical5_left, vertical3_left, vertical, vertical5_right, vertical3_right],
            BoolFunc.OR,
        )
        if inplace:
            self.cy = merged.cy
            return self
        return merged

    def erode5x5(self, inplace: bool = False) -> "RLEMask":
        """Erode a mask with a round 5x5 kernel.

        The kernel is 0 in the four corners, otherwise 1.

        ::

            0 1 1 1 0
            1 1 1 1 1
            1 1 1 1 1
            1 1 1 1 1
            0 1 1 1 0

        Args:
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the eroded mask (self if inplace=True)

        See Also:
            :meth:`erode` for arbitrary kernel shapes.
            :meth:`erode3x3` for a 3x3 kernel.
        """
        result = self.complement(inplace=inplace)
        result.dilate5x5(inplace=True)
        return result.complement(inplace=True)

    def contours(self):
        """An RLE consisting of those foreground pixels that have a background neighbor.

        4-neighbourhood is used.

        See Also:
            :meth:`perimeter` for the contour length.
        """
        return RLEMask._init(self.cy._r_contours())

    def largest_interior_rectangle(self, aspect_ratio: Optional[float] = None):
        """The largest axis-aligned rectangle that fits entirely inside the foreground.

        Args:
            aspect_ratio: the desired aspect ratio of the rectangle (width/height)

        Returns:
            An array (x, y, width, height) of the top-left corner and dimensions of the rectangle.

        See Also:
            :meth:`largest_interior_rectangle_around` for a rectangle around a given center point.
        """
        if aspect_ratio is None:
            return self.cy.largest_interior_rectangle()
        return self.cy.largest_interior_rectangle_aspect(aspect_ratio)

    def largest_interior_rectangle_around(
        self, center_point: Sequence[int], aspect_ratio: Optional[float] = None
    ):
        """The largest axis-aligned foreground rectangle with a given center point.

        Without `aspect_ratio`, the rectangle will have odd height and odd width and have
        the pixel at position `center_point` as its central pixel.
        When aspect_ratio is given, the rectangle may have non-integer dimensions.

        If the output is `r`, the following will hold:

        - ``center_point[0] == r[0] + (r[2]-1) / 2``
        - ``center_point[1] == r[1] + (r[3]-1) / 2``
        - ``r[2] / r[3] == aspect_ratio``
        - the region defined by ``r`` is entirely inside the foreground

        Args:
            center_point: the (x,y) center pixel coordinates of the rectangle
            aspect_ratio: the desired aspect ratio of the rectangle (width/height)

        Returns:
            An array (x, y, width, height) of the top-left corner and dimensions of the rectangle.

        See Also:
            :meth:`largest_interior_rectangle` for the largest rectangle without specifying the center point.
        """
        if aspect_ratio is None:
            aspect_ratio = 0.0

        # cx_round = int(center_point[0])
        # cy_round = int(center_point[1])
        rect = self.cy.largest_interior_rectangle_around_center(
            center_point[1], center_point[0], aspect_ratio
        )
        #
        # if np.all(rect == 0):
        #     return rect
        #
        # # adjust it to make it precisely centered on center_point, we can only shrink it.
        # current_center = (rect[0]-0.5 + rect[2]/2, rect[1]-0.5 + rect[3]/2)
        # dx = center_point[0] - current_center[0]
        # dy = center_point[1] - current_center[1]
        # print(dx, dy, current_center, center_point)
        #
        # # if dx > 0:
        # #     rect[0] += dx * 2
        # #     rect[2] -= dx * 2
        # # else:
        # #     rect[2] += dx * 2
        # #
        # # if dy > 0:
        # #     rect[1] += dy * 2
        # #     rect[3] -= dy * 2
        # # else:
        # #     rect[3] += dy * 2

        return rect

    def merge(self, other: "RLEMask", func: BoolFunc) -> "RLEMask":
        """Merge this mask with another using a Boolean function.

        Args:
            other: the other RLE mask
            func: the Boolean function to apply

        Returns:
            A new RLEMask object representing the result of the merge.

        See Also:
            :meth:`merge_custom`, which allows merging with custom n-ary Boolean functions.
            :meth:`merge_many`, which allows merging with different binary Boolean functions.
        """
        self._raise_if_different_shape(other)
        return RLEMask._init(self.cy._r_boolfunc(other.cy, func.value))

    @staticmethod
    def intersection(masks: Sequence["RLEMask"]) -> "RLEMask":
        """Return a mask where each pixel is set if and only if all input masks have the pixel set.

        Args:
            masks: a list of RLEMask objects

        Returns:
            A new RLEMask object representing the intersection of the masks.

        See Also:
            :meth:`__and__`, which provides the intersection as the ``&`` operator.
            :meth:`merge_many`, which allows merging with different binary Boolean functions.
            :meth:`merge_many_custom`, which allows merging with custom n-ary Boolean functions.
        """
        return RLEMask.merge_many(masks, BoolFunc.AND)

    @staticmethod
    def union(masks: Sequence["RLEMask"]) -> "RLEMask":
        """Return a mask where each pixel is set if at least one of the input masks has the pixel set.

        Args:
            masks: a list of RLEMask objects

        Returns:
            A new RLEMask object representing the union of the masks.

        See Also:
            :meth:`__or__`, which provides the union as the ``|`` operator.
        """
        return RLEMask.merge_many(masks, BoolFunc.OR)

    @staticmethod
    def merge_many(
        masks: Sequence["RLEMask"], func: Union[BoolFunc, Sequence[BoolFunc]]
    ) -> "RLEMask":
        """Merge many masks using either the same or different Boolean functions.

        This is a reduce operation from the left, as:

        merge(merge(masks[0], masks[1], func[0]), masks[2], func[1]), ...

        If only one function is provided, it is applied in all steps.

        Args:
            masks: a sequence of RLE masks
            func: a single Boolean function or a sequence of Boolean functions

        Returns:
            A new RLEMask with the merged result.

        See Also:
            :meth:`merge_many_custom`, which allows merging with custom n-ary Boolean functions.
        """

        if len(masks) == 0:
            raise ValueError("At least one mask must be provided")

        if not all(m.shape == masks[0].shape for m in masks[1:]):
            raise ValueError("All masks must have the same shape.")

        if isinstance(func, BoolFunc):
            return RLEMask._init(RLECy.merge_many_singlefunc([m.cy for m in masks], func.value))

        # if ((isinstance(func, BoolFunc) and func == BoolFunc.OR) or
        #         all(x==BoolFunc.OR for x in func)):
        #     return RLEMask._init(RLECy.merge_many_or([m.cy for m in masks]))
        #
        # if ((isinstance(func, BoolFunc) and func == BoolFunc.AND) or
        #         all(x==BoolFunc.AND for x in func)):
        #     return RLEMask._init(RLECy.merge_many_and([m.cy for m in masks]))
        #
        # if ((isinstance(func, BoolFunc) and func == BoolFunc.DIFFERENCE) or
        #         all(x == BoolFunc.DIFFERENCE for x in func)):
        #     if len(masks) == 1:
        #         return masks[0].copy()
        #     diffs = RLEMask._init(RLECy.merge_many_or([m.cy for m in masks[1:]]))
        #     return masks[0] - diffs

        func = [func] * (len(masks) - 1)

        return RLEMask._init(
            RLECy.merge_many_multifunc([m.cy for m in masks], [f.value for f in func])
        )

    @staticmethod
    def merge_many_custom(masks: Sequence["RLEMask"], func: Callable[..., bool]) -> "RLEMask":
        """Merge many masks using a custom n-ary boolean function.

        This first calls func with all combinations of n boolean arguments and stores the resulting
        truth table.
        It then returns a new mask where the ith pixel is the result of func applied to the ith
        pixel of all input masks.

        Args:
            masks: a sequence of at RLE masks
            func: a callable that takes n bools and returns a bool

        Returns:
            A new RLEMask with the merged result.

        See Also:
            :meth:`merge_many`, which allows merging with binary Boolean functions.
            :meth:`make_merge_function`, which creates a merge function from a custom n-ary function.
        """
        if len(masks) == 0:
            raise ValueError("At least one mask must be provided")

        mergefun = RLEMask.make_merge_function(func, arity=len(masks))
        return mergefun(*masks)

    @staticmethod
    def make_merge_function(func: Callable[..., bool], arity: Optional[int] = None):
        """Create a merge function from a custom n-ary boolean function.

        This first calls func with all combinations of n boolean arguments and stores the resulting
        truth table.
        It then returns a function that takes n masks and merges them using the truth table.

        Args:
            func: a callable that takes n bools and returns a bool
            arity: the number of arguments to the function (default: None, which is determined
                automatically)

        Returns:
            A callable that takes masks and returns the merged mask.

        Examples:
            >>> mergefun = RLEMask.make_merge_function(lambda a, b, c: (a | b) & ~c)
            >>> rle1 = RLEMask(np.eye(3))
            >>> rle2 = RLEMask(np.eye(3)[::-1])
            >>> rle3 = RLEMask(np.eye(3, k=-1))
            >>> rle = mergefun(rle1, rle2, rle3)
            >>> np.array(rle)
            array([[1, 0, 0],
                   [0, 0, 0],
                   [0, 0, 1]], dtype=uint8)

        See Also:
            :meth:`merge_many_custom`
        """

        if arity is None:
            arity = func.__code__.co_argcount

        if arity == 0:
            raise ValueError("The function must take at least one argument")

        multiboolfunc = 0
        for i, args in enumerate(itertools.product([False, True], repeat=arity)):
            result = int(bool(func(*reversed(args))))
            multiboolfunc |= result << i

        mask = (1 << 64) - 1
        n_bits = multiboolfunc.bit_length()
        mbf = np.empty([(n_bits + 63) // 64], dtype=np.uint64)
        i = 0
        while multiboolfunc > 0:
            mbf[i] = mask & multiboolfunc
            multiboolfunc >>= 64
            i += 1

        def merge(*masks: "RLEMask") -> "RLEMask":
            if len(masks) != arity:
                raise ValueError(f"Expected {arity} masks, got {len(masks)}")

            if not all(m.shape == masks[0].shape for m in masks[1:]):
                raise ValueError("All masks must have the same shape.")

            return RLEMask._init(RLECy.merge_many_custom([m.cy for m in masks], mbf))

        return merge

    @staticmethod
    def merge_to_label_map(rles: Sequence['RLEMask']) -> np.ndarray:
        """Merge a list of RLE masks to a label map indicating which masks contains each pixel.

        That is, the output is an integer-valued numpy array containg for each pixel the (index+1)
        of the mask that has the pixel set, or 0 if no mask has the pixel set.

        If multiple masks have the pixel set, the index of the last among the input masks will
        be used in the output.
        """
        return RLECy.merge_to_label_map([r.cy for r in rles])

    def repeat(self, num_h: int, num_w: int, inplace: bool = False) -> "RLEMask":
        """Repeat the mask pixels multiple times along the axes.

        This method is analogous to :func:`np.repeat <numpy.repeat>` (not :func:`np.tile <numpy.tile>`).

        This repeats each pixel in the mask `num_h` times along the vertical axis and `num_w` times
        along the horizontal axis.

        Args:
            num_h: the number of times to repeat the mask along the vertical axis
            num_w: the number of times to repeat the mask along the horizontal axis
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object representing the repeated mask (self if inplace=True)

        See Also:
            :meth:`tile`
        """
        if inplace:
            self.cy._i_repeat(num_h, num_w)
            return self
        else:
            return RLEMask._init(self.cy._r_repeat(num_h, num_w))

    def centroid(self) -> np.ndarray:
        """The centroid of the mask, as a numpy float32 array [x, y]."""
        return self.cy.centroid().astype(np.float32)

    def connected_components(self, connectivity: int = 4, min_size: int = 1):
        """Extract connected components from the mask.

        Args:
            connectivity: 4 or 8, the connectivity of the components. If 4, then only horizontal
                and vertical connections are considered, if 8, then also diagonal connections are
                considered.
            min_size: the minimum size of a component to return. Small components are ignored.

        Returns:
            A list of RLEMask objects, each representing a connected component of this mask.

        See Also:
            :meth:`largest_connected_component`, :meth:`remove_small_components`,
            :meth:`fill_small_holes`
        """
        components_cy = self.cy.connected_components(connectivity, min_size)
        return [RLEMask._init(c) for c in components_cy]

    def largest_connected_component(
        self, connectivity: int = 4, inplace: bool = False
    ) -> "RLEMask":
        """Extract the largest connected component of the mask.

        Args:
            connectivity: 4 or 8, the connectivity of the components. If 4, then only horizontal
                and vertical connections are considered, if 8, then also diagonal connections are
                considered.
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object representing the largest connected component of this mask.

        See Also:
            :meth:`connected_components`, :meth:`remove_small_components`
        """
        result = self if inplace else self.copy()
        result.cy._i_largest_connected_component(connectivity)
        return result

    def remove_small_components(
        self, min_size: int = 1, connectivity: int = 4, inplace: bool = False
    ) -> "RLEMask":
        """Remove small connected components from the mask.

        Args:
            min_size: the minimum size of a component to keep. Small components are removed.
            connectivity: 4 or 8, the neighborhood connectivity of the components
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object with small components removed.

        See Also:
            :meth:`fill_small_holes`
        """
        result = self if inplace else self.copy()
        result.cy._i_remove_small_components(min_size, connectivity)
        return result

    def fill_small_holes(
        self, min_size: int = 1, connectivity: int = 4, inplace: bool = False
    ) -> "RLEMask":
        """Fill small holes (i.e., connected components of the background) in the mask.

        Args:
            min_size: the minimum size of a hole to keep. Smaller holes are filled.
            connectivity: 4 or 8, the neighborhood connectivity of the components
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            A new RLEMask object with small holes filled.

        See Also:
            :meth:`remove_small_components`
        """
        result = self.complement(inplace=inplace)
        result.remove_small_components(min_size, connectivity, inplace=True)
        result.complement(inplace=True)
        return result

    def bbox(self) -> np.ndarray:
        """The bounding box of the foreground, i.e. the smallest rectangle that contains the mask.

        Returns:
            A float32 numpy array [x, y, width, height] of the bounding box of the mask.

        See Also:
            :meth:`largest_interior_rectangle` for the largest rectangle inside the mask.
        """
        return self.cy.bbox().astype(np.float32)

    def crop(self, bbox: np.ndarray, inplace=False) -> "RLEMask":
        """Crop the mask to the bounding box.

        Args:
            bbox: a bounding box, in the format [x_start, y_start, width, height]
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object representing the cropped mask.

        See Also:
            :meth:`tight_crop`
        """

        x0, y0, bw, bh = np.asanyarray(bbox, dtype=np.uint32)
        if inplace:
            self.cy._i_crop(y0, x0, bh, bw, 1, 1)
            return self
        else:
            return RLEMask._init(self.cy._r_crop(y0, x0, bh, bw, 1, 1))

    def tight_crop(self, inplace: bool = False) -> tuple["RLEMask", np.ndarray]:
        """Crop the mask to the bounding box of the foreground.

        Args:
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            A tuple consisting of the cropped mask and the box.

        See Also:
            :meth:`crop`
        """
        if inplace:
            box = self.cy._i_tight_crop()
            return self, box
        else:
            result, box = self.cy._r_tight_crop()
            return RLEMask._init(result), box

    def transpose(self) -> "RLEMask":
        """Transpose the mask, i.e. swap the axes such that columns become rows and vice versa.

        Returns:
            A new RLEMask object

        See Also:
            :meth:`T` for the transpose of the mask as a property.
        """
        return RLEMask._init(self.cy._r_transpose())

    def rot90(self, k=1, inplace=False) -> "RLEMask":
        """Rotate the mask by a multiple of 90 degrees.

        Args:
            k: the number of counter-clockwise 90-degree rotations to apply
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            The RLEMask object representing the rotated mask (self if inplace=True)

        See Also:
            :meth:`warp_affine` for arbitrary affine transformations including arbitrary rotations.
        """
        k %= 4
        if k == 0:
            return self if inplace else self.copy()
        elif k == 1:
            result_cy = self.cy._r_transpose()._r_vertical_flip()
            if inplace:
                self.cy = result_cy
                return self
            else:
                return RLEMask._init(result_cy)
        elif k == 2:
            if inplace:
                self.cy._i_rotate_180()
                return self
            else:
                return RLEMask._init(self.cy._r_rotate_180())
        elif k == 3:
            result_cy = self.cy._r_vertical_flip()._r_transpose()
            if inplace:
                self.cy = result_cy
                return self
            else:
                return RLEMask._init(result_cy)

    def flip(self, axis: int) -> "RLEMask":
        """Flip the mask along an axis.

        Args:
            axis: 0 for vertical flip, 1 for horizontal flip

        Returns:
            An RLEMask object of the flipped mask.

        See Also:
            :meth:`flipud`, :meth:`fliplr`
        """
        if axis == 0:
            return self.flipud()
        elif axis == 1:
            return self.fliplr()
        else:
            raise ValueError("Invalid axis")

    def flipud(self) -> "RLEMask":
        """Flip the mask vertically.

        Returns:
            A new RLEMask object

        See Also:
            :meth:`fliplr`, :meth:`flip`
        """
        return RLEMask._init(self.cy._r_vertical_flip())

    def fliplr(self) -> "RLEMask":
        """Flip the mask horizontally.

        Returns:
            An RLEMask object of the horizontally flipped mask.

        See Also:
            :meth:`flipud`, :meth:`flip`
        """

        result_cy = self.cy._r_vertical_flip()
        result_cy._i_rotate_180()
        return RLEMask._init(result_cy)

    @staticmethod
    def concatenate(masks: Iterable["RLEMask"], axis: int = 0) -> "RLEMask":
        """Concatenate masks along an axis.

        Args:
            masks: a sequence of RLE masks
            axis: the axis along which to concatenate (0 or 1)

        Returns:
            A new RLEMask object representing the concatenated masks.

        See Also:
            :meth:`hconcat`, :meth:`vconcat`
        """
        if axis == 0:
            return RLEMask.vconcat(masks)
        elif axis == 1:
            return RLEMask.hconcat(masks)
        else:
            raise ValueError("Invalid axis")

    @staticmethod
    def hconcat(masks: Iterable["RLEMask"]) -> "RLEMask":
        """Horizontally concatenate masks.

        Args:
            masks: a sequence of RLE masks

        Returns:
            A new RLEMask object representing the horizontally concatenated masks.

        See Also:
            :meth:`vconcat`, :meth:`concatenate`
        """
        cys = [m.cy for m in masks]
        if len(cys) == 0:
            raise ValueError("Cannot concatenate empty iterable of masks")
        if not all(cy.shape[0] == cys[0].shape[0] for cy in cys):
            raise ValueError("Masks must have the same height to be concatenated horizontally")

        return RLEMask._init(RLECy.concat_horizontal([m.cy for m in masks]))

    @staticmethod
    def vconcat(masks: Iterable["RLEMask"]) -> "RLEMask":
        """Vertically concatenate masks.

        Args:
            masks: a sequence of RLE masks

        Returns:
            A new RLEMask object representing the vertically concatenated masks.

        See Also:
            :meth:`hconcat`, :meth:`concatenate`
        """
        cys = [m.cy for m in masks]
        if len(cys) == 0:
            raise ValueError("Cannot concatenate empty iterable of masks")
        if not all(cy.shape[1] == cys[0].shape[1] for cy in cys):
            raise ValueError("Masks must have the same width to be concatenated vertically")

        return RLEMask._init(RLECy.concat_vertical([m.cy for m in masks]))

    def tile(self, num_h: int, num_w: int) -> "RLEMask":
        """Tile the mask multiple times along the axes, analogous to :func:`np.tile <numpy.tile>`.

        This repeats the mask `num_h` times along the vertical axis and `num_w` times along the
        horizontal axis.

        Args:
            num_h: the number of times to repeat the mask along the vertical axis
            num_w: the number of times to repeat the mask along the horizontal axis

        Returns:
            A new RLEMask object representing the tiled mask.

        See Also:
            :meth:`repeat`
        """

        if num_h == 0 or num_w == 0:
            return RLEMask.zeros((self.shape[0] * num_h, self.shape[1] * num_w))

        return RLEMask.hconcat([RLEMask.vconcat([self] * num_w)] * num_h)

    def copy(self) -> "RLEMask":
        """Clone the mask.

        Returns:
            A new RLEMask object representing the same mask.
        """
        return RLEMask._init(self.cy.clone())

    def fill_rectangle(self, rect: np.ndarray, value: int = 1, inplace: bool = False) -> "RLEMask":
        """Fill a rectangle in the mask.

        Args:
            rect: a rectangle, in the format [x_start, y_start, width, height]
            value: the value to fill with (0 or 1)
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object with the rectangle filled (self if inplace=True)

        See Also:
            :meth:`fill_circle`
        """
        boxmask = RLEMask.from_bbox(rect, imshape=self.shape)
        return self._fill_mask(boxmask, value, inplace=inplace)

    def fill_circle(
        self, center: np.ndarray, radius: float, value: int = 1, inplace: bool = False
    ) -> "RLEMask":
        """Fill a circle in the mask.

        Args:
            center: the center of the circle, in the format [x, y]
            radius: the radius of the circle
            value: the value to fill with (0 or 1)
            inplace: whether to perform the operation in place or to return a new object

        Returns:
            An RLEMask object with the circle filled (self if inplace=True)

        See Also:
            :meth:`fill_rectangle`
        """
        circle_mask = RLEMask.from_circle(center, radius, imshape=self.shape)
        return self._fill_mask(circle_mask, value, inplace=inplace)

    def _fill_mask(self, mask: "RLEMask", value: int, inplace: bool = False) -> "RLEMask":
        if inplace:
            if value == 1:
                return self.__ior__(mask)
            else:
                return self.__isub__(mask)
        else:
            if value == 1:
                return self | mask
            else:
                return self - mask

    def to_dict(self, zlevel: Optional[int] = None) -> dict:
        """Convert the RLE mask to a dictionary.

        Returns:
            A dictionary with the keys ``"size"`` and ``"counts"`` or ``"zcounts"``.

            - ``"size"`` -- [height, width] of the mask
            - ``"counts"`` -- LEB128-like compressed run-length counts as in pycocotools, or
            - ``"zcounts"``-- if zlevel is provided, ``"counts"`` is further compressed using zlib

        See Also:
            :meth:`from_dict`
        """
        return self.cy.to_dict(zlevel)

    def to_array(self, value: int = 1, order='F') -> np.ndarray:
        """Convert the RLE mask to a dense 2D uint8 numpy array.

        False (background) values become 0 and True (foreground) values become the specified value.
        The RLE is internally stored for the Fortran order, so order='F' is faster, because
        'C' requires a transpose. To improve efficiency, the transpose is done either in RLE or
        in dense form, depending on the sparseness of the mask.

        Args:
            value: the "True" value to use in the resulting array
            order: the order of the array ('C' for row-major, 'F' for column-major)

        Returns:
            An F or C-contiguous 2D numpy array of type uint8 representing the mask.

        See Also:
            :meth:`__array__`, :meth:`from_array`
        """
        return self.cy._r_to_dense_array(value, order)

    # def __getstate__(self):
    #     return self.to_dict()
    #
    # def __setstate__(self, state):
    #     self.cy = RLECy()
    #     self.cy._i_from_dict(state)

    def iou(self, other: "RLEMask") -> float:
        """Compute the intersection-over-union (IoU) between two masks.

        Args:
            other: another RLE mask

        Returns:
            The IoU value between the two masks.

        See Also:
            :meth:`iou_matrix` for computing the IoU between pairs of multiple masks.
        """

        return self.cy.iou(other.cy)

    @staticmethod
    def iou_matrix(masks1: Sequence["RLEMask"], masks2: Sequence["RLEMask"]) -> np.ndarray:
        """Compute the intersection-over-union (IoU) between two sets of masks.

        Args:
            masks1: a sequence of RLE masks
            masks2: a sequence of RLE masks

        Returns:
            A 2D numpy array of shape (len(masks1), len(masks2)) with the IoU values.

        See Also:
            :meth:`iou` for computing the IoU between two masks.
        """

        return RLECy.iou_matrix([m.cy for m in masks1], [m.cy for m in masks2])

    def _raise_if_different_shape(self, other: "RLEMask"):
        if self.shape != other.shape:
            raise ValueError("The masks must have the same shape.")


def _get_imshape(imshape=None, imsize=None):
    assert imshape is not None or imsize is not None
    if imshape is None:
        imshape = [imsize[1], imsize[0]]
    return imshape[:2]


def _forward_slice(slice_obj, length):
    """Convert a slice object to a forward slice.

    Args:
        slice_obj: a slice object
        length: the length of the array

    Returns:
        A tuple of (start, stop, step) for a forward slice and a boolean indicating if the slice
            is reversed.
    """
    r = range(length)[slice_obj]
    r_out = r[::-1] if r.step < 0 else r
    span = r_out[-1] + 1 - r_out.start if len(r_out) > 0 else 0
    step = max(1, min(span, r_out.step))
    flip = r.step < 0 and len(r_out) > 1

    if range(length)[0:length:step] == range(length)[r_out.start : r_out.start + span : step]:
        return 0, length, step, flip

    return r_out.start, span, step, flip


# def resize_mask_dense(mask_dict: dict, out_shape):
#     x = RLEMask.from_dict(mask_dict)
#     fx = out_shape[1] / x.shape[1]
#     fy = out_shape[0] / x.shape[0]
#     box_old = x.bbox()
#     box_end = box_old[:2] + box_old[2:]
#     box_start = np.maximum(0, box_old[:2] - 1)
#     box_end = np.minimum([x.shape[1], x.shape[0]], box_end + 1)
#     box_old = np.concatenate([box_start, box_end - box_start])
#     box_new = box_old * [fx, fy, fx, fy]
#     start = np.floor(box_new[:2]).astype(np.int32)
#     end = np.ceil(box_new[:2] + box_new[2:]).astype(np.int32)
#     box_new = np.concatenate([start, end - start])
#     box_old = box_new / [fx, fy, fx, fy]
#     start = np.floor(box_old[:2]).astype(np.int32)
#     end = np.ceil(box_old[:2] + box_old[2:]).astype(np.int32)
#     box_old = np.concatenate([start, end - start])
#     cropped = x.crop(box_old, inplace=False)
#     interp = cv2.INTER_LINEAR if fx > 1 and fy > 1 else cv2.INTER_AREA
#     resized = cv2.resize(
#         cropped.to_array(order='C', value=255),
#         (box_new[2], box_new[3]),
#         fx=fx, fy=fy, interpolation=interp)
#     resized = RLEMask.from_array(resized, thresh128=True, is_sparse=x.density < 0.04)
#     resized.pad(
#         top=box_new[1],
#         bottom=out_shape[0] - box_new[1] - resized.shape[0],
#         left=box_new[0],
#         right=out_shape[1] - box_new[0] - resized.shape[1],
#         inplace=True)
#     return resized.to_dict()
