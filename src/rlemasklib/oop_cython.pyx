# cython: language_level=3
# distutils: language = c

cimport cython
from cython cimport floating

import zlib
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, calloc
from typing import Optional
from collections.abc import Sequence, Iterable
from libc.stdint cimport uint64_t, uint32_t, uint8_t

# intialized Numpy. must do.
np.import_array()

# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memoery management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)
    int PyArray_SetBaseObject(np.ndarray arr, object base)

from cpython.ref cimport Py_INCREF

cdef extern from "stdbool.h":
    ctypedef int bool

cdef extern from "basics.h" nogil:
    ctypedef uint32_t uint
    ctypedef uint64_t siz
    ctypedef uint8_t byte
    ctypedef double *BB
    ctypedef struct RLE:
        siz h
        siz w
        siz m
        uint *cnts
        uint *alloc
    void rlesInit(RLE **R, siz n)
    uint *rleInit(RLE *R, siz h, siz w, siz m)
    uint *rleFrCnts(RLE *R, siz h, siz w, siz m, uint *cnts)
    void rleEliminateZeroRuns(RLE *R)
    void rleBorrow(RLE *R, siz h, siz w, siz m, uint *cnts)
    void rleCopy(const RLE *R, RLE *M)
    void rleMoveTo(RLE *R, RLE *M)
    byte rleGet(const RLE *R, siz i, siz j)
    void rleSetInplace(RLE *R, siz i, siz j, byte v)
    void rleOnes(RLE *R, siz h, siz w)
    void rleZeros(RLE *R, siz h, siz w)
    bool rleEqual(const RLE *A, const RLE *B)
    void rlesFree(RLE **R, siz n)
    void rleFree(RLE *R)
    void rleSwap(RLE *R, RLE *M)

cdef extern from "encode_decode.h" nogil:
    void rleEncode(RLE *R, const byte *M, siz h, siz w, siz n)
    void rleEncodeThresh128(RLE *R, const byte *M, siz h, siz w, siz n)
    void rleEncodeThreshold(RLE *R, const byte *M, siz h, siz w, siz n, int threshold)
    bool rleDecode(const RLE *R, byte *mask, siz n, byte value)
    bool rleDecodeStrided(const RLE *R, byte *M, siz row_stride, siz col_stride, byte value)
    bool rleDecodeBroadcast(const RLE *R, byte *M, siz num_channels, byte value)
    bool rleDecodeMultiValue(const RLE *R, byte *M, siz num_channels, const byte *values)
    char *rleToString(const RLE *R)
    bool rleFrString(RLE *R, const char *s, siz h, siz w)
    void rlesToLabelMapZeroInit(const RLE **R, byte *label_map, siz n, siz buffer_size)
    siz rleFromLabelMap(const byte *M, siz h, siz w, RLE *Rs)


cdef extern from "boolfuncs.h" nogil:
    void rleComplement(const RLE *R_in, RLE *R_out, siz n)
    void rleComplementInplace(RLE *R_in, siz n)
    void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc)
    void rleMergePtr(const RLE **R, RLE *M, siz n, uint boolfunc)
    void rleMerge2(const RLE *A, const RLE *B, RLE *M, uint boolfunc)
    void rleMergeMultiFunc(const RLE **R, RLE *M, siz n, uint *boolfuncs)
    void rleMergeDiffOr(const RLE *A, const RLE *B, const RLE *C, RLE *M)
    void rleMergeAtLeast2(const RLE **R, RLE *M, siz n, uint k)
    void rleMergeWeightedAtLeast2(const RLE **R, RLE *M, siz n, double *weights, double threshold)
    void rleMergeLookup(const RLE **R, RLE *M, siz n, uint64_t *multiboolfunc, siz nmbf)

cdef extern from "misc.h" nogil:
    void rleStrideInplace(RLE *R, siz sy, siz sx)
    void rleRepeatInplace(RLE *R, siz nh, siz nw)
    void rleRepeat(const RLE *R, RLE *M, siz nh, siz nw)
    void rleDilateVerticalInplace(RLE *R, uint up, uint down)
    void rleConcatHorizontal(const RLE **R, RLE *M, siz n)
    void rleConcatVertical(const RLE **R, RLE *M, siz n)
    void rleContours(const RLE *R, RLE *M)

cdef extern from "moments.h" nogil:
    void rleArea(const RLE *R, siz n, uint *a)
    void rleCentroid(const RLE *R, double *xys, siz n)
    void rleNonZeroIndices(const RLE *R, uint **coords_out, siz *n_out)
    void rleRawMoments(const RLE *R, double *moments)
    void rleMoments(const RLE *R, double *out)
    void rleHuMoments(const RLE *R, double *hu)

cdef extern from "connected_components.h" nogil:
    void rleConnectedComponents(
            const RLE *R_in, int connectivity, siz min_size, RLE **components, siz *n)
    void rleRemoveSmallConnectedComponentsInplace(RLE *R_in, siz min_size, int connectivity)
    void rleLargestConnectedComponentInplace(RLE *R_in, int connectivity)

    ctypedef struct CCState:
        pass

    CCState *rleConnectedComponentsBegin(
            const RLE *R_in,
            int connectivity,
            siz min_size,
            siz *n_components_out,
            siz **areas_out,
            int **bboxes_out,
            double **centroids_out) nogil

    void rleConnectedComponentsExtract(
            CCState *state,
            bool *selected,
            RLE **components_out,
            siz *n_selected_out) nogil

    void rleConnectedComponentsEnd(CCState *state) nogil

    siz rleConnectedComponentStats(
            const RLE *R_in,
            int connectivity,
            siz min_size,
            siz **areas_out,
            int **bboxes_out,
            double **centroids_out) nogil

    siz rleCountConnectedComponents(const RLE *R_in, int connectivity, siz min_size) nogil

cdef extern from "pad_crop.h" nogil:
    void rleCrop(const RLE *R_in, RLE *R_out, siz n, const uint *bbox)
    void rleCropInplace(RLE *R_in, siz n, const uint *bbox)
    void rleZeroPad(const RLE *R_in, RLE *R_out, siz n, const uint *pad_amounts)
    void rleZeroPadInplace(RLE *R, siz n, const uint *pad_amounts)
    void rlePadReplicate(const RLE *R_in, RLE *R_out, const uint *pad_amounts)

cdef extern from "shapes.h" nogil:
    void rleToBbox(const RLE *R, BB bb, siz n)
    void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n)
    void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w)
    void rleFrCircle(RLE *R, const double *center_xy, double radius, siz h, siz w)
    void rleToUintBbox(const RLE *R, uint *bb)

cdef extern from "transpose_flip.h" nogil:
    void rleTranspose(const RLE *R, RLE *M)
    void rleVerticalFlip(const RLE *R, RLE *M)
    void rleRotate180Inplace(RLE *R)
    void rleRotate180(const RLE *R, RLE *M)
    void rleRoll(const RLE *R, RLE *M)

cdef extern from "iou_nms.h" nogil:
    void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o)
    void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o)

cdef extern from "largest_interior_rectangle.h" nogil:
    void rleLargestInteriorRectangle(const RLE *R, uint *rect)
    void rleLargestInteriorRectangleAspect(const RLE *R, double *rect, double aspect_ratio)
    void rleLargestInteriorRectangleAroundCenter(const RLE *R, double *rect, double cy, double cx,
                                                 double aspect_ratio)

cdef extern from "warp_affine.h" nogil:
    bool rleWarpAffine(const RLE *R, RLE *M, siz h_out, siz w_out, double *H)

cdef extern from "warp_perspective.h" nogil:
    bool rleWarpPerspective(const RLE *R, RLE *M, siz h_out, siz w_out, double *H)

cdef extern from "png_to_rle.h" nogil:
    bint rleFromPngBytes(RLE *R, const byte *data, siz length, int threshold, int channel)
    bint rleFromPngFile(RLE *R, const char *path, int threshold, int channel)
    siz rlesFromLabelMapPngBytes(RLE *Rs, const byte *data, siz length)
    siz rlesFromLabelMapPngFile(RLE *Rs, const char *path)

cdef extern from "warp_distorted.h" nogil:
    struct ValidRegion:
        float *ru
        float *tu
        float ru2_max
        float ru2_min
        float *rd
        float *td
        float rd2_max
        float rd2_min
        siz n

    struct Camera:
        double f[2]
        double c[2]
        double s
        double R[9]
        double d[12]
        double tilt[9]
        double tilt_inv[9]
        bool has_tilt
        ValidRegion valid

    bool rleWarpDistorted(
            const RLE *R, RLE *M, siz h_out, siz w_out, Camera *old_camera, Camera *new_camera)


ctypedef RLE *RLEPtr
ctypedef RLE**RLEPtrPtr
ctypedef const RLE *ConstRLEPtr
ctypedef const ConstRLEPtr *ConstRLEPtrConstPtr
ctypedef const RLEPtr *ConstRLEPtrPtr


def _pad_distortion_coeffs(d):
    d = np.ascontiguousarray(d, dtype=np.float64)
    if len(d) < 12:
        d = np.concatenate([d, np.zeros(12 - len(d), dtype=np.float64)])
    return d


def _tilt_matrices(d):
    # OpenCV-style sensor tilt homography of the 14-coefficient distortion model:
    # matTilt = matProjZ @ RotY(tau_y) @ RotX(tau_x), applied to normalized coordinates
    # after the 12-coefficient distortion
    if len(d) > 12 and (d[12] != 0 or d[13] != 0):
        cx, sx = np.cos(d[12]), np.sin(d[12])
        cy, sy = np.cos(d[13]), np.sin(d[13])
        tilt = np.array([
            [cx, 0, 0],
            [-sx * sy, cy, 0],
            [sy, -cy * sx, cx * cy]], dtype=np.float64)
        tilt_inv = np.linalg.inv(tilt)
        return (np.ascontiguousarray(tilt.reshape(-1)),
                np.ascontiguousarray(tilt_inv.reshape(-1)), True)
    identity = np.ascontiguousarray(np.eye(3, dtype=np.float64).reshape(-1))
    return identity, identity, False


def _check_shape_domain(h, w):
    if h < 0 or w < 0:
        raise ValueError(f"Mask dimensions must be non-negative, got height {h} and width {w}")
    if int(h) * int(w) > 0xFFFFFFFF:
        raise ValueError(
            f"Masks may have at most 2**32 - 1 pixels, got height {h} and width {w} "
            f"({int(h) * int(w)} pixels)")


@cython.boundscheck(False)
@cython.wraparound(False)
cdef bint _rle_counts_valid(RLE *r) noexcept nogil:
    cdef siz total = 0
    cdef siz i
    for i in range(r.m):
        total += r.cnts[i]
    return total == r.h * r.w


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _rle_decode_typed(RLE *r, floating[::1] out, floating fg_value) noexcept nogil:
    cdef siz i, j, pos = 0
    for i in range(r.m):
        if i % 2 == 1:
            for j in range(r.cnts[i]):
                out[pos + j] = fg_value
        pos += r.cnts[i]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _rle_decode_typed_multi(RLE *r, floating[::1] out, siz n_channels, floating[::1] fg_values) noexcept nogil:
    cdef siz i, j, c, pos = 0
    for i in range(r.m):
        if i % 2 == 1:
            for j in range(r.cnts[i]):
                for c in range(n_channels):
                    out[(pos + j) * n_channels + c] = fg_values[c]
        pos += r.cnts[i]


# python class to wrap RLE array in C
# the class handles the memory allocation and deallocation
cdef class RLECy:
    __slots__ = ["r"]
    cdef RLE r

    def __dealloc__(self):
        rleFree(&self.r)

    def _i_from_counts(self, shape: Sequence[int], counts: np.ndarray, order: str):
        _check_shape_domain(shape[0], shape[1])
        counts = np.ascontiguousarray(counts, dtype=np.uint32)
        cdef uint[::1] data = counts
        cdef RLE tmp
        if len(data) > 0:
            if order == 'F':
                rleFrCnts(&self.r, shape[0], shape[1], len(data), &data[0])
                # user-supplied counts may hold interior zero-length runs; normalize
                rleEliminateZeroRuns(&self.r)
            else:
                # own the counts and canonicalize before transposing: rleTranspose merges
                # internally and would overflow on non-canonical input (which a borrowed,
                # non-reallocatable view could not be normalized in place)
                rleFrCnts(&tmp, shape[1], shape[0], len(data), &data[0])
                rleEliminateZeroRuns(&tmp)
                rleTranspose(&tmp, &self.r)
                rleFree(&tmp)
        else:
            rleInit(&self.r, shape[0], shape[1], 0)

    def _i_from_array(self, mask: np.ndarray, int threshold=1, is_sparse: bool = True):
        cdef byte[::1, :] data
        arr = np.asanyarray(mask)
        _check_shape_domain(arr.shape[0], arr.shape[1])
        if arr.size > 0:
            if arr.dtype == np.bool_:
                arr = arr.view(np.uint8)
            if arr.dtype == np.uint8:
                if not 1 <= threshold <= 255:
                    raise ValueError(
                        f"threshold must be in [1, 255] for uint8 or bool input, "
                        f"got {threshold}")
            else:
                # non-uint8 input is thresholded in its native dtype (foreground is
                # value >= threshold); the resulting 0/1 array needs only the nonzero test
                arr = np.asfortranarray(arr >= threshold, dtype=np.uint8)
                threshold = 1

            if is_sparse and arr.flags.c_contiguous:
                # It's typically cheaper to do the transpose already in RLE
                data = arr.T
                tmp = RLECy()
                rleEncodeThreshold(&tmp.r, &data[0][0], mask.shape[1], mask.shape[0], 1, threshold)
                rleTranspose(&tmp.r, &self.r)
            else:
                data = np.asfortranarray(arr, dtype=np.uint8)
                rleEncodeThreshold(&self.r, &data[0][0], mask.shape[0], mask.shape[1], 1, threshold)
        else:
            rleInit(&self.r, mask.shape[0], mask.shape[1], 0)

    cpdef _i_from_dict(self, d: dict):
        cdef uint[::1] data
        size = d["size"]
        if size[0] != int(size[0]) or size[1] != int(size[1]):
            raise ValueError(f"Mask size must be integers, got {list(size)}")
        cdef siz h = int(size[0])
        cdef siz w = int(size[1])
        if h > 0xFFFFFFFF or w > 0xFFFFFFFF or h * w > <siz>0xFFFFFFFF:
            raise ValueError(
                f"Image dimensions {h}x{w} exceed maximum supported size "
                f"(h*w must fit in uint32)")
        if 'counts' in d:
            counts = d["counts"]
            if isinstance(counts, str):
                counts = counts.encode('utf-8')
            if not rleFrString(&self.r, <const char *> counts, h, w):
                raise ValueError(
                    "Invalid RLE string: sum of run lengths does not match h*w")
        elif 'ucounts' in d:
            ucounts = np.ascontiguousarray(d["ucounts"], dtype=np.uint32)
            if ucounts.sum() != h * w:
                raise ValueError(
                    f'Invalid RLE: Sum of runlengths is {ucounts.sum()}, which does not match '
                    f'the expected {h * w} based on the mask height {h} and width {w}')
            data = ucounts
            if len(data) > 0:
                rleFrCnts(&self.r, h, w, len(data), &data[0])
            else:
                rleInit(&self.r, h, w, 0)
        elif 'zcounts' in d:
            counts = zlib.decompress(d["zcounts"])
            if not rleFrString(&self.r, <const char *> counts, h, w):
                raise ValueError(
                    "Invalid RLE string: sum of run lengths does not match h*w")
        else:
            raise ValueError(
                "RLE dict must contain 'counts', 'ucounts', or 'zcounts' key")
        # Externally supplied counts may contain interior zero-length runs (valid sum, but
        # violating the canonical-form invariant that run-walking algorithms rely on).
        rleEliminateZeroRuns(&self.r)

    def _i_from_bbox(self, bbox, imshape):
        cdef np.ndarray[np.double_t, ndim=1] bbox_double = np.ascontiguousarray(
            bbox, dtype=np.float64)
        if bbox_double.shape[0] < 4:
            raise ValueError(
                f"Bounding box must have 4 elements (x, y, width, height), "
                f"got {bbox_double.shape[0]}")
        rleFrBbox(&self.r, <double *> bbox_double.data, imshape[0], imshape[1], 1)

    def _i_from_polygon(self, poly, imshape):
        cdef np.ndarray[np.double_t, ndim=1] np_poly = np.ascontiguousarray(poly, dtype=np.double)
        rleFrPoly(
            &self.r, <const double *> np_poly.data, int(len(poly) / 2), imshape[0], imshape[1])

    def _i_from_circle(self, center, radius, imshape):
        cdef np.ndarray[np.double_t, ndim=1] center_double = np.ascontiguousarray(
            center, dtype=np.float64)
        if center_double.shape[0] < 2:
            raise ValueError(
                f"Circle center must have 2 elements (x, y), got {center_double.shape[0]}")
        if not np.all(np.isfinite(center_double)) or not np.isfinite(radius):
            raise ValueError("Circle center and radius must be finite")
        rleFrCircle(&self.r, <double *> center_double.data, radius, imshape[0], imshape[1])

    def _i_from_png_file(self, str path, int threshold=1, int channel=-1):
        cdef bytes path_bytes = path.encode('utf-8')
        if not rleFromPngFile(&self.r, <const char *> path_bytes, threshold, channel):
            raise ValueError("Failed to read PNG (must be 8-bit, supported types: grayscale, gray+alpha, RGB, RGBA)")

    def _i_from_png_bytes(self, data, int threshold=1, int channel=-1):
        cdef const byte[::1] data_view = data
        if data_view.shape[0] == 0:
            raise ValueError("Empty PNG data")
        if not rleFromPngBytes(&self.r, &data_view[0], len(data_view), threshold, channel):
            raise ValueError("Failed to decode PNG (must be 8-bit, supported types: grayscale, gray+alpha, RGB, RGBA)")

    @staticmethod
    cdef RLECy _r_from_C_rle(RLE *rle, steal=False):
        rleCy = RLECy()
        if steal:
            rleMoveTo(rle, &rleCy.r)
        else:
            rleCopy(rle, &rleCy.r)
        return rleCy

    def _get_int_index(self, i, j):
        return int(rleGet(&self.r, i, j))

    def _i_set_int_index(self, i, j, v):
        rleSetInplace(&self.r, i, j, v)

    def _i_crop(self, start_h, start_w, span_h, span_w, step_h, step_w):
        span_w = max(0, min(span_w, self.r.w - start_w))
        span_h = max(0, min(span_h, self.r.h - start_h))
        cdef uint[4] box;
        box = [start_w, start_h, span_w, span_h]
        if box[3] != self.r.h or box[2] != self.r.w:
            rleCropInplace(&self.r, 1, box)
        if step_h != 1 or step_w != 1:
            rleStrideInplace(&self.r, step_h, step_w)

    cpdef RLECy _r_crop(self, start_h, start_w, span_h, span_w, step_h, step_w):
        span_w = max(0, min(span_w, self.r.w - start_w))
        span_h = max(0, min(span_h, self.r.h - start_h))
        cdef uint[4] box = [start_w, start_h, span_w, span_h]
        cdef RLECy result = RLECy()
        if box[3] != self.r.h or box[2] != self.r.w:
            rleCrop(&self.r, &result.r, 1, box)
        else:
            rleCopy(&self.r, &result.r)
        if step_h != 1 or step_w != 1:
            rleStrideInplace(&result.r, step_h, step_w)
        return result

    def _r_tight_crop(self):
        cdef RLECy result = RLECy()
        cdef uint[4] box;
        rleToUintBbox(&self.r, &box[0])
        rleCrop(&self.r, &result.r, 1, box)
        return result, np.array(box)

    def _i_tight_crop(self):
        cdef uint[4] box;
        rleToUintBbox(&self.r, &box[0])
        rleCropInplace(&self.r, 1, box)
        return np.array(box)

    def _r_transpose(self):
        cdef RLECy result = RLECy()
        rleTranspose(&self.r, &result.r)
        return result

    def _r_zeropad(self, left, right, top, bottom, v):
        cdef uint[4] np_pads = [left, right, top, bottom]
        cdef RLECy result = RLECy()
        if v == 0:
            rleZeroPad(&self.r, &result.r, 1, np_pads)
        else:
            rleComplement(&self.r, &result.r, 1)
            rleZeroPadInplace(&result.r, 1, np_pads)
            rleComplementInplace(&result.r, 1)
        return result

    def _i_zeropad(self, left, right, top, bottom, v):
        cdef uint[4] np_pads = [left, right, top, bottom]
        if v == 0:
            rleZeroPadInplace(&self.r, 1, np_pads)
        else:
            rleComplementInplace(&self.r, 1)
            rleZeroPadInplace(&self.r, 1, np_pads)
            rleComplementInplace(&self.r, 1)

    def _r_pad_replicate(self, left, right, top, bottom):
        cdef uint[4] np_pads = [left, right, top, bottom]
        cdef RLECy result = RLECy()
        rlePadReplicate(&self.r, &result.r, np_pads)
        return result

    def _i_repeat(self, nh, nw):
        rleRepeatInplace(&self.r, nh, nw)

    def _r_repeat(self, nh, nw):
        cdef RLECy result = RLECy()
        rleRepeat(&self.r, &result.r, nh, nw)
        return result

    def _r_diffor(self, other1: RLECy, other2: RLECy):
        cdef RLECy result = RLECy()
        rleMergeDiffOr(&self.r, &other1.r, &other2.r, &result.r)
        return result

    def _r_warp_affine(self, M: np.ndarray, h_out, w_out):
        _check_shape_domain(h_out, w_out)
        cdef RLECy result = RLECy()
        cdef double[::1] M_double = np.ascontiguousarray(M.reshape(-1), dtype=np.float64)
        if M_double.shape[0] < 6:
            raise ValueError(
                f"Affine matrix must have at least 6 elements, got {M_double.shape[0]}")
        if not rleWarpAffine(&self.r, &result.r, h_out, w_out, &M_double[0]):
            raise ValueError("Degenerate affine transformation matrix")
        return result

    def _r_warp_perspective(self, H: np.ndarray, h_out, w_out):
        _check_shape_domain(h_out, w_out)
        cdef RLECy result = RLECy()
        cdef double[::1] H_double = np.ascontiguousarray(H.reshape(-1), dtype=np.float64)
        if H_double.shape[0] < 9:
            raise ValueError(
                f"Perspective matrix must have at least 9 elements, got {H_double.shape[0]}")
        if not rleWarpPerspective(&self.r, &result.r, h_out, w_out, &H_double[0]):
            raise ValueError("Degenerate perspective transformation matrix")
        return result

    def _r_contours(self):
        cdef RLECy result = RLECy()
        rleContours(&self.r, &result.r)
        return result

    def largest_interior_rectangle(self):
        cdef np.ndarray[np.uint32_t, ndim=1] rect = np.empty(4, dtype=np.uint32)
        rleLargestInteriorRectangle(&self.r, &rect[0])
        return rect

    def largest_interior_rectangle_aspect(self, aspect_ratio: float):
        cdef np.ndarray[np.float64_t, ndim=1] rect = np.empty(4, dtype=np.float64)
        rleLargestInteriorRectangleAspect(&self.r, &rect[0], aspect_ratio)
        return rect

    def largest_interior_rectangle_around_center(self, cy, cx, aspect_ratio: float):
        cdef np.ndarray[np.float64_t, ndim=1] rect = np.empty(4, dtype=np.float64)
        rleLargestInteriorRectangleAroundCenter(&self.r, &rect[0], cy, cx, aspect_ratio)
        return rect

    @staticmethod
    cdef Camera _make_camera(R, K, d,
                              double[::1] tilt, double[::1] tilt_inv, bint has_tilt,
                              float[::1] ru, float[::1] tu,
                              float[::1] rd, float[::1] td):
        cdef Camera cam = Camera()
        for i in range(9):
            cam.R[i] = R.flat[i]
        cam.c[0] = K[0, 2]
        cam.c[1] = K[1, 2]
        cam.f[0] = K[0, 0]
        cam.f[1] = K[1, 1]
        cam.s = K[0, 1]
        for i in range(12):
            cam.d[i] = d[i]
        for i in range(9):
            cam.tilt[i] = tilt[i]
            cam.tilt_inv[i] = tilt_inv[i]
        cam.has_tilt = has_tilt

        cam.valid.ru = &ru[0]
        cam.valid.tu = &tu[0]
        cam.valid.ru2_max = np.square(np.asarray(ru).max())
        cam.valid.ru2_min = np.square(np.asarray(ru).min())
        cam.valid.rd = &rd[0]
        cam.valid.td = &td[0]
        cam.valid.rd2_max = np.square(np.asarray(rd).max())
        cam.valid.rd2_min = np.square(np.asarray(rd).min())
        cam.valid.n = ru.shape[0]
        return cam

    def _r_warp_distorted(
            self, R1, R2, K1, K2, d1, d2, polar_ud1, polar_ud2, h_out, w_out):
        # Allocate buffers here so they stay alive for the duration of rleWarpDistorted
        tilt1_arr, tilt1_inv_arr, has_tilt1 = _tilt_matrices(d1)
        tilt2_arr, tilt2_inv_arr, has_tilt2 = _tilt_matrices(d2)
        d1 = _pad_distortion_coeffs(d1)
        d2 = _pad_distortion_coeffs(d2)
        cdef double[::1] tilt1_buf = tilt1_arr
        cdef double[::1] tilt1_inv_buf = tilt1_inv_arr
        cdef double[::1] tilt2_buf = tilt2_arr
        cdef double[::1] tilt2_inv_buf = tilt2_inv_arr

        (ru1, tu1), (rd1, td1) = polar_ud1
        cdef float[::1] ru1_buf = np.ascontiguousarray(ru1, dtype=np.float32)
        cdef float[::1] tu1_buf = np.ascontiguousarray(tu1, dtype=np.float32)
        cdef float[::1] rd1_buf = np.ascontiguousarray(rd1, dtype=np.float32)
        cdef float[::1] td1_buf = np.ascontiguousarray(td1, dtype=np.float32)
        cdef Camera old_cam = RLECy._make_camera(
            R1, K1, d1, tilt1_buf, tilt1_inv_buf, has_tilt1,
            ru1_buf, tu1_buf, rd1_buf, td1_buf)

        (ru2, tu2), (rd2, td2) = polar_ud2
        cdef float[::1] ru2_buf = np.ascontiguousarray(ru2, dtype=np.float32)
        cdef float[::1] tu2_buf = np.ascontiguousarray(tu2, dtype=np.float32)
        cdef float[::1] rd2_buf = np.ascontiguousarray(rd2, dtype=np.float32)
        cdef float[::1] td2_buf = np.ascontiguousarray(td2, dtype=np.float32)
        cdef Camera new_cam = RLECy._make_camera(
            R2, K2, d2, tilt2_buf, tilt2_inv_buf, has_tilt2,
            ru2_buf, tu2_buf, rd2_buf, td2_buf)

        _check_shape_domain(h_out, w_out)
        cdef RLECy result = RLECy()
        if not rleWarpDistorted(&self.r, &result.r, h_out, w_out, &old_cam, &new_cam):
            raise ValueError("Degenerate camera transformation")
        return result

    def _r_avg_pool2x2(self):
        h = self.r.h
        w = self.r.w
        hr = h - h % 2
        wr = w - w % 2
        cdef RLECy rlemask0 = self._r_crop(0, 0, hr, wr, 2, 2)
        cdef RLECy rlemask1 = self._r_crop(0, 1, hr, wr, 2, 2)
        cdef RLECy rlemask2 = self._r_crop(1, 0, hr, wr, 2, 2)
        cdef RLECy rlemask3 = self._r_crop(1, 1, hr, wr, 2, 2)
        cdef ConstRLEPtr[4] rles = [&rlemask0.r, &rlemask1.r, &rlemask2.r, &rlemask3.r]
        cdef RLECy result = RLECy()
        rleMergeAtLeast2(rles, &result.r, 4, 2)
        return result

    @staticmethod
    def merge_many_multifunc(rles: Sequence[RLECy], boolfuncs: Iterable[int]):
        cdef siz n = len(rles)
        cdef const RLE **rles_ptr = <const RLE **> malloc(n * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef RLECy result
        cdef np.ndarray[np.uint32_t, ndim=1] bfs
        cdef uint *bfs_ptr = NULL
        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1

            bfs = np.ascontiguousarray(boolfuncs, dtype=np.uint32)
            if n > 1 and <siz> bfs.shape[0] != n - 1:
                raise ValueError(
                    f"Expected {n - 1} boolean functions for {n} masks, got {bfs.shape[0]}")
            if bfs.shape[0] > 0:
                bfs_ptr = &bfs[0]
            result = RLECy()
            rleMergeMultiFunc(rles_ptr, &result.r, n, bfs_ptr)
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def merge_many_singlefunc(rles: Sequence[RLECy], boolfunc: int):
        cdef siz n = len(rles)
        cdef const RLE **rles_ptr = <const RLE **> malloc(n * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef RLECy result
        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1

            result = RLECy()
            rleMergePtr(rles_ptr, &result.r, n, boolfunc)
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def merge_many_custom(rles: Sequence[RLECy], multiboolfunc: np.ndarray):
        if len(rles) > 32:
            raise ValueError(
                f"At most 32 masks can be merged with a custom boolean function, "
                f"got {len(rles)}")
        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef RLECy result
        cdef uint64_t[::1] mbf
        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1

            mbf = np.ascontiguousarray(multiboolfunc, dtype=np.uint64)
            if mbf.shape[0] == 0:
                raise ValueError("The truth table must not be empty")
            result = RLECy()
            rleMergeLookup(rles_ptr, &result.r, len(rles), &mbf[0], mbf.shape[0])
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def merge_many_weighted_atleast(rles: Sequence[RLECy], weights: np.ndarray, threshold: float):
        if len(weights) != len(rles):
            raise ValueError("The number of weights must be equal to the number of RLEs")

        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef RLECy result
        cdef double[::1] weights_double
        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1

            weights_double = np.ascontiguousarray(weights, dtype=np.float64)
            result = RLECy()
            rleMergeWeightedAtLeast2(rles_ptr, &result.r, len(rles), &weights_double[0],
                                     threshold)
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def merge_many_atleast(rles: Sequence[RLECy], threshold: int):
        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef RLECy result
        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1

            result = RLECy()
            rleMergeAtLeast2(rles_ptr, &result.r, len(rles), threshold)
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def concat_horizontal(rles: Sequence[RLECy]):
        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef RLECy result
        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1

            result = RLECy()
            rleConcatHorizontal(rles_ptr, &result.r, len(rles))
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def concat_vertical(rles: Sequence[RLECy]):
        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef RLECy result
        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1

            result = RLECy()
            rleConcatVertical(rles_ptr, &result.r, len(rles))
            return result
        finally:
            free(rles_ptr)

    def _r_conv2d_valid(self, kernel: np.ndarray, threshold: float, stride_h: int = 1,
                        stride_w: int = 1):
        kh, kw = kernel.shape[:2]
        k_area = kh * kw
        h = self.r.h
        w = self.r.w
        cys = [
            self._r_crop(
                i, j,
                max(0, h - kh + 1),
                max(0, w - kw + 1),
                stride_h, stride_w)
            for i in range(kh)
            for j in range(kw)
        ]

        cdef RLECy cy

        # assert same shape
        shape1 = cys[0].shape
        for cy in cys[1:]:
            if cy.shape != shape1:
                raise ValueError("All RLEs must have the same shape")

        cdef const RLE **rles_ptr = <const RLE **> malloc(k_area * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")
        cdef RLECy result
        cdef double[::1] weights = np.ascontiguousarray(kernel.reshape(-1), dtype=np.float64)
        try:
            for i, cy in enumerate(cys):
                rles_ptr[i] = &cy.r

            result = RLECy()
            rleMergeWeightedAtLeast2(rles_ptr, &result.r, k_area, &weights[0], threshold)
            return result
        finally:
            free(rles_ptr)

    def _r_avg_pool_valid(
            self, kernel_h: int, kernel_w: int, threshold: int = -1, stride_h: int = 1,
            stride_w: int = 1):

        kh, kw = kernel_h, kernel_w
        k_area = kh * kw
        if threshold == -1:
            threshold = k_area - (k_area // 2)
        h = self.r.h
        w = self.r.w
        cys = [
            self._r_crop(
                i, j,
                max(0, h - kh + 1),
                max(0, w - kw + 1),
                stride_h, stride_w)
            for i in range(kh)
            for j in range(kw)
        ]

        cdef RLECy cy

        # assert same shape
        shape1 = cys[0].shape
        for cy in cys[1:]:
            if cy.shape != shape1:
                raise ValueError("All RLEs must have the same shape")

        cdef const RLE **rles_ptr = <const RLE **> malloc(k_area * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")
        cdef RLECy result
        try:
            for i, cy in enumerate(cys):
                rles_ptr[i] = &cy.r

            result = RLECy()
            rleMergeAtLeast2(rles_ptr, &result.r, k_area, threshold)
            return result
        finally:
            free(rles_ptr)

    def _i_complement(self):
        rleComplementInplace(&self.r, 1)

    def _r_complement(self):
        cdef RLECy result = RLECy()
        rleComplement(&self.r, &result.r, 1)
        return result

    def _r_vertical_flip(self):
        cdef RLECy result = RLECy()
        rleVerticalFlip(&self.r, &result.r)
        return result

    def _r_boolfunc(self, other: RLECy, boolfunc: int):
        cdef RLECy result = RLECy()
        rleMerge2(&self.r, &other.r, &result.r, boolfunc & 0xffffffff)
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """Get the shape of the mask.

        Returns:
            A tuple of (height, width) of the mask.
        """
        return self.r.h, self.r.w

    @shape.setter
    def shape(self, new_shape: tuple[int, int]):
        """Set the shape of the mask without modifying the run-length data.

        This is a low-level power-user API. The caller is responsible for ensuring
        that the run-length counts still sum to new_h * new_w. Setting an incompatible
        shape will result in undefined behavior on subsequent operations.

        Args:
            new_shape: the new shape of the mask (height, width)
        """
        self.r.h = new_shape[0]
        self.r.w = new_shape[1]

    def _decode_into(self, arr: np.ndarray, value = 1):
        """Decode RLE into an existing array, only setting foreground pixels to `value`.

        Background pixels are left unchanged, allowing overlay of multiple masks.

        Args:
            arr: Target array. Can be:
                - 2D array (H, W): writes scalar value to foreground pixels
                - 3D array (H, W, C): writes to all channels of foreground pixels
            value: Value(s) to write. Can be:
                - scalar int: same value to all channels (broadcast)
                - tuple/list/array of length C: per-channel values
        """
        cdef byte[::1] data
        cdef byte[::1] values_arr
        cdef RLECy transp
        cdef bool success
        cdef siz num_channels

        if arr.ndim == 2:
            self._decode_into_2d(arr, value)
        elif arr.ndim == 3:
            if arr.shape[0] != self.r.h or arr.shape[1] != self.r.w:
                raise ValueError(
                    f"Array shape ({arr.shape[0]}, {arr.shape[1]}) does not match "
                    f"RLE shape ({self.r.h}, {self.r.w})")

            num_channels = arr.shape[2]
            if self.r.h == 0 or self.r.w == 0:
                return

            # Check if value is scalar (broadcast) or per-channel
            if isinstance(value, (int, np.integer)):
                # Broadcast: same value to all channels
                if arr.flags.c_contiguous:
                    transp = self._r_transpose()
                    data = arr.ravel(order='C')
                    success = rleDecodeBroadcast(&transp.r, &data[0], num_channels, value)
                elif arr.flags.f_contiguous:
                    # F-contiguous HWC: channels not interleaved, fall back to per-channel
                    for c in range(num_channels):
                        self._decode_into_2d(arr[:, :, c], value)
                    return
                else:
                    raise ValueError(
                        "3D array must be C-contiguous for efficient decode_into. "
                        "Use np.ascontiguousarray() first.")
            else:
                # Per-channel values
                values_arr = np.ascontiguousarray(value, dtype=np.uint8)
                if len(values_arr) != num_channels:
                    raise ValueError(
                        f"Value length ({len(values_arr)}) must match number of channels ({num_channels})")
                if arr.flags.c_contiguous:
                    transp = self._r_transpose()
                    data = arr.ravel(order='C')
                    success = rleDecodeMultiValue(&transp.r, &data[0], num_channels,
                                                  &values_arr[0])
                elif arr.flags.f_contiguous:
                    # F-contiguous HWC: channels not interleaved, fall back to per-channel
                    for c in range(num_channels):
                        self._decode_into_2d(arr[:, :, c], values_arr[c])
                    return
                else:
                    raise ValueError(
                        "3D array must be C-contiguous for efficient decode_into. "
                        "Use np.ascontiguousarray() first.")

            if not success:
                raise ValueError("Invalid RLE, sum of runlengths exceeds the number of pixels")
        else:
            raise ValueError(f"Array must be 2D or 3D, got {arr.ndim}D")

    def _decode_into_2d(self, arr: np.ndarray, value: int):
        """Decode RLE into a 2D array."""
        cdef byte[::1] data
        cdef RLECy transp
        cdef bool success
        cdef siz row_stride, col_stride

        if arr.shape[0] != self.r.h or arr.shape[1] != self.r.w:
            raise ValueError(
                f"Array shape ({arr.shape[0]}, {arr.shape[1]}) does not match RLE shape ({self.r.h}, {self.r.w})")

        if self.r.h > 0 and self.r.w > 0:
            if arr.flags.f_contiguous:
                data = arr.ravel(order='F')
                success = rleDecode(&self.r, &data[0], 1, value)
                if not success:
                    raise ValueError("Invalid RLE, sum of runlengths exceeds the number of pixels")
            elif arr.flags.c_contiguous:
                transp = self._r_transpose()
                data = arr.ravel(order='C')
                success = rleDecode(&transp.r, &data[0], 1, value)
                if not success:
                    raise ValueError("Invalid RLE, sum of runlengths exceeds the number of pixels")
            else:
                # Strided array (e.g., channel slice of HWC image)
                # Use strided C function - strides are in bytes, arr.itemsize is 1 for uint8
                if not _rle_counts_valid(&self.r):
                    raise ValueError(
                        "Invalid RLE: sum of runlengths does not match pixel count")
                row_stride = arr.strides[0]
                col_stride = arr.strides[1]
                rleDecodeStrided(&self.r, <byte *> np.PyArray_DATA(arr), row_stride, col_stride,
                                 value)

    def _r_to_dense_array(self, fg_value, bg_value, order, dtype=np.uint8) -> np.ndarray:
        cdef np.ndarray arr

        dtype = np.dtype(dtype)
        shape = (self.r.h, self.r.w)
        if self.r.h == 0 or self.r.w == 0:
            return np.full(shape, bg_value, dtype=dtype)

        # For C-order with sparse masks, allocate C directly to skip full transpose
        alloc_order = order if (order == 'F' or self.r.m < self.r.h * self.r.w * 0.04) else 'F'

        if bg_value == 0:
            arr = np.zeros(shape, dtype=dtype, order=alloc_order)
        else:
            arr = np.full(shape, bg_value, dtype=dtype, order=alloc_order)

        if dtype == np.uint8:
            self._decode_into(arr, fg_value)
        elif dtype == np.float32 or dtype == np.float64:
            self._decode_typed_into(arr, fg_value)
        else:
            mask01 = np.zeros(shape, dtype=np.uint8, order=alloc_order)
            self._decode_into(mask01, 1)
            arr[mask01 != 0] = fg_value

        if order == 'C' and not arr.flags.c_contiguous:
            return np.ascontiguousarray(arr)
        return arr

    def _decode_typed_into(self, np.ndarray arr, fg_value):
        """Decode RLE into a typed 2D array (float32/float64) using fused types."""
        cdef RLECy transp
        cdef float[::1] flat_f32
        cdef double[::1] flat_f64
        cdef RLE *rle_ptr

        if arr.shape[0] != self.r.h or arr.shape[1] != self.r.w:
            raise ValueError(
                f"Array shape ({arr.shape[0]}, {arr.shape[1]}) does not match RLE shape ({self.r.h}, {self.r.w})")

        if arr.dtype != np.float32 and arr.dtype != np.float64:
            raise ValueError(f"_decode_typed_into only supports float32 and float64, got {arr.dtype}")

        if arr.flags.f_contiguous:
            rle_ptr = &self.r
        elif arr.flags.c_contiguous:
            transp = self._r_transpose()
            rle_ptr = &transp.r
        else:
            raise ValueError("Array must be C-contiguous or F-contiguous")

        if not _rle_counts_valid(rle_ptr):
            raise ValueError("Invalid RLE: sum of runlengths does not match pixel count")

        order = 'F' if arr.flags.f_contiguous else 'C'
        if arr.dtype == np.float32:
            flat_f32 = arr.ravel(order=order)
            _rle_decode_typed(rle_ptr, flat_f32, <float>fg_value)
        else:
            flat_f64 = arr.ravel(order=order)
            _rle_decode_typed(rle_ptr, flat_f64, <double>fg_value)

    def _decode_typed_multi_into(self, np.ndarray arr, np.ndarray fg_values):
        """Decode RLE into a typed 3D HWC array (float32/float64) using fused types."""
        cdef RLECy transp
        cdef float[::1] flat_f32, fgv_f32
        cdef double[::1] flat_f64, fgv_f64
        cdef RLE *rle_ptr
        cdef siz n_channels = arr.shape[2]

        if arr.shape[0] != self.r.h or arr.shape[1] != self.r.w:
            raise ValueError(
                f"Array shape ({arr.shape[0]}, {arr.shape[1]}) does not match RLE shape ({self.r.h}, {self.r.w})")

        if fg_values.shape[0] != n_channels:
            raise ValueError(
                f"fg_values length ({fg_values.shape[0]}) does not match "
                f"number of channels ({n_channels})")

        if not arr.flags.c_contiguous:
            raise ValueError("3D array must be C-contiguous for typed multi-channel decode")

        transp = self._r_transpose()
        rle_ptr = &transp.r

        if not _rle_counts_valid(rle_ptr):
            raise ValueError("Invalid RLE: sum of runlengths does not match pixel count")

        if arr.dtype == np.float32:
            flat_f32 = arr.ravel(order='C')
            fgv_f32 = np.ascontiguousarray(fg_values, dtype=np.float32)
            _rle_decode_typed_multi(rle_ptr, flat_f32, n_channels, fgv_f32)
        else:
            flat_f64 = arr.ravel(order='C')
            fgv_f64 = np.ascontiguousarray(fg_values, dtype=np.float64)
            _rle_decode_typed_multi(rle_ptr, flat_f64, n_channels, fgv_f64)

    def _i_zeros(self, shape):
        _check_shape_domain(shape[0], shape[1])
        rleZeros(&self.r, shape[0], shape[1])

    def _i_ones(self, shape):
        _check_shape_domain(shape[0], shape[1])
        rleOnes(&self.r, shape[0], shape[1])

    def __eq__(self, other: RLECy) -> bool:
        return rleEqual(&self.r, &other.r) == 1

    cpdef np.ndarray _counts_view(self):
        """Return a writable NumPy view of the internal run-length counts.

        This is a low-level power-user API. The returned array directly aliases the
        internal RLE buffer. Modifying it will change the mask in place -- the caller
        is responsible for maintaining the invariant that counts sum to h*w.
        """
        cdef np.npy_intp shape[1]
        shape[0] = self.r.m
        arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, self.r.cnts)
        Py_INCREF(self)
        PyArray_SetBaseObject(arr, self)
        return arr

    def area(self) -> int:
        cdef uint a
        rleArea(&self.r, 1, &a)
        return int(a)

    def centroid(self) -> np.ndarray:
        cdef np.ndarray[np.double_t, ndim=1] xy = np.empty(2, dtype=np.double)
        rleCentroid(&self.r, &xy[0], 1)
        return xy

    def hu_moments(self) -> np.ndarray:
        cdef np.ndarray[np.double_t, ndim=1] hu = np.empty(7, dtype=np.double)
        rleHuMoments(&self.r, &hu[0])
        return hu

    def raw_moments(self) -> np.ndarray:
        cdef np.ndarray[np.double_t, ndim=1] moments = np.empty(10, dtype=np.double)
        rleRawMoments(&self.r, &moments[0])
        return moments

    def moments(self) -> np.ndarray:
        cdef np.ndarray[np.double_t, ndim=1] out = np.empty(24, dtype=np.double)
        rleMoments(&self.r, &out[0])
        return out

    def connected_components(self, connectivity: int = 4, min_size: int = 1):
        cdef RLE *components
        cdef siz n
        rleConnectedComponents(&self.r, connectivity, min_size, &components, &n)
        try:
            return [RLECy._r_from_C_rle(&components[i], steal=True) for i in range(n)]
        finally:
            rlesFree(&components, n)

    def connected_components_with_stats(self, filter_func=None, connectivity: int = 4,
                                        min_size: int = 1):
        """Extract connected components and their stats in a single pass.

        Optionally filter components using a filter function that receives stats.
        """
        cdef CCState *state = NULL
        cdef siz n_components = 0
        cdef siz *areas_ptr = NULL
        cdef int *bboxes_ptr = NULL
        cdef double *centroids_ptr = NULL
        cdef RLE *components = NULL
        cdef siz n_selected = 0
        cdef np.ndarray[np.uint8_t, ndim=1] selected_arr
        cdef np.npy_intp areas_shape[1]
        cdef np.npy_intp bboxes_shape[2]
        cdef np.npy_intp centroids_shape[2]

        try:
            state = rleConnectedComponentsBegin(
                &self.r, connectivity, min_size,
                &n_components, &areas_ptr, &bboxes_ptr, &centroids_ptr)

            if n_components == 0:
                return [], None

            # Convert C arrays to numpy (take ownership)
            areas_shape[0] = n_components
            areas = np.PyArray_SimpleNewFromData(1, areas_shape, np.NPY_UINT64, areas_ptr)
            PyArray_ENABLEFLAGS(areas, np.NPY_ARRAY_OWNDATA)
            areas_ptr = NULL

            bboxes_shape[0] = n_components
            bboxes_shape[1] = 4
            bboxes = np.PyArray_SimpleNewFromData(2, bboxes_shape, np.NPY_INT32, bboxes_ptr)
            PyArray_ENABLEFLAGS(bboxes, np.NPY_ARRAY_OWNDATA)
            bboxes_ptr = NULL

            centroids_shape[0] = n_components
            centroids_shape[1] = 2
            centroids = np.PyArray_SimpleNewFromData(2, centroids_shape, np.NPY_FLOAT64,
                                                     centroids_ptr)
            PyArray_ENABLEFLAGS(centroids, np.NPY_ARRAY_OWNDATA)
            centroids_ptr = NULL

            # Apply filter or select all
            if filter_func is not None:
                selected = filter_func(areas, bboxes, centroids)
                selected_arr = np.ascontiguousarray(selected, dtype=np.uint8)
                if <siz> selected_arr.shape[0] != n_components:
                    raise ValueError(
                        f"filter_func must return one value per component "
                        f"({n_components}), got {selected_arr.shape[0]}")
            else:
                selected_arr = np.ones(n_components, dtype=np.uint8)

            rleConnectedComponentsExtract(
                state, <bool *> &selected_arr[0], &components, &n_selected)

            result = [RLECy._r_from_C_rle(&components[i], steal=True) for i in range(n_selected)]

            # Filter stats to match selected components
            if filter_func is not None:
                mask = selected_arr.astype(np.bool_)
                areas = areas[mask]
                bboxes = bboxes[mask]
                centroids = centroids[mask]

            return result, (areas, bboxes, centroids)
        finally:
            if areas_ptr != NULL:
                free(areas_ptr)
            if bboxes_ptr != NULL:
                free(bboxes_ptr)
            if centroids_ptr != NULL:
                free(centroids_ptr)
            if components != NULL:
                rlesFree(&components, n_selected)
            if state != NULL:
                rleConnectedComponentsEnd(state)

    def connected_components_filtered(
            self, filter_func, connectivity: int = 4, min_size: int = 1):
        """Extract connected components with a filter function.

        The filter function receives three numpy arrays (areas, bboxes, centroids)
        and should return a boolean array indicating which components to extract.

        Args:
            filter_func: A callable that takes (areas, bboxes, centroids) and returns
                a boolean array. areas is shape (n,), bboxes is shape (n, 4) with
                columns (x, y, w, h), centroids is shape (n, 2) with columns (x, y).
            connectivity: 4 or 8 for neighborhood connectivity.
            min_size: Minimum component size to consider.

        Returns:
            A list of RLECy objects for the selected components.
        """
        cdef CCState *state = NULL
        cdef siz n_components = 0
        cdef siz *areas_ptr = NULL
        cdef int *bboxes_ptr = NULL
        cdef double *centroids_ptr = NULL
        cdef RLE *components = NULL
        cdef siz n_selected = 0
        cdef np.ndarray[np.uint8_t, ndim=1] selected_arr
        cdef np.npy_intp areas_shape[1]
        cdef np.npy_intp bboxes_shape[2]
        cdef np.npy_intp centroids_shape[2]

        try:
            # Phase 1: Get stats
            state = rleConnectedComponentsBegin(
                &self.r, connectivity, min_size,
                &n_components, &areas_ptr, &bboxes_ptr, &centroids_ptr)

            if n_components == 0:
                return []

            # Convert C arrays to numpy (take ownership)
            areas_shape[0] = n_components
            areas = np.PyArray_SimpleNewFromData(1, areas_shape, np.NPY_UINT64, areas_ptr)
            PyArray_ENABLEFLAGS(areas, np.NPY_ARRAY_OWNDATA)
            areas_ptr = NULL  # numpy now owns this

            bboxes_shape[0] = n_components
            bboxes_shape[1] = 4
            bboxes = np.PyArray_SimpleNewFromData(2, bboxes_shape, np.NPY_INT32, bboxes_ptr)
            PyArray_ENABLEFLAGS(bboxes, np.NPY_ARRAY_OWNDATA)
            bboxes_ptr = NULL

            centroids_shape[0] = n_components
            centroids_shape[1] = 2
            centroids = np.PyArray_SimpleNewFromData(2, centroids_shape, np.NPY_FLOAT64,
                                                     centroids_ptr)
            PyArray_ENABLEFLAGS(centroids, np.NPY_ARRAY_OWNDATA)
            centroids_ptr = NULL

            # Call user filter function
            selected = filter_func(areas, bboxes, centroids)
            selected_arr = np.ascontiguousarray(selected, dtype=np.uint8)
            if <siz> selected_arr.shape[0] != n_components:
                raise ValueError(
                    f"filter_func must return one value per component "
                    f"({n_components}), got {selected_arr.shape[0]}")

            # Phase 2: Extract selected components
            rleConnectedComponentsExtract(
                state, <bool *> &selected_arr[0], &components, &n_selected)

            result = [RLECy._r_from_C_rle(&components[i], steal=True) for i in range(n_selected)]
            return result
        finally:
            if areas_ptr != NULL:
                free(areas_ptr)
            if bboxes_ptr != NULL:
                free(bboxes_ptr)
            if centroids_ptr != NULL:
                free(centroids_ptr)
            if components != NULL:
                rlesFree(&components, n_selected)
            if state != NULL:
                rleConnectedComponentsEnd(state)

    def connected_component_stats(self, connectivity: int = 4, min_size: int = 1):
        """Get statistics for all connected components without extracting them.

        Returns:
            A tuple of (areas, bboxes, centroids) numpy arrays, or (None, None, None)
            if there are no components.
        """
        cdef siz n_components = 0
        cdef siz *areas_ptr = NULL
        cdef int *bboxes_ptr = NULL
        cdef double *centroids_ptr = NULL
        cdef np.npy_intp areas_shape[1]
        cdef np.npy_intp bboxes_shape[2]
        cdef np.npy_intp centroids_shape[2]

        n_components = rleConnectedComponentStats(
            &self.r, connectivity, min_size,
            &areas_ptr, &bboxes_ptr, &centroids_ptr)

        if n_components == 0:
            return None, None, None

        # Convert C arrays to numpy (take ownership)
        areas_shape[0] = n_components
        areas = np.PyArray_SimpleNewFromData(1, areas_shape, np.NPY_UINT64, areas_ptr)
        PyArray_ENABLEFLAGS(areas, np.NPY_ARRAY_OWNDATA)

        bboxes_shape[0] = n_components
        bboxes_shape[1] = 4
        bboxes = np.PyArray_SimpleNewFromData(2, bboxes_shape, np.NPY_INT32, bboxes_ptr)
        PyArray_ENABLEFLAGS(bboxes, np.NPY_ARRAY_OWNDATA)

        centroids_shape[0] = n_components
        centroids_shape[1] = 2
        centroids = np.PyArray_SimpleNewFromData(2, centroids_shape, np.NPY_FLOAT64, centroids_ptr)
        PyArray_ENABLEFLAGS(centroids, np.NPY_ARRAY_OWNDATA)

        return areas, bboxes, centroids

    def count_connected_components(self, connectivity: int = 4, min_size: int = 1) -> int:
        """Count connected components without extracting them."""
        return rleCountConnectedComponents(&self.r, connectivity, min_size)

    def bbox(self) -> np.ndarray:
        cdef np.ndarray[np.double_t, ndim=1] bb = np.empty(4, dtype=np.double)
        rleToBbox(&self.r, &bb[0], 1)
        return bb

    def nonzero_indices(self) -> np.ndarray:
        cdef uint *coords
        cdef siz n
        rleNonZeroIndices(&self.r, &coords, &n)
        if n == 0:
            free(coords)
            return np.empty((0, 2), dtype=np.uint32)
        cdef np.npy_intp shape[2]
        shape[0] = n // 2
        shape[1] = 2

        arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_UINT32, coords)
        PyArray_ENABLEFLAGS(arr, np.NPY_ARRAY_OWNDATA)
        return arr

    cpdef RLECy clone(self):
        return RLECy._r_from_C_rle(&self.r, steal=False)

    def _i_largest_connected_component(self, connectivity: int = 4):
        rleLargestConnectedComponentInplace(&self.r, connectivity)

    def _i_remove_small_components(self, min_size: int = 1, connectivity: int = 4):
        rleRemoveSmallConnectedComponentsInplace(&self.r, min_size, connectivity)

    def _i_rotate_180(self):
        rleRotate180Inplace(&self.r)

    def _r_rotate_180(self):
        cdef RLECy result = RLECy()
        rleRotate180(&self.r, &result.r)
        return result

    def _i_dilate_vertical(self, up=1, down=1):
        rleDilateVerticalInplace(&self.r, up, down)

    def _i_erode_vertical(self, up=1, down=1):
        rleComplementInplace(&self.r, 1)
        rleDilateVerticalInplace(&self.r, down, up)
        rleComplementInplace(&self.r, 1)

    cpdef to_dict(self, zlevel: Optional[int] = None):
        cdef char *c_string = rleToString(&self.r)
        if c_string == NULL:
            raise MemoryError("rleToString allocation failed")
        try:
            if zlevel is not None:
                compressed = zlib.compress(memoryview(c_string), zlevel)
                return {"size": [self.r.h, self.r.w], "zcounts": compressed}
            else:
                return {"size": [self.r.h, self.r.w], "counts": bytes(c_string)}
        finally:
            free(c_string)

    def iou(self, other: RLECy) -> float:
        cdef double o
        rleIou(&self.r, &other.r, 1, 1, NULL, &o)
        return o

    @staticmethod
    def iou_matrix(gt: Sequence[RLECy], dt: Sequence[RLECy]) -> np.ndarray:
        if len(dt) == 0 or len(gt) == 0:
            return np.zeros((len(gt), len(dt)), dtype=np.float64)

        cdef double[::1] o = np.empty(len(dt) * len(gt), dtype=np.float64)
        cdef RLE *dt_c = <RLE *> malloc(len(dt) * sizeof(RLE))
        cdef RLE *gt_c = <RLE *> malloc(len(gt) * sizeof(RLE))
        cdef RLECy rle

        if not dt_c or not gt_c:
            free(dt_c)
            free(gt_c)
            raise MemoryError("Failed to allocate memory for RLE pointers")

        try:
            for i, rle in enumerate(dt):
                rleBorrow(&dt_c[i], rle.r.h, rle.r.w, rle.r.m, rle.r.cnts)
            for i, rle in enumerate(gt):
                rleBorrow(&gt_c[i], rle.r.h, rle.r.w, rle.r.m, rle.r.cnts)

            rleIou(dt_c, gt_c, len(dt), len(gt), NULL, &o[0])
            return np.array(o).reshape(len(gt), len(dt))
        finally:
            free(dt_c)
            free(gt_c)

    @staticmethod
    def merge_to_label_map(rles: Sequence[RLECy]) -> np.ndarray:
        # this outputs an uint8 array like decode, but in each pixel the value
        # is the label from 1 to n, where n is the number of RLEs
        # and bg remains 0

        if len(rles) == 0:
            raise ValueError("Cannot create label map from empty sequence of RLEs")
        if len(rles) > 255:
            raise ValueError(
                f"At most 255 masks can be merged into a uint8 label map, got {len(rles)}")

        cdef RLECy rle
        cdef RLECy first = rles[0]
        cdef siz h = first.r.h
        cdef siz w = first.r.w
        for rle in rles:
            if rle.r.h != h or rle.r.w != w:
                raise ValueError(
                    "All masks must have the same shape to be merged into a label map")

        cdef np.ndarray[np.uint8_t, ndim=2, mode='fortran'] labelmap = np.zeros(
            (h, w), dtype=np.uint8, order='F')
        if h == 0 or w == 0:
            return labelmap

        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef siz i = 0
        try:
            for rle in rles:
                rles_ptr[i] = &rle.r
                i += 1
            rlesToLabelMapZeroInit(rles_ptr, &labelmap[0, 0], len(rles), h * w)
            return labelmap
        finally:
            free(rles_ptr)

    @staticmethod
    def from_label_map(label_map: np.ndarray):
        """Convert label map to list of RLEs.

        Label 0 is background, labels 1-255 become RLEs.
        Returns list of (label, RLECy) for non-empty labels.
        """
        label_map = np.asanyarray(label_map)
        if label_map.dtype != np.uint8 and label_map.size > 0:
            if label_map.min() < 0 or label_map.max() > 255:
                raise ValueError(
                    f"Label map values must be in [0, 255], got range "
                    f"[{label_map.min()}, {label_map.max()}]")
        cdef np.ndarray[np.uint8_t, ndim=2, mode='fortran'] lm = np.asfortranarray(
            label_map, dtype=np.uint8)
        cdef siz h = lm.shape[0]
        cdef siz w = lm.shape[1]

        # Allocate zero-initialized array of 255 RLEs (calloc ensures cnts/alloc are NULL)
        cdef RLE *Rs = <RLE *> calloc(255, sizeof(RLE))
        if not Rs:
            raise MemoryError("Failed to allocate memory for RLEs")

        cdef siz n_active
        try:
            n_active = rleFromLabelMap(&lm[0, 0], h, w, Rs)

            # Collect non-empty RLEs with their labels
            result = []
            for i in range(255):
                if Rs[i].cnts != NULL:  # Active label
                    result.append((i + 1, RLECy._r_from_C_rle(&Rs[i], steal=True)))
            return result
        finally:
            for i in range(255):
                rleFree(&Rs[i])
            free(Rs)

    @staticmethod
    def from_label_map_png_file(str path):
        """Convert PNG label map file directly to list of RLEs.

        Decodes PNG and builds RLEs in single pass.
        Label 0 is background, labels 1-255 become RLEs.
        Returns list of (label, RLECy) for non-empty labels.
        """
        cdef bytes path_bytes = path.encode('utf-8')
        cdef RLE *Rs = <RLE *> calloc(255, sizeof(RLE))
        if not Rs:
            raise MemoryError("Failed to allocate memory for RLEs")

        cdef siz n_active
        try:
            n_active = rlesFromLabelMapPngFile(Rs, <const char *> path_bytes)
            if n_active == <siz>-1:
                raise ValueError("Failed to read PNG (must be 8-bit grayscale)")

            result = []
            for i in range(255):
                if Rs[i].cnts != NULL:
                    result.append((i + 1, RLECy._r_from_C_rle(&Rs[i], steal=True)))
            return result
        finally:
            for i in range(255):
                rleFree(&Rs[i])
            free(Rs)

    @staticmethod
    def from_label_map_png_bytes(data):
        """Convert PNG label map bytes directly to list of RLEs.

        Decodes PNG and builds RLEs in single pass.
        Label 0 is background, labels 1-255 become RLEs.
        Returns list of (label, RLECy) for non-empty labels.
        """
        cdef const byte[::1] data_view = data
        cdef RLE *Rs = <RLE *> calloc(255, sizeof(RLE))
        if not Rs:
            raise MemoryError("Failed to allocate memory for RLEs")

        cdef siz n_active
        try:
            n_active = rlesFromLabelMapPngBytes(Rs, &data_view[0], len(data))
            if n_active == <siz>-1:
                raise ValueError("Failed to decode PNG (must be 8-bit grayscale)")

            result = []
            for i in range(255):
                if Rs[i].cnts != NULL:
                    result.append((i + 1, RLECy._r_from_C_rle(&Rs[i], steal=True)))
            return result
        finally:
            for i in range(255):
                rleFree(&Rs[i])
            free(Rs)

    def roll(self):
        cdef RLECy result = RLECy()
        rleRoll(&self.r, &result.r)
        return result

    @staticmethod
    def _unpack(d):
        cy = RLECy()
        cy._i_from_dict(d)
        return cy

    cpdef __reduce__(self):
        return RLECy._unpack, (self.to_dict(),)
