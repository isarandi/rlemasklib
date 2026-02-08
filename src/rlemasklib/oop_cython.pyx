# cython: language_level=3
# distutils: language = c

import zlib
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, calloc
from libc.string cimport strlen, memcpy
from typing import Union, Optional
from collections.abc import Sequence, Iterable
from .boolfunc import BoolFunc
from libc.stdint cimport uint64_t, uint32_t, uint8_t
import struct

# intialized Numpy. must do.
np.import_array()

# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memoery management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void PyArray_CLEARFLAGS(np.ndarray arr, int flags)

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
    bool rleDecode(const RLE *R, byte *mask, siz n, byte value)
    bool rleDecodeStrided(const RLE *R, byte *M, siz row_stride, siz col_stride, byte value)
    bool rleDecodeBroadcast(const RLE *R, byte *M, siz num_channels, byte value)
    bool rleDecodeMultiValue(const RLE *R, byte *M, siz num_channels, const byte *values)
    char *rleToString(const RLE *R)
    void rleFrString(RLE *R, const char *s, siz h, siz w)
    void rlesToLabelMapZeroInit(const RLE **R, byte *label_map, siz n)


cdef extern from "boolfuncs.h" nogil:
    void rleComplement(const RLE *R_in, RLE *R_out, siz n)
    void rleComplementInplace(RLE *R_in, siz n)
    void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc)
    void rleMergePtr(const RLE **R, RLE *M, siz n, uint boolfunc)
    void rleMerge2(const RLE *A, const RLE *B, RLE *M, uint boolfunc)
    void rleMergeMultiFunc(const RLE **R, RLE *M, siz n, uint *boolfuncs)
    void rleMergeDiffOr(const RLE *A, const RLE *B, const RLE *C, RLE *M)
    void rleMergeAtLeast(const RLE **R, RLE *M, siz n, uint k)
    void rleMergeAtLeast2(const RLE **R, RLE *M, siz n, uint k)
    void rleMergeWeightedAtLeast(const RLE **R, RLE *M, siz n, double *weights, double threshold)
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
    void rleWarpAffine(const RLE *R, RLE *M, siz h_out, siz w_out, double *H)

cdef extern from "warp_perspective.h" nogil:
    void rleWarpPerspective(const RLE *R, RLE *M, siz h_out, siz w_out, double *H)

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
        ValidRegion valid

    void rleWarpDistorted(
            const RLE *R, RLE *M, siz h_out, siz w_out, Camera *old_camera, Camera *new_camera)


ctypedef RLE *RLEPtr
ctypedef RLE**RLEPtrPtr
ctypedef const RLE *ConstRLEPtr
ctypedef const ConstRLEPtr *ConstRLEPtrConstPtr
ctypedef const RLEPtr *ConstRLEPtrPtr

# python class to wrap RLE array in C
# the class handles the memory allocation and deallocation
cdef class RLECy:
    __slots__ = ["r"]
    cdef RLE r

    def __dealloc__(self):
        rleFree(&self.r)

    def _i_from_counts(self, shape: Sequence[int], counts: np.ndarray, order: str):
        counts = np.ascontiguousarray(counts, dtype=np.uint32)
        cdef uint[::1] data = counts
        cdef RLE tmp
        if len(data) > 0:
            if order == 'F':
                rleFrCnts(&self.r, shape[0], shape[1], len(data), &data[0])
            else:
                rleBorrow(&tmp, shape[1], shape[0], len(data), &data[0])
                rleTranspose(&tmp, &self.r)
        else:
            rleInit(&self.r, shape[0], shape[1], 0)

    def _i_from_array(self, mask: np.ndarray, thresh128: bool = False, is_sparse: bool = True):
        cdef byte[::1, :] data
        arr = np.asanyarray(mask)
        if arr.size > 0:
            if arr.dtype == np.bool_:
                arr = arr.view(np.uint8)
            elif arr.dtype != np.uint8:
                arr = np.asfortranarray(arr, dtype=np.uint8)

            if is_sparse and arr.flags.c_contiguous:
                # It's typically cheaper to do the transpose already in RLE
                data = arr.T
                tmp = RLECy()
                if thresh128:
                    rleEncodeThresh128(&tmp.r, &data[0][0], mask.shape[1], mask.shape[0], 1)
                else:
                    rleEncode(&tmp.r, &data[0][0], mask.shape[1], mask.shape[0], 1)
                rleTranspose(&tmp.r, &self.r)
            else:
                data = np.asfortranarray(arr, dtype=np.uint8)
                if thresh128:
                    rleEncodeThresh128(&self.r, &data[0][0], mask.shape[0], mask.shape[1], 1)
                else:
                    rleEncode(&self.r, &data[0][0], mask.shape[0], mask.shape[1], 1)
        else:
            rleInit(&self.r, mask.shape[0], mask.shape[1], 0)

    cpdef _i_from_dict(self, d: dict):
        cdef uint[::1] data
        if 'counts' in d:
            rleFrString(&self.r, <const char *> d["counts"], d["size"][0], d["size"][1])
        elif 'ucounts' in d:
            data = np.array(d["ucounts"], dtype=np.uint32)
            rleFrCnts(&self.r, d["size"][0], d["size"][1], len(d["ucounts"]), &data[0])
        elif 'zcounts' in d:
            counts = zlib.decompress(d["zcounts"])
            rleFrString(&self.r, <const char *> counts, d["size"][0], d["size"][1])

    def _i_from_bbox(self, bbox, imshape):
        cdef np.ndarray[np.double_t, ndim=1] bbox_double = np.ascontiguousarray(
            bbox, dtype=np.float64)
        rleFrBbox(&self.r, <double *> bbox_double.data, imshape[0], imshape[1], 1)

    def _i_from_polygon(self, poly, imshape):
        cdef np.ndarray[np.double_t, ndim=1] np_poly = np.ascontiguousarray(poly, dtype=np.double)
        rleFrPoly(
            &self.r, <const double *> np_poly.data, int(len(poly) / 2), imshape[0], imshape[1])

    def _i_from_circle(self, center, radius, imshape):
        cdef np.ndarray[np.double_t, ndim=1] center_double = np.ascontiguousarray(
            center, dtype=np.float64)
        rleFrCircle(&self.r, <double *> center_double.data, radius, imshape[0], imshape[1])

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
        cdef RLECy result = RLECy()
        cdef double[::1] M_double = np.ascontiguousarray(M.reshape(-1), dtype=np.float64)
        rleWarpAffine(&self.r, &result.r, h_out, w_out, &M_double[0])
        return result

    def _r_warp_perspective(self, H: np.ndarray, h_out, w_out):
        cdef RLECy result = RLECy()
        cdef double[::1] H_double = np.ascontiguousarray(H.reshape(-1), dtype=np.float64)
        rleWarpPerspective(&self.r, &result.r, h_out, w_out, &H_double[0])
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
    cdef Camera _make_camera(R, K, d, polar_ud):
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

        (ru, tu), (rd, td) = polar_ud
        cdef float[::1] ru_ = np.ascontiguousarray(ru, dtype=np.float32)
        cdef float[::1] tu_ = np.ascontiguousarray(tu, dtype=np.float32)
        cam.valid.ru = &ru_[0]
        cam.valid.tu = &tu_[0]
        cam.valid.ru2_max = np.square(np.max(ru))
        cam.valid.ru2_min = np.square(np.min(ru))
        cdef float[::1] rd_ = np.ascontiguousarray(rd, dtype=np.float32)
        cdef float[::1] td_ = np.ascontiguousarray(td, dtype=np.float32)
        cam.valid.rd = &rd_[0]
        cam.valid.td = &td_[0]
        cam.valid.rd2_max = np.square(np.max(rd))
        cam.valid.rd2_min = np.square(np.min(rd))
        cam.valid.n = len(ru)
        return cam

    def _r_warp_distorted(
            self, R1, R2, K1, K2, d1, d2, polar_ud1, polar_ud2, h_out, w_out):
        cdef Camera old_cam = RLECy._make_camera(R1, K1, d1, polar_ud1)
        cdef Camera new_cam = RLECy._make_camera(R2, K2, d2, polar_ud2)
        cdef RLECy result = RLECy()
        rleWarpDistorted(&self.r, &result.r, h_out, w_out, &old_cam, &new_cam)
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

    # @staticmethod
    # def merge_many_multifunc(rles: Sequence[RLECy], boolfuncs: Sequence[int]):
    #     cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE*))
    #     if not rles_ptr:
    #         raise MemoryError("Failed to allocate memory for RLE pointers")
    #
    #     cdef RLECy result
    #     cdef uint[::1] bfs
    #
    #     try:
    #         for i, rle in enumerate(rles):
    #             rles_ptr[i] = &(<RLECy> rle).r
    #
    #         result = RLECy()
    #         bfs = np.ascontiguousarray(boolfuncs, dtype=np.uint32)
    #         rleMergeMultiFunc(rles_ptr, &result.r, len(rles), &bfs[0])
    #         return result
    #     finally:
    #         free(rles_ptr)

    @staticmethod
    def merge_many_multifunc(rles: Sequence[RLECy], boolfuncs: Iterable[int]):
        cdef siz n = len(rles)
        cdef const RLE **rles_ptr = <const RLE **> malloc(n * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy rle
        cdef siz i = 0
        for rle in rles:
            rles_ptr[i] = &rle.r
            i += 1

        cdef np.ndarray[np.uint32_t, ndim=1] bfs = np.ascontiguousarray(boolfuncs, dtype=np.uint32)
        cdef RLECy result = RLECy()
        try:
            rleMergeMultiFunc(rles_ptr, &result.r, n, &bfs[0])
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
        cdef siz i = 0
        for rle in rles:
            rles_ptr[i] = &rle.r
            i += 1

        cdef RLECy result = RLECy()
        try:
            rleMergePtr(rles_ptr, &result.r, n, boolfunc)
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def merge_many_custom(rles: Sequence[RLECy], multiboolfunc: np.ndarray):
        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy result
        cdef uint64_t[::1] mbf
        try:
            for i, rle in enumerate(rles):
                rles_ptr[i] = &(<RLECy> rle).r

            mbf = np.ascontiguousarray(multiboolfunc, dtype=np.uint64)
            result = RLECy()
            rleMergeLookup(rles_ptr, &result.r, len(rles), &mbf[0], mbf.shape[0])
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def merge_many_weighted_atleast(rles: Sequence[RLECy], weights: np.ndarray, threshold: float):
        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        if len(weights) != len(rles):
            raise ValueError("The number of weights must be equal to the number of RLEs")

        cdef RLECy result
        cdef double[::1] weights_double
        try:
            for i in range(len(rles)):
                rles_ptr[i] = &(<RLECy> rles[i]).r

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

        cdef RLECy result
        try:
            for i in range(len(rles)):
                rles_ptr[i] = &(<RLECy> rles[i]).r

            result = RLECy()
            rleMergeAtLeast(rles_ptr, &result.r, len(rles), threshold)
            return result
        finally:
            free(rles_ptr)

    @staticmethod
    def concat_horizontal(rles: Sequence[RLECy]):
        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef RLECy result
        try:
            for i in range(len(rles)):
                rles_ptr[i] = &(<RLECy> rles[i]).r

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

        cdef RLECy result
        try:
            for i in range(len(rles)):
                rles_ptr[i] = &(<RLECy> rles[i]).r

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
        """Set the shape of the mask.

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
                row_stride = arr.strides[0]
                col_stride = arr.strides[1]
                rleDecodeStrided(&self.r, <byte *> np.PyArray_DATA(arr), row_stride, col_stride,
                                 value)

    def _r_to_dense_array(self, value, order) -> np.ndarray:
        if self.r.h == 0 or self.r.w == 0:
            return np.empty((self.r.h, self.r.w), dtype=np.uint8)

        if order == 'F':
            arr = np.zeros((self.r.h, self.r.w), dtype=np.uint8, order='F')
        else:
            is_sparse = self.r.m < self.r.h * self.r.w * 0.04
            if is_sparse:
                arr = np.zeros((self.r.h, self.r.w), dtype=np.uint8, order='C')
            else:
                arr = np.zeros((self.r.h, self.r.w), dtype=np.uint8, order='F')

        self._decode_into(arr, value)

        if order == 'C' and arr.flags.f_contiguous:
            return np.ascontiguousarray(arr)
        return arr

    def _i_zeros(self, shape):
        rleZeros(&self.r, shape[0], shape[1])

    def _i_ones(self, shape):
        rleOnes(&self.r, shape[0], shape[1])

    def __eq__(self, other: RLECy) -> bool:
        return rleEqual(&self.r, &other.r) == 1

    cpdef np.ndarray _counts_view(self):
        cdef np.npy_intp shape[1]
        shape[0] = self.r.m
        return np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, self.r.cnts)

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
        cdef double[::1] o = np.empty(len(dt) * len(gt), dtype=np.float64)
        cdef RLE *dt_c = <RLE *> malloc(len(dt) * sizeof(RLE))
        cdef RLE *gt_c = <RLE *> malloc(len(gt) * sizeof(RLE))
        cdef RLECy rle

        if len(dt) == 0 or len(gt) == 0:
            return np.zeros((len(gt), len(dt)), dtype=np.float64)

        if not dt_c or not gt_c:
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

        cdef const RLE **rles_ptr = <const RLE **> malloc(len(rles) * sizeof(RLE *))
        if not rles_ptr:
            raise MemoryError("Failed to allocate memory for RLE pointers")

        cdef siz i
        for i in range(len(rles)):
            rles_ptr[i] = &(<RLECy> rles[i]).r
        # prepare the zeros np array:
        cdef np.ndarray[np.uint8_t, ndim=2, mode='fortran'] labelmap = np.zeros(
            rles[0].shape, dtype=np.uint8, order='F')
        rlesToLabelMapZeroInit(rles_ptr, &labelmap[0, 0], len(rles))
        free(rles_ptr)
        return labelmap

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
