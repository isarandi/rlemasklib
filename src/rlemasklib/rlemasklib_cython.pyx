# cython: language_level=3

#**************************************************************************
# Based on code from the Microsoft COCO Toolbox.      version 2.0
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Modifications by Istvan Sarandi, 2023-2025
# Licensed under the Simplified BSD License [see coco/license.txt]
#**************************************************************************

# import both Python-level and C-level symbols of Numpy
# the API uses Numpy to interface C and Python
import zlib

import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, calloc
from libc.stdint cimport uint64_t, uint32_t, uint8_t

# initialize Numpy. must do.
np.import_array()

_INTERSECTION = 8
_UNION = 14

# import numpy C function
# we use PyArray_ENABLEFLAGS to make Numpy ndarray responsible to memory management
cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)

cdef extern from "stdbool.h":
    ctypedef int bool

# Declare the prototype of the C functions in rlemasklib.h
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
    void rlesFree(RLE **R, siz n)
    void rleFree(RLE *R)
    uint *rleFrCnts(RLE *R, siz h, siz w, siz m, uint *cnts)
    void rleBorrow(RLE *R, siz h, siz w, siz m, uint *cnts)

cdef extern from "encode_decode.h" nogil:
    void rleEncode(RLE *R, const byte *M, siz h, siz w, siz n)
    bool rleDecode(const RLE *R, byte *mask, siz n, byte value)
    char *rleToString(const RLE *R)
    bool rleFrString(RLE *R, const char *s, siz h, siz w)

cdef extern from "boolfuncs.h" nogil:
    void rleComplement(const RLE *R_in, RLE *R_out, siz n)
    void rleComplementInplace(RLE *R_in, siz n)
    void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc)

cdef extern from "moments.h" nogil:
    void rleArea(const RLE *R, siz n, uint *a)
    void rleCentroid(const RLE *R, double *xys, siz n)

cdef extern from "pad_crop.h" nogil:
    void rleCrop(const RLE *R_in, RLE *R_out, siz n, const uint *bbox)
    void rleCropInplace(RLE *R_in, siz n, const uint *bbox)
    void rleZeroPad(const RLE *R_in, RLE *R_out, siz n, const uint *pad_amounts)

cdef extern from "iou_nms.h" nogil:
    void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o)
    void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o)

cdef extern from "shapes.h" nogil:
    void rleToBbox(const RLE *R, BB bb, siz n)
    void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n)
    void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w)

cdef extern from "connected_components.h" nogil:
    void rleConnectedComponents(const RLE *R_in, int connectivity, siz min_size, RLE **components,
                                siz *n)
    void rleRemoveSmallConnectedComponentsInplace(RLE *R_in, siz min_size, int connectivity)

cdef extern from "transpose_flip.h" nogil:
    void rleTranspose(const RLE *R, RLE *M)


#
# def leb128_enc(np.ndarray[np.int32_t, ndim=1] cnts):
#     cdef char *encoded
#     cdef siz n_encoded
#     leb128_encode(<int *> cnts.data, cnts.shape[0], &encoded, &n_encoded)
#     cdef np.npy_intp shape[1]
#     shape[0] = <np.npy_intp> n_encoded
#     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, encoded)
#     PyArray_ENABLEFLAGS(a, np.NPY_ARRAY_OWNDATA)
#     return a
#
# def leb128_enc2(np.ndarray[np.int32_t, ndim=1] cnts):
#     cdef char *encoded
#     cdef siz n_encoded
#     leb128_encode2(<int *> cnts.data, cnts.shape[0], &encoded, &n_encoded)
#     cdef np.npy_intp shape[1]
#     shape[0] = <np.npy_intp> n_encoded
#     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, encoded)
#     PyArray_ENABLEFLAGS(a, np.NPY_ARRAY_OWNDATA)
#     return a


# python class to wrap RLE array in C
# the class handles the memory allocation and deallocation
cdef class RLEs:
    cdef RLE *_R
    cdef siz _n

    def __cinit__(self, siz n=0):
        if n > 0:
            rlesInit(&self._R, n)
        else:
            self._R = <RLE *> 0  # Don't allocate when n=0 to avoid leak when _R is overwritten
        self._n = n

    # free the RLE array here
    def __dealloc__(self):
        rlesFree(&self._R, self._n)
    def __getattr__(self, key):
        if key == 'n':
            return self._n
        raise AttributeError(key)

# python class to wrap Mask array in C
# the class handles the memory allocation and deallocation
cdef class Masks:
    cdef byte *_mask
    cdef siz _h
    cdef siz _w
    cdef siz _n
    cdef bint _owns_data

    def __cinit__(self, h, w, n):
        self._mask = <byte *> calloc(h * w * n, sizeof(byte))
        if self._mask == NULL and h * w * n > 0:
            raise MemoryError(f"Failed to allocate mask buffer of {h * w * n} bytes")
        self._h = h
        self._w = w
        self._n = n
        self._owns_data = True

    def __dealloc__(self):
        if self._owns_data and self._mask != NULL:
            free(self._mask)

    # return an np.ndarray in column-major order
    def to_array(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self._h * self._w * self._n
        # Create a 1D array, and reshape it to fortran/Matlab column-major array
        base = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._mask)
        # The _mask allocated by Masks is now handled by base
        PyArray_ENABLEFLAGS(base, np.NPY_ARRAY_OWNDATA)
        self._owns_data = False
        return base.reshape((self._h, self._w, self._n), order='F')

# internal conversion from Python RLEs object to compressed RLE format
def _to_leb128_dicts(RLEs Rs):
    cdef siz n = Rs.n
    cdef bytes py_string
    cdef char *c_string
    objs = []
    for i in range(n):
        c_string = rleToString(<RLE *> &Rs._R[i])
        if c_string == NULL:
            raise MemoryError("rleToString allocation failed")
        try:
            py_string = c_string
        finally:
            free(c_string)
        objs.append({
            'size': [Rs._R[i].h, Rs._R[i].w],
            'counts': py_string
        })
    return objs

def _to_uncompressed_dicts(RLEs Rs):
    cdef siz n = Rs.n
    cdef siz m
    cdef np.npy_intp shape[1]
    objs = []
    for i in range(n):
        shape[0] = <np.npy_intp> Rs._R[i].m
        ucounts = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, Rs._R[i].cnts)
        objs.append({'size': [Rs._R[i].h, Rs._R[i].w], 'ucounts': ucounts.copy()})
    return objs

def decompress(rleObjs):
    return _to_uncompressed_dicts(_from_leb128_dicts(rleObjs))

# internal conversion from RLE dicts (counts, zcounts or ucounts) to Python RLEs object
def _from_leb128_dicts(rleObjs):
    cdef siz n = len(rleObjs)
    Rs = RLEs(n)
    cdef bytes py_string
    cdef char *c_string
    cdef np.ndarray[np.uint32_t, ndim=1] ucounts
    cdef siz h, w
    for i, obj in enumerate(rleObjs):
        h = obj['size'][0]
        w = obj['size'][1]
        if h > 0xFFFFFFFF or w > 0xFFFFFFFF or h * w > 0xFFFFFFFF:
            raise ValueError(
                f'Masks may have at most 2**32 - 1 pixels, got height {h} and width {w}')
        if 'counts' in obj or 'zcounts' in obj:
            if 'counts' in obj:
                py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
            else:
                py_string = zlib.decompress(obj['zcounts'])
            c_string = py_string
            if not rleFrString(<RLE *> &Rs._R[i], <const  char *> c_string, h, w):
                raise ValueError(
                    "Invalid RLE string: sum of run lengths does not match h*w")
        elif 'ucounts' in obj:
            ucounts = np.ascontiguousarray(obj['ucounts'], dtype=np.uint32)
            if ucounts.sum() != h * w:
                raise ValueError(
                    f'Invalid RLE: Sum of runlengths is {ucounts.sum()}, which does not match the '
                    f'expected {h * w} based on the mask height {h} and width {w}')
            if len(ucounts) > 0:
                rleFrCnts(&Rs._R[i], h, w, len(ucounts), <uint *> &ucounts[0])
            else:
                rleFrCnts(&Rs._R[i], h, w, 0, NULL)
        else:
            raise ValueError("RLE dict must contain 'counts', 'zcounts' or 'ucounts'")

    return Rs

# encode mask to RLEs objects
# list of RLE string can be generated by RLEs member function
def encode(np.ndarray[np.uint8_t, ndim=3, mode='fortran'] mask, compress_leb128=True):
    h, w, n = mask.shape[0], mask.shape[1], mask.shape[2]
    cdef RLEs Rs = RLEs(n)
    rleEncode(Rs._R, <const byte *> mask.data, h, w, n)
    if compress_leb128:
        return _to_leb128_dicts(Rs)
    else:
        return _to_uncompressed_dicts(Rs)

def encode_C_order_sparse(
        np.ndarray[np.uint8_t, ndim=3, mode='c'] mask, compress_leb128=True):
    n, h, w = mask.shape[0], mask.shape[1], mask.shape[2]
    cdef RLEs Rs = RLEs(n)
    rleEncode(Rs._R, <const byte *> mask.data, w, h, n)

    cdef RLEs Rs_transp = RLEs(n)
    for i in range(n):
        rleTranspose(<RLE *> &Rs._R[i], <RLE *> &Rs_transp._R[i])

    if compress_leb128:
        return _to_leb128_dicts(Rs_transp)
    else:
        return _to_uncompressed_dicts(Rs_transp)

# decode mask from compressed list of RLE string or RLEs object
def decode(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    if Rs._n == 0:
        return np.empty((0, 0, 0), dtype=np.uint8)
    h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
    for i in range(1, n):
        if Rs._R[i].h != h or Rs._R[i].w != w:
            raise ValueError('All RLEs must have the same size to be decoded together')
    masks = Masks(h, w, n)
    cdef bool success = rleDecode(<RLE *> Rs._R, masks._mask, n, 1)
    if not success:
        raise ValueError('Invalid RLE: Run-lengths do not match the mask size')
    return masks.to_array()

def _from_uncompressed_dicts(rleObjs):
    cdef siz n = len(rleObjs)
    Rs = RLEs(n)
    cdef np.ndarray[np.uint32_t, ndim=1] counts
    cdef siz h, w
    for i, obj in enumerate(rleObjs):
        counts = np.ascontiguousarray(obj['ucounts'], dtype=np.uint32)
        h, w = obj['size'][0], obj['size'][1]
        if h > 0xFFFFFFFF or w > 0xFFFFFFFF or h * w > 0xFFFFFFFF:
            raise ValueError(
                f'Masks may have at most 2**32 - 1 pixels, got height {h} and width {w}')
        if counts.sum() != h * w:
            raise ValueError(
                f'Invalid RLE: Sum of runlengths is {counts.sum()}, which does not match the '
                f'expected {h * w} based on the mask height {h} and width {w}')
        rleFrCnts(&Rs._R[i], h, w, len(counts), <uint *> &counts[0])

    return Rs

def decodeUncompressed(ucRles):
    cdef RLEs Rs = _from_uncompressed_dicts(ucRles)
    if Rs._n == 0:
        return np.empty((0, 0, 0), dtype=np.uint8)
    h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
    for i in range(1, n):
        if Rs._R[i].h != h or Rs._R[i].w != w:
            raise ValueError('All RLEs must have the same size to be decoded together')
    masks = Masks(h, w, n)
    cdef bool success = rleDecode(<RLE *> Rs._R, masks._mask, n, 1)
    if not success:
        raise ValueError('Invalid RLE: Run-lengths do not match the mask size')
    return masks.to_array()

def merge(rleObjs, boolfunc=14):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    for i in range(1, Rs._n):
        if Rs._R[i].h != Rs._R[0].h or Rs._R[i].w != Rs._R[0].w:
            raise ValueError('All RLEs must have the same size to be merged')
    cdef RLEs R = RLEs(1)
    rleMerge(<RLE *> Rs._R, <RLE *> R._R, <siz> Rs._n, boolfunc & 0xffff)
    return _to_leb128_dicts(R)[0]

def area(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    cdef uint *_a = <uint *> malloc(Rs._n * sizeof(uint))
    if _a == NULL:
        raise MemoryError("Failed to allocate area buffer")
    rleArea(Rs._R, Rs._n, _a)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> Rs._n
    a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
    PyArray_ENABLEFLAGS(a, np.NPY_ARRAY_OWNDATA)
    return a

def crop(rleObjs, np.ndarray[np.uint32_t, ndim=2] bb):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    if bb.shape[0] != <np.npy_intp> Rs._n or bb.shape[1] != 4:
        raise ValueError(
            f'Expected a bounding box array of shape ({Rs._n}, 4), got '
            f'({bb.shape[0]}, {bb.shape[1]})')
    bb = np.ascontiguousarray(bb)
    rleCropInplace(Rs._R, Rs._n, <const uint *> bb.data)
    return _to_leb128_dicts(Rs)

def pad(rleObjs, np.ndarray[np.uint32_t, ndim=1] paddings):
    if paddings.shape[0] != 4:
        raise ValueError(f'Expected 4 padding amounts, got {paddings.shape[0]}')
    paddings = np.ascontiguousarray(paddings)
    cdef RLEs Rs_in = _from_leb128_dicts(rleObjs)
    cdef RLEs Rs_out = RLEs(Rs_in._n)
    rleZeroPad(Rs_in._R, Rs_out._R, Rs_in._n, <const uint *> paddings.data)
    return _to_leb128_dicts(Rs_out)

def complement(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    rleComplementInplace(Rs._R, Rs._n)
    return _to_leb128_dicts(Rs)

def iouMulti(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    for i in range(1, Rs._n):
        if Rs._R[i].h != Rs._R[0].h or Rs._R[i].w != Rs._R[0].w:
            raise ValueError('All RLEs must have the same size to compute their IoU')
    cdef RLEs Rs_merged = RLEs(1)  # intersection and union

    cdef uint intersection_area
    rleMerge(Rs._R, Rs_merged._R, Rs._n, _INTERSECTION)
    rleArea(Rs_merged._R, 1, &intersection_area)

    if intersection_area == 0:
        return 0

    cdef uint union_area
    rleFree(&Rs_merged._R[0])  # free before reusing
    rleMerge(Rs._R, Rs_merged._R, Rs._n, _UNION)
    rleArea(Rs_merged._R, 1, &union_area)

    return intersection_area / union_area

# iou computation. support function overload (RLEs-RLEs and bbox-bbox).
def iou(dt, gt, pyiscrowd):
    def _preproc(objs):
        if len(objs) == 0:
            return objs
        if type(objs) == np.ndarray:
            if len(objs.shape) == 1:
                objs = objs.reshape((1, -1))
            # check if it's Nx4 bbox
            if not len(objs.shape) == 2 or not objs.shape[1] == 4:
                raise TypeError(
                    'numpy ndarray input is only for *bounding boxes* and should have Nx4 dimension')
            objs = objs.astype(np.double)
        elif type(objs) == list:
            # check if list is in box format and convert it to np.ndarray
            isbox = np.all(
                np.array(
                    [(len(obj) == 4) and ((type(obj) == list) or (type(obj) == np.ndarray)) for obj
                     in objs]))
            isrle = np.all(np.array([type(obj) == dict for obj in objs]))
            if isbox:
                objs = np.array(objs, dtype=np.double)
                if len(objs.shape) == 1:
                    objs = objs.reshape((1, objs.shape[0]))
            elif isrle:
                objs = _from_leb128_dicts(objs)
            else:
                raise TypeError('list input can be bounding box (Nx4) or RLEs ([RLE])')
        else:
            raise TypeError(
                'unrecognized type.  The following type: RLEs (rle), np.ndarray (box), and list (box) are supported.')
        return objs
    def _rleIou(RLEs dt, RLEs gt, np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n,
                np.ndarray[np.double_t, ndim=1] _iou):
        rleIou(<RLE *> dt._R, <RLE *> gt._R, m, n, <byte *> iscrowd.data, <double *> _iou.data)
    def _bbIou(np.ndarray[np.double_t, ndim=2] dt, np.ndarray[np.double_t, ndim=2] gt,
               np.ndarray[np.uint8_t, ndim=1] iscrowd, siz m, siz n,
               np.ndarray[np.double_t, ndim=1] _iou):
        bbIou(<BB> dt.data, <BB> gt.data, m, n, <byte *> iscrowd.data, <double *> _iou.data)
    def _len(obj):
        cdef siz N = 0
        if type(obj) == RLEs:
            N = obj.n
        elif len(obj) == 0:
            pass
        elif type(obj) == np.ndarray:
            N = obj.shape[0]
        return N
    # convert iscrowd to numpy array
    cdef np.ndarray[np.uint8_t, ndim=1] iscrowd = np.array(pyiscrowd, dtype=np.uint8)
    # simple type checking
    cdef siz m, n
    dt = _preproc(dt)
    gt = _preproc(gt)
    m = _len(dt)
    n = _len(gt)
    if m == 0 or n == 0:
        return []
    if not type(dt) == type(gt):
        raise TypeError(
            'The dt and gt should have the same data type, either RLEs, list or np.ndarray')
    if iscrowd.shape[0] != <np.npy_intp> n:
        raise ValueError(
            f'iscrowd must have the same length as gt ({n}), got {iscrowd.shape[0]}')

    # define local variables
    cdef double *_iou = <double *> 0
    cdef np.npy_intp shape[1]
    # check type and assign iou function
    if type(dt) == RLEs:
        _iouFun = _rleIou
    elif type(dt) == np.ndarray:
        _iouFun = _bbIou
    else:
        raise TypeError('input data type not allowed.')
    _iou = <double *> malloc(m * n * sizeof(double))
    if _iou == NULL:
        raise MemoryError("Failed to allocate IoU buffer")
    shape[0] = <np.npy_intp> m * n
    iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
    PyArray_ENABLEFLAGS(iou, np.NPY_ARRAY_OWNDATA)
    _iouFun(dt, gt, iscrowd, m, n, iou)
    return iou.reshape((m, n), order='F')

def toBbox(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    cdef siz n = Rs.n
    cdef BB _bb = <BB> malloc(4 * n * sizeof(double))
    if _bb == NULL:
        raise MemoryError("Failed to allocate bounding box buffer")
    rleToBbox(<const RLE *> Rs._R, _bb, n)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 4 * n
    base = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb)
    PyArray_ENABLEFLAGS(base, np.NPY_ARRAY_OWNDATA)
    return base.reshape((n, 4))

def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w):
    cdef siz n = bb.shape[0]
    if bb.shape[1] != 4:
        raise ValueError(f'Expected a bounding box array of shape (n, 4), got (n, {bb.shape[1]})')
    bb = np.ascontiguousarray(bb)
    Rs = RLEs(n)
    rleFrBbox(<RLE *> Rs._R, <const BB> bb.data, h, w, n)
    objs = _to_leb128_dicts(Rs)
    return objs

def frPoly(poly, siz h, siz w):
    cdef np.ndarray[np.double_t, ndim=1] np_poly
    n = len(poly)
    Rs = RLEs(n)
    for i, p in enumerate(poly):
        np_poly = np.array(p, dtype=np.double, order='F')
        rleFrPoly(<RLE *> &Rs._R[i], <const  double *> np_poly.data, int(len(p) / 2), h, w)
    objs = _to_leb128_dicts(Rs)
    return objs

def frUncompressedRLE(ucRles):
    cdef np.ndarray[np.uint32_t, ndim=1] cnts
    cdef siz h, w
    n = len(ucRles)
    objs = []
    for i in range(n):
        Rs = RLEs(1)
        cnts = np.ascontiguousarray(ucRles[i]['ucounts'], dtype=np.uint32)
        h = ucRles[i]['size'][0]
        w = ucRles[i]['size'][1]
        if cnts.sum() != h * w:
            raise ValueError(
                f'Invalid RLE: Sum of runlengths is {cnts.sum()}, which does not match the '
                f'expected {h * w} based on the mask height {h} and width {w}')
        if h > 0xFFFFFFFF or w > 0xFFFFFFFF or h * w > 0xFFFFFFFF:
            raise ValueError(
                f'Masks may have at most 2**32 - 1 pixels, got height {h} and width {w}')
        if len(cnts) > 0:
            rleFrCnts(&Rs._R[0], h, w, len(cnts), <uint *> &cnts[0])
        else:
            rleFrCnts(&Rs._R[0], h, w, 0, NULL)
        objs.append(_to_leb128_dicts(Rs)[0])
    return objs

def frPyObjects(pyobj, h, w):
    # encode rle from a list of python objects
    if type(pyobj) == np.ndarray:
        objs = frBbox(pyobj, h, w)
    elif type(pyobj) == list and len(pyobj[0]) == 4:
        objs = frBbox(pyobj, h, w)
    elif type(pyobj) == list and len(pyobj[0]) > 4:
        objs = frPoly(pyobj, h, w)
    elif type(pyobj) == list and type(pyobj[0]) == dict \
            and 'counts' in pyobj[0] and 'size' in pyobj[0]:
        objs = frUncompressedRLE(pyobj)
    # encode rle from single python object
    elif type(pyobj) == list and len(pyobj) == 4:
        objs = frBbox([pyobj], h, w)[0]
    elif type(pyobj) == list and len(pyobj) > 4:
        objs = frPoly([pyobj], h, w)[0]
    elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
        objs = frUncompressedRLE([pyobj])[0]
    else:
        raise TypeError('input type is not supported.')
    return objs

def connectedComponents(rleObj, connectivity=4, min_size=1):
    cdef RLEs Rs = _from_leb128_dicts([rleObj])
    cdef RLEs Rs_out = RLEs(0)
    rleConnectedComponents(<RLE *> Rs._R, connectivity, min_size, &Rs_out._R, &Rs_out._n)
    return _to_leb128_dicts(Rs_out)

def removeSmallComponents(rleObj, min_size=1, connectivity=4):
    cdef RLEs Rs = _from_leb128_dicts([rleObj])
    rleRemoveSmallConnectedComponentsInplace(Rs._R, min_size, connectivity)
    return _to_leb128_dicts(Rs)[0]

def centroid(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    cdef siz n = Rs.n
    cdef double *_xys = <double *> malloc(2 * n * sizeof(double))
    if _xys == NULL:
        raise MemoryError("Failed to allocate centroid buffer")
    rleCentroid(<const RLE *> Rs._R, _xys, n)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 2 * n
    base = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _xys)
    PyArray_ENABLEFLAGS(base, np.NPY_ARRAY_OWNDATA)
    return base.reshape((n, 2))
