# cython: language_level=3

#**************************************************************************
# Based on code from the Microsoft COCO Toolbox.      version 2.0
# Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
# Modifications by Istvan Sarandi, 2023-2025
# Licensed under the Simplified BSD License [see coco/license.txt]
#**************************************************************************

# import both Python-level and C-level symbols of Numpy
# the API uses Numpy to interface C and Python
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free, calloc

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
    ctypedef unsigned int uint
    ctypedef unsigned long siz
    ctypedef unsigned char byte
    ctypedef double * BB
    ctypedef struct RLE:
        siz h,
        siz w,
        siz m,
        uint * cnts,
    void rlesInit(RLE ** R, siz n)
    void rlesFree(RLE ** R, siz n)

cdef extern from "encode_decode.h" nogil:
    void rleEncode(RLE *R, const byte *M, siz h, siz w, siz n)
    bool rleDecode(const RLE *R, byte *mask, siz n, byte value)
    char *rleToString(const RLE *R)
    void rleFrString(RLE *R, const char *s, siz h, siz w)

cdef extern from "boolfuncs.h" nogil:
    void rleComplement(const RLE *R_in, RLE *R_out, siz n)
    void rleComplementInplace(RLE *R_in, siz n)
    void rleMerge(const RLE *R, RLE *M, siz n, uint boolfunc)

cdef extern from "moments.h" nogil:
    void rleArea(const RLE *R, siz n, uint *a)
    void rleCentroid(const RLE *R, double *xys, siz n)

cdef extern from "pad_crop.h" nogil:
    void rleCrop(const RLE *R_in, RLE *R_out, siz n, const uint * bbox);
    void rleCropInplace(RLE *R_in, siz n, const uint * bbox);
    void rleZeroPad(const RLE *R_in, RLE *R_out, siz n, const uint * pad_amounts);

cdef extern from "iou_nms.h" nogil:
    void rleIou(RLE *dt, RLE *gt, siz m, siz n, byte *iscrowd, double *o)
    void bbIou(BB dt, BB gt, siz m, siz n, byte *iscrowd, double *o)

cdef extern from "shapes.h" nogil:
    void rleToBbox(const RLE *R, BB bb, siz n)
    void rleFrBbox(RLE *R, const BB bb, siz h, siz w, siz n)
    void rleFrPoly(RLE *R, const double *xy, siz k, siz h, siz w)

cdef extern from "connected_components.h" nogil:
    void rleConnectedComponents(const RLE *R_in, int connectivity, siz min_size, RLE ** components,
                                siz *n)

cdef extern from "transpose_flip.h" nogil:
    void rleTranspose(const RLE *R, RLE * M)


#
# def leb128_enc(np.ndarray[np.int32_t, ndim=1] cnts):
#     cdef char *encoded
#     cdef siz n_encoded
#     leb128_encode(<int *> cnts.data, cnts.shape[0], &encoded, &n_encoded)
#     cdef np.npy_intp shape[1]
#     shape[0] = <np.npy_intp> n_encoded
#     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, encoded)
#     PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
#     return a
#
# def leb128_enc2(np.ndarray[np.int32_t, ndim=1] cnts):
#     cdef char *encoded
#     cdef siz n_encoded
#     leb128_encode2(<int *> cnts.data, cnts.shape[0], &encoded, &n_encoded)
#     cdef np.npy_intp shape[1]
#     shape[0] = <np.npy_intp> n_encoded
#     a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, encoded)
#     PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
#     return a


# python class to wrap RLE array in C
# the class handles the memory allocation and deallocation
cdef class RLEs:
    cdef RLE *_R
    cdef siz _n

    def __cinit__(self, siz n =0):
        rlesInit(&self._R, n)
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

    def __cinit__(self, h, w, n):
        self._mask = <byte *> calloc(h * w * n, sizeof(byte))
        self._h = h
        self._w = w
        self._n = n
    # def __dealloc__(self):
    # the memory management of _mask has been passed to np.ndarray
    # it doesn't need to be freed here

    # return an np.ndarray in column-major order
    def to_array(self):
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self._h * self._w * self._n
        # Create a 1D array, and reshape it to fortran/Matlab column-major array
        ndarray = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT8, self._mask).reshape(
            (self._h, self._w, self._n), order='F')
        # The _mask allocated by Masks is now handled by ndarray
        PyArray_ENABLEFLAGS(ndarray, np.NPY_OWNDATA)
        return ndarray

# internal conversion from Python RLEs object to compressed RLE format
def _to_leb128_dicts(RLEs Rs):
    cdef siz n = Rs.n
    cdef bytes py_string
    cdef char *c_string
    objs = []
    for i in range(n):
        c_string = rleToString(<RLE *> &Rs._R[i])
        py_string = c_string
        objs.append({
            'size': [Rs._R[i].h, Rs._R[i].w],
            'counts': py_string
        })
        free(c_string)
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

# internal conversion from compressed RLE format to Python RLEs object
def _from_leb128_dicts(rleObjs):
    cdef siz n = len(rleObjs)
    Rs = RLEs(n)
    cdef bytes py_string
    cdef char * c_string
    cdef uint sum_counts
    for i, obj in enumerate(rleObjs):
        py_string = str.encode(obj['counts']) if type(obj['counts']) == str else obj['counts']
        c_string = py_string
        rleFrString(<RLE *> &Rs._R[i], <const char *> c_string, obj['size'][0], obj['size'][1])

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
    h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
    masks = Masks(h, w, n)
    cdef bool success = rleDecode(<RLE *> Rs._R, masks._mask, n, 1)
    if not success:
        raise ValueError('Invalid RLE: Run-lengths do not match the mask size')
    return masks.to_array()

def _from_uncompressed_dicts(rleObjs):
    cdef siz n = len(rleObjs)
    Rs = RLEs(n)
    cdef bytes py_string
    cdef char * c_string
    cdef uint sum_counts
    for i, obj in enumerate(rleObjs):
        counts = np.asarray(obj['ucounts'], dtype=np.uint32)
        Rs._R[i].cnts = <uint *> malloc(counts.shape[0] * sizeof(uint))

        data = <uint *> malloc(len(counts) * sizeof(uint))
        sum_counts = 0
        for j in range(len(counts)):
            data[j] = <uint> counts[j]
            sum_counts += data[j]
        Rs._R[i] = RLE(obj['size'][0], obj['size'][1], len(counts), <uint *> data)

        if sum_counts != Rs._R[i].h * Rs._R[i].w:
            raise ValueError(
                f'Invalid RLE: Sum of runlengths is {sum_counts}, which does not match the '
                f'expected {Rs._R[i].h * Rs._R[i].w} based on the mask height {Rs._R[i].h} and '
                f'width {Rs._R[i].w}')

    return Rs

def decodeUncompressed(ucRles):
    cdef RLEs Rs = _from_uncompressed_dicts(ucRles)
    h, w, n = Rs._R[0].h, Rs._R[0].w, Rs._n
    masks = Masks(h, w, n)
    cdef bool success = rleDecode(<RLE *> Rs._R, masks._mask, n, 1)
    if not success:
        raise ValueError('Invalid RLE: Run-lengths do not match the mask size')
    return masks.to_array()

def merge(rleObjs, boolfunc=14):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    cdef RLEs R = RLEs(1)
    rleMerge(<RLE *> Rs._R, <RLE *> R._R, <siz> Rs._n, boolfunc)
    return _to_leb128_dicts(R)[0]

def area(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    cdef uint * _a = <uint *> malloc(Rs._n * sizeof(uint))
    rleArea(Rs._R, Rs._n, _a)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> Rs._n
    a = np.PyArray_SimpleNewFromData(1, shape, np.NPY_UINT32, _a)
    PyArray_ENABLEFLAGS(a, np.NPY_OWNDATA)
    return a

def crop(rleObjs, np.ndarray[np.uint32_t, ndim=2] bb):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    rleCropInplace(Rs._R, Rs._n, <const uint *> bb.data)
    return _to_leb128_dicts(Rs)

def pad(rleObjs, np.ndarray[np.uint32_t, ndim=1] paddings):
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
    cdef RLEs Rs_merged = RLEs(1)  # intersection and union

    cdef uint intersection_area;
    rleMerge(Rs._R, Rs_merged._R, Rs._n, _INTERSECTION)
    rleArea(Rs_merged._R, 1, &intersection_area)

    if intersection_area == 0:
        return 0

    cdef uint union_area;
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
                objs = objs.reshape((objs[0], 1))
            # check if it's Nx4 bbox
            if not len(objs.shape) == 2 or not objs.shape[1] == 4:
                raise Exception(
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
                raise Exception('list input can be bounding box (Nx4) or RLEs ([RLE])')
        else:
            raise Exception(
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
        raise Exception(
            'The dt and gt should have the same data type, either RLEs, list or np.ndarray')

    # define local variables
    cdef double * _iou = <double *> 0
    cdef np.npy_intp shape[1]
    # check type and assign iou function
    if type(dt) == RLEs:
        _iouFun = _rleIou
    elif type(dt) == np.ndarray:
        _iouFun = _bbIou
    else:
        raise Exception('input data type not allowed.')
    _iou = <double *> malloc(m * n * sizeof(double))
    iou = np.zeros((m * n,), dtype=np.double)
    shape[0] = <np.npy_intp> m * n
    iou = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _iou)
    PyArray_ENABLEFLAGS(iou, np.NPY_OWNDATA)
    _iouFun(dt, gt, iscrowd, m, n, iou)
    return iou.reshape((m, n), order='F')

def toBbox(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    cdef siz n = Rs.n
    cdef BB _bb = <BB> malloc(4 * n * sizeof(double))
    rleToBbox(<const RLE *> Rs._R, _bb, n)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 4 * n
    bb = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _bb).reshape((n, 4))
    PyArray_ENABLEFLAGS(bb, np.NPY_OWNDATA)
    return bb

def frBbox(np.ndarray[np.double_t, ndim=2] bb, siz h, siz w):
    cdef siz n = bb.shape[0]
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
        rleFrPoly(<RLE *> &Rs._R[i], <const double *> np_poly.data, int(len(p) / 2), h, w)
    objs = _to_leb128_dicts(Rs)
    return objs

def frUncompressedRLE(ucRles):
    cdef np.ndarray[np.uint32_t, ndim=1] cnts
    cdef RLE R
    cdef uint *data
    n = len(ucRles)
    objs = []
    for i in range(n):
        Rs = RLEs(1)
        cnts = np.asarray(ucRles[i]['ucounts'], dtype=np.uint32)

        data = <uint *> malloc(len(cnts) * sizeof(uint))
        for j in range(len(cnts)):
            data[j] = <uint> cnts[j]
        R = RLE(ucRles[i]['size'][0], ucRles[i]['size'][1], len(cnts), data)
        Rs._R[0] = R
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
        raise Exception('input type is not supported.')
    return objs

def connectedComponents(rleObj, connectivity=4, min_size=1):
    cdef RLEs Rs = _from_leb128_dicts([rleObj])
    cdef RLEs Rs_out = RLEs(0)
    rleConnectedComponents(<RLE *> Rs._R, connectivity, min_size, &Rs_out._R, &Rs_out._n)
    return _to_leb128_dicts(Rs_out)

def centroid(rleObjs):
    cdef RLEs Rs = _from_leb128_dicts(rleObjs)
    cdef siz n = Rs.n
    cdef double * _xys = <double *> malloc(2 * n * sizeof(double))
    rleCentroid(<const RLE *> Rs._R, _xys, n)
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> 2 * n
    xys = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, _xys).reshape((n, 2))
    PyArray_ENABLEFLAGS(xys, np.NPY_OWNDATA)
    return xys
