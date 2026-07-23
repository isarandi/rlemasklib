"""Regression tests for validation, boundary and correctness fixes.

Each test corresponds to a formerly incorrect behavior: infinite loops or process aborts on
malformed or mismatched inputs, wrong results for Boolean functions that are true on
all-background input, off-by-one and overflow errors in geometric operations, and missing
input validation at the API boundaries.
"""

import itertools

import numpy as np
import pytest
import scipy.ndimage

import rlemasklib
from rlemasklib import BoolFunc, RLEMask


def encode_arr(a):
    return rlemasklib.encode(np.asfortranarray(a, dtype=np.uint8))


class TestShapeMismatchValidation:
    """Mismatched or malformed inputs used to hang the merge loops forever."""

    def test_iou_shape_mismatch(self):
        with pytest.raises(ValueError):
            rlemasklib.iou([encode_arr(np.ones((3, 3))), encode_arr(np.ones((5, 7)))])

    def test_merge_count_shape_mismatch(self):
        with pytest.raises(ValueError):
            RLEMask.merge_count([RLEMask.ones((2, 2)), RLEMask.ones((3, 3))], 1)

    def test_setitem_shape_mismatch(self):
        m = RLEMask.zeros((4, 4))
        with pytest.raises(ValueError):
            m[0:2, 0:2] = RLEMask.ones((3, 3))

    def test_from_dict_bad_ucounts_sum(self):
        with pytest.raises(ValueError):
            RLEMask.from_dict({'size': [2, 2], 'ucounts': [1, 1]})

    def test_functional_merge_shape_mismatch(self):
        with pytest.raises(ValueError):
            rlemasklib.union([encode_arr(np.ones((2, 2))), encode_arr(np.ones((3, 3)))])

    def test_decode_mixed_sizes(self):
        with pytest.raises(ValueError):
            rlemasklib.decode([encode_arr(np.ones((3, 3))), encode_arr(np.ones((2, 2)))])

    def test_merge_to_label_map_shape_mismatch(self):
        with pytest.raises(ValueError):
            RLEMask.merge_to_label_map([RLEMask.ones((2, 2)), RLEMask.ones((100, 100))])

    def test_merge_to_label_map_too_many(self):
        with pytest.raises(ValueError):
            RLEMask.merge_to_label_map([RLEMask.ones((2, 2))] * 256)

    def test_iou_oop_shape_mismatch(self):
        with pytest.raises(ValueError):
            RLEMask.ones((2, 2)).iou(RLEMask.ones((3, 3)))
        with pytest.raises(ValueError):
            RLEMask.iou_matrix([RLEMask.ones((2, 2))], [RLEMask.ones((3, 3))])


class TestEmptyInputValidation:
    def test_merge_empty(self):
        with pytest.raises(ValueError):
            rlemasklib.merge([], BoolFunc.UNION)

    def test_union_empty(self):
        with pytest.raises(ValueError):
            rlemasklib.union([])

    def test_iou_empty(self):
        with pytest.raises(ValueError):
            rlemasklib.iou([])

    def test_merge_count_empty(self):
        with pytest.raises(ValueError):
            RLEMask.merge_count([], 1)

    def test_empty_counts_string(self):
        with pytest.raises(ValueError):
            RLEMask({'size': [2, 2], 'counts': ''})
        with pytest.raises(ValueError):
            rlemasklib.decompress({'size': [2, 2], 'counts': b''})

    def test_empty_ucounts(self):
        with pytest.raises(ValueError):
            RLEMask.from_dict({'size': [2, 2], 'ucounts': []})

    def test_empty_png_bytes(self):
        with pytest.raises(ValueError):
            RLEMask.from_png(data=b'')


class TestBoolFuncTruthTables:
    """Functions that are true on all-background input used to produce wrong masks."""

    def test_all_16_truth_tables_exhaustive(self):
        def all_masks(h, w):
            for bits in range(2 ** (h * w)):
                flat = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=np.uint8)
                yield flat.reshape((h, w), order='F')

        masks = list(all_masks(2, 2))
        rles = [RLEMask.from_array(m) for m in masks]
        for bf in range(16):
            for (ma, ra), (mb, rb) in itertools.product(zip(masks, rles), repeat=2):
                idx = (ma.astype(np.uint8) << 1) | mb
                ref = ((bf >> idx) & 1).astype(np.uint8)
                got = ra.merge(rb, BoolFunc(bf) if bf in BoolFunc._value2member_map_ else bf)
                assert np.array_equal(np.array(got), ref), f'boolfunc {bf}'

    def test_nor_of_zeros(self):
        z = RLEMask.zeros((2, 2))
        assert z.merge(z, BoolFunc.NOR).area() == 4

    def test_f00_true_members(self):
        a = RLEMask.zeros((2, 2))
        a[0, 1] = 1
        z = RLEMask.zeros((2, 2))
        assert a.merge(z, BoolFunc.NOR) == ~a
        assert a.merge(z, BoolFunc.NAND).area() == 4
        assert a.merge(z, BoolFunc.XNOR) == ~a
        assert a.merge(z, BoolFunc.IMPLICATION) == ~a

    def test_functional_nor(self):
        d = encode_arr(np.zeros((2, 2)))
        assert rlemasklib.decode(rlemasklib.merge([d, d], BoolFunc.NOR)).sum() == 4

    def test_merge_count_threshold_zero(self):
        a = RLEMask.zeros((2, 2))
        a[0, 1] = 1
        assert RLEMask.merge_count([a, RLEMask.zeros((2, 2))], 0).area() == 4

    def test_conv2d_threshold_zero(self):
        a = RLEMask.zeros((2, 2))
        a[0, 1] = 1
        assert a.conv2d_valid(np.ones((2, 1)), threshold=0.0).area() == 2

    def test_boolfunc_values_non_negative(self):
        assert all(bf.value >= 0 for bf in BoolFunc)

    def test_merge_many_negative_origin_members(self):
        m1 = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        m2 = RLEMask.from_array(np.eye(3, dtype=np.uint8)[::-1])
        for bf in (BoolFunc.EQUIVALENCE, BoolFunc.NAND, BoolFunc.NOR, BoolFunc.IMPLICATION):
            assert RLEMask.merge_many([m1, m2], bf) == m1.merge(m2, bf)

    def test_merge_many_plain_int_func(self):
        m1 = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        m2 = RLEMask.from_array(np.eye(3, dtype=np.uint8)[::-1])
        func = ~BoolFunc.A & BoolFunc.B
        assert RLEMask.merge_many([m1, m2], func) == m1.merge(m2, func)


class TestCustomMergeFunctions:
    def test_arity7_all_false_on_ones(self):
        f = RLEMask.make_merge_function(lambda a, b, c, d, e, f_, g: not (a or b or c or d or e or f_ or g))
        assert f(*[RLEMask.ones((3, 3))] * 7).area() == 0

    def test_always_false(self):
        f = RLEMask.make_merge_function(lambda a, b: False)
        assert f(RLEMask.ones((2, 2)), RLEMask.ones((2, 2))).area() == 0

    def test_unary_not_is_complement(self):
        f = RLEMask.make_merge_function(lambda a: not a)
        eye = RLEMask.from_array(np.eye(4, dtype=np.uint8))
        assert f(eye) == ~eye

    def test_arity_cap(self):
        with pytest.raises(ValueError):
            RLEMask.make_merge_function(lambda *a: True, arity=33)

    def test_random_ternary_tables(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            table = rng.integers(0, 2, 8)

            def func(a, b, c, table=table):
                return bool(table[(int(c) << 2) | (int(b) << 1) | int(a)])

            arrs = [rng.integers(0, 2, (5, 7)).astype(np.uint8) for _ in range(3)]
            rles = [RLEMask.from_array(x) for x in arrs]
            got = RLEMask.merge_many_custom(rles, func)
            idx = (arrs[2].astype(int) << 2) | (arrs[1].astype(int) << 1) | arrs[0]
            ref = table[idx].astype(np.uint8)
            assert np.array_equal(np.array(got), ref)


class TestSingleMaskMergeVariants:
    """n==1 shortcuts used to ignore the predicate entirely."""

    def test_merge_count_single_mask(self):
        m = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        assert RLEMask.merge_count([m], 0).area() == 9
        assert RLEMask.merge_count([m], 1) == m
        assert RLEMask.merge_count([m], 2).area() == 0

    def test_conv2d_1x1_kernel(self):
        m = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        assert m.conv2d_valid(np.array([[0.2]]), threshold=0.5).area() == 0
        assert m.conv2d_valid(np.array([[-1.0]]), threshold=-0.5) == ~m

    def test_merge_count_over_32_masks(self):
        m = RLEMask.from_array(np.eye(3, dtype=np.uint8))
        assert RLEMask.merge_count([m] * 33, 17) == m


class TestWarpRotations:
    """The flip decision was wrong for 90-degree rotations of non-square masks."""

    @staticmethod
    def rot_mat(k, h, w):
        # exact grid rotation by k*90 degrees CCW mapping (h, w) to output shape
        if k % 4 == 1:
            return np.array([[0., 1, 0], [-1, 0, w - 1]]), (w, h)
        elif k % 4 == 2:
            return np.array([[-1., 0, w - 1], [0, -1, h - 1]]), (h, w)
        elif k % 4 == 3:
            return np.array([[0., -1, h - 1], [1, 0, 0]]), (w, h)
        return np.array([[1., 0, 0], [0, 1, 0]]), (h, w)

    @pytest.mark.parametrize('shape', [(40, 100), (100, 40), (7, 31), (31, 7), (50, 50)])
    @pytest.mark.parametrize('k', [1, 2, 3])
    def test_rot90_exact(self, shape, k):
        rng = np.random.default_rng(hash(shape + (k,)) % 2 ** 32)
        arr = (rng.random(shape) < 0.5).astype(np.uint8)
        m = RLEMask.from_array(arr)
        mat, out_shape = self.rot_mat(k, *shape)
        ref = np.rot90(arr, k=k)
        got_a = m.warp_affine(mat, out_shape)
        assert np.array_equal(np.array(got_a), ref), f'affine k={k} shape={shape}'
        got_p = m.warp_perspective(np.vstack([mat, [0, 0, 1]]), out_shape)
        assert np.array_equal(np.array(got_p), ref), f'perspective k={k} shape={shape}'

    def test_degenerate_matrices(self):
        with pytest.raises(ValueError):
            RLEMask.ones((4, 4)).warp_affine(np.array([[1., 1, 0], [1, 1, 0]]), (4, 4))
        with pytest.raises(ValueError):
            RLEMask.ones((40, 60)).warp_affine(np.array([[1., 0, 0], [10, 0.5, 0]]), (40, 60))
        with pytest.raises(ValueError):
            RLEMask.ones((4, 4)).warp_affine(np.array([[np.nan, 0, 0], [0, 1, 0]]), (4, 4))
        with pytest.raises(ValueError):
            RLEMask.ones((5, 5)).warp_affine(np.eye(2), (5, 5))


class TestLargestInteriorRectangle:
    def test_fg_only_in_last_column(self):
        arr = np.zeros((3, 3), np.uint8)
        arr[:, 2] = 1
        assert tuple(RLEMask.from_array(arr).largest_interior_rectangle()) == (2, 0, 1, 3)

    def test_aspect_variant_fg_only_in_last_column(self):
        arr = np.zeros((3, 3), np.uint8)
        arr[:, 2] = 1
        rect = RLEMask.from_array(arr).largest_interior_rectangle(aspect_ratio=1.0)
        assert rect is not None

    def test_around_center_no_background(self):
        m = RLEMask.from_counts([0, 1, 6, 8], shape=(3, 5), order='F')
        assert tuple(m.largest_interior_rectangle_around([2, 1])) == (2, 1, 1, 1)

    def test_around_center_random_never_covers_background(self):
        rng = np.random.default_rng(5)
        for _ in range(200):
            h, w = rng.integers(2, 9), rng.integers(2, 9)
            arr = (rng.random((h, w)) < 0.6).astype(np.uint8)
            fg = np.argwhere(arr)
            if len(fg) == 0:
                continue
            cy, cx = fg[rng.integers(len(fg))]
            x, y, rw, rh = RLEMask.from_array(arr).largest_interior_rectangle_around([cx, cy])
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = int(np.ceil(x + rw)), int(np.ceil(y + rh))
            if rw > 0 and rh > 0:
                assert arr[y0:y1, x0:x1].all(), (arr, (cx, cy), (x, y, rw, rh))


class TestFromCircle:
    @staticmethod
    def circle_ref(cx, cy, radius, h, w):
        yy, xx = np.mgrid[:h, :w]
        return ((xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2).astype(np.uint8)

    def test_border_crossing_circles(self):
        for (cx, cy, r) in [(1, 4, 3), (4, 0.5, 2.5), (-1, -1, 3), (7.5, 7.5, 4)]:
            got = RLEMask.from_circle((cx, cy), r, imshape=(8, 8))
            ref = self.circle_ref(cx, cy, r, 8, 8)
            assert np.array_equal(np.array(got), ref), (cx, cy, r)

    def test_nan_rejected(self):
        with pytest.raises(ValueError):
            RLEMask.from_circle((np.nan, 1), 2, imshape=(8, 8))


class TestSetItemIntPath:
    def test_non_binary_value_rejected(self):
        m = RLEMask.ones((2, 2))
        with pytest.raises(ValueError):
            m[0, 0] = 2

    def test_out_of_bounds(self):
        m = RLEMask.zeros((4, 4))
        with pytest.raises(IndexError):
            m[10, 10] = 1
        with pytest.raises(IndexError):
            m[-10, 0] = 1

    def test_numpy_integer_indices(self):
        m = RLEMask.ones((3, 3))
        assert m[np.int64(1), np.int64(1)] == 1
        m[np.int64(0), np.int64(0)] = np.uint8(0)
        assert m.area() == 8

    def test_no_interior_zero_run(self):
        m = RLEMask.from_counts([2, 1, 2], shape=(5, 1), order='F')
        m[2, 0] = 0
        assert m.counts.tolist() == [5]
        assert m == RLEMask.zeros((5, 1))

    def test_setitem_fuzz_canonical(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            h, w = rng.integers(1, 7), rng.integers(1, 7)
            arr = rng.integers(0, 2, (h, w)).astype(np.uint8)
            m = RLEMask.from_array(arr)
            for _ in range(10):
                i, j, v = rng.integers(h), rng.integers(w), rng.integers(2)
                m[i, j] = int(v)
                arr[i, j] = v
            counts = m.counts
            assert counts.sum() == h * w
            assert (counts[1:] > 0).all()
            assert np.array_equal(np.array(m), arr)


class TestVerticalMorphology:
    """Vertical dilation used to bleed across column boundaries for amounts > 1."""

    @staticmethod
    def dilate_vertical_ref(arr, up, down):
        # a foreground pixel at row r covers rows [r - up, r + down], clipped to the column
        h = arr.shape[0]
        ref = np.zeros_like(arr)
        for dy in range(-up, down + 1):
            if abs(dy) >= h:
                continue
            shifted = np.zeros_like(arr)
            if dy >= 0:
                shifted[dy:h, :] = arr[0:h - dy, :]
            else:
                shifted[0:h + dy, :] = arr[-dy:h, :]
            ref |= shifted
        return ref

    def test_dilate_vertical_no_column_bleed(self):
        arr = np.zeros((4, 2), np.uint8)
        arr[1, 1] = 1
        got = RLEMask.from_array(arr).dilate_vertical(up=2, down=1)
        ref = np.zeros((4, 2), np.uint8)
        ref[0:3, 1] = 1
        assert np.array_equal(np.array(got), ref)

    @pytest.mark.parametrize('pos', [(1, 1), (4, 1), (0, 0), (5, 2)])
    def test_dilate_kernel5_vs_scipy(self, pos):
        arr = np.zeros((6, 3), np.uint8)
        arr[pos] = 1
        got = np.array(RLEMask.from_array(arr).dilate(kernel_shape='square', kernel_size=5))
        ref = scipy.ndimage.binary_dilation(arr, structure=np.ones((5, 5))).astype(np.uint8)
        assert np.array_equal(got, ref)

    def test_dilate_vertical_fuzz(self):
        rng = np.random.default_rng(11)
        for _ in range(100):
            h, w = rng.integers(1, 9), rng.integers(1, 9)
            arr = (rng.random((h, w)) < 0.4).astype(np.uint8)
            up, down = int(rng.integers(0, 4)), int(rng.integers(0, 4))
            got = np.array(RLEMask.from_array(arr).dilate_vertical(up=up, down=down))
            ref = self.dilate_vertical_ref(arr, up, down)
            assert np.array_equal(got, ref), (arr, up, down)


class TestLargeMasks:
    def test_domain_limit_enforced(self):
        with pytest.raises(ValueError):
            RLEMask.zeros((10 ** 5, 10 ** 5))
        with pytest.raises(ValueError):
            RLEMask.ones((10 ** 5, 10 ** 5))
        with pytest.raises(ValueError):
            rlemasklib.ones((10 ** 5, 10 ** 5))
        with pytest.raises(ValueError):
            RLEMask.from_counts([10 ** 10], shape=(10 ** 5, 10 ** 5), order='F')

    def test_counts_beyond_uint32_raise_cleanly(self):
        # numpy >= 2 raises OverflowError when converting out-of-bounds Python ints;
        # a ValueError must result on every numpy version, even without the sum check
        with pytest.raises(ValueError):
            RLEMask.from_counts(
                [10 ** 10], shape=(10 ** 5, 10 ** 5), order='F', validate_sum=False)
        with pytest.raises(ValueError):
            RLEMask.from_counts([-5, 30], shape=(5, 5), validate_sum=False)

    def test_ucounts_beyond_uint32_raise_cleanly(self):
        # [2**32 + 20, 5] wraps to the plausible [20, 5] on numpy 1 (silently accepted
        # as corrupt data) and raises OverflowError on numpy 2; both the OOP and the
        # functional dict paths must give a clean ValueError instead
        bad = {'size': [5, 5], 'ucounts': [2 ** 32 + 20, 5]}
        with pytest.raises(ValueError):
            RLEMask.from_dict(bad)
        with pytest.raises(ValueError):
            rlemasklib.area(bad)
        with pytest.raises(ValueError):
            rlemasklib.compress(bad)
        with pytest.raises(ValueError):
            RLEMask.from_dict({'size': [5, 5], 'ucounts': [-7, 32]})

    def test_2_31_pixel_union(self):
        u = RLEMask.ones((65536, 32768)) | RLEMask.zeros((65536, 32768))
        assert u.area() == 2 ** 31
        assert u.counts.tolist() == [0, 2 ** 31]

    def test_2_31_pixel_nor(self):
        u = RLEMask.ones((65536, 32768)).merge(RLEMask.zeros((65536, 32768)), BoolFunc.NOR)
        assert u.area() == 0


class TestFunctionalDictFormats:
    """area/any/all/centroid should accept counts, ucounts and zcounts dicts."""

    def test_all_key_types(self):
        ud = {'size': [2, 2], 'ucounts': np.array([1, 3], np.uint32)}
        cd = rlemasklib.compress(ud)
        zd = rlemasklib.compress(cd, zlevel=-1)
        for d in (ud, cd, zd):
            assert rlemasklib.area(d) == 3
            assert rlemasklib.any(d)
            assert not rlemasklib.all(d)
            assert rlemasklib.centroid(d) is not None
            assert rlemasklib.decode(d).sum() == 3

    def test_decompress_returns_copy(self):
        ud = {'size': [2, 2], 'ucounts': np.array([1, 3], np.uint32)}
        assert rlemasklib.decompress(ud) is not ud


class TestFunctionalValidation:
    def test_crop_broadcast(self):
        d = encode_arr(np.ones((4, 4)))
        res = rlemasklib.crop([d, d], [0, 0, 2, 2])
        assert all(r['size'] == [2, 2] for r in res)
        assert all(rlemasklib.decode(r).sum() == 4 for r in res)

    def test_crop_length_mismatch(self):
        d = encode_arr(np.ones((4, 4)))
        with pytest.raises(ValueError):
            rlemasklib.crop([d, d, d], [[0, 0, 2, 2]] * 2)

    def test_pad_negative(self):
        d = encode_arr(np.ones((3, 3)))
        with pytest.raises(ValueError):
            rlemasklib.pad(d, [-1, 0, 0, 0])
        with pytest.raises(ValueError):
            RLEMask.ones((3, 3)).pad(-1, 0, 0, 0)

    def test_pad_strided_paddings(self):
        d = encode_arr(np.ones((3, 3)))
        base = np.array([1, 99, 2, 98, 1, 97, 3, 96], np.uint32)
        strided = base[::2]
        res = rlemasklib.pad(d, strided)
        assert res['size'] == [3 + 1 + 3, 3 + 1 + 2]

    def test_shift_zero_offset_copy(self):
        d = encode_arr(np.eye(3))
        assert rlemasklib.shift(d, (0, 0)) is not d

    def test_iou_returns_float(self):
        val = rlemasklib.iou([encode_arr(np.eye(3)), encode_arr(1 - np.eye(3))])
        assert isinstance(val, float)

    def test_from_bbox_short(self):
        with pytest.raises(ValueError):
            RLEMask.from_bbox([1, 2, 3], imshape=(5, 5))

    def test_from_circle_short(self):
        with pytest.raises(ValueError):
            RLEMask.from_circle([5], 2, imshape=(8, 8))


class TestFromArraySemantics:
    def test_float_native_threshold(self):
        assert RLEMask.from_array(np.full((2, 2), 0.5)).area() == 0
        assert RLEMask.from_array(np.full((2, 2), 1.5)).area() == 4
        assert RLEMask.from_array(np.full((2, 2), -3, np.int32)).area() == 0

    def test_uint8_threshold_domain(self):
        arr = np.array([[0, 100], [200, 255]], np.uint8)
        with pytest.raises(ValueError):
            RLEMask.from_array(arr, threshold=300)
        with pytest.raises(ValueError):
            RLEMask.from_array(arr, threshold=0)
        assert RLEMask.from_array(arr, threshold=128).area() == 2

    def test_nested_list_constructor(self):
        assert RLEMask([[1, 0], [0, 1]]) == RLEMask.from_array(np.eye(2))


class TestLabelMaps:
    def test_out_of_range_labels_rejected(self):
        with pytest.raises(ValueError):
            RLEMask.from_label_map(np.array([[256, 1], [0, 1]]))

    def test_merge_to_label_map_roundtrip(self):
        rng = np.random.default_rng(13)
        lm = rng.integers(0, 5, (9, 11)).astype(np.uint8)
        masks = RLEMask.from_label_map(lm)
        rles = [masks[label] for label in sorted(masks)]
        out = RLEMask.merge_to_label_map(rles)
        # labels are renumbered 1..n in sorted order
        relabel = np.zeros(256, np.uint8)
        for new, label in enumerate(sorted(masks), 1):
            relabel[label] = new
        assert np.array_equal(out, relabel[lm])


class TestMiscValidation:
    def test_fill_value_validated(self):
        with pytest.raises(ValueError):
            RLEMask.ones((4, 4)).fill_rectangle([0, 0, 2, 2], value=2)

    def test_module_original_preserved_for_aliases(self):
        assert rlemasklib.ones._module_original_ == 'rlemasklib._functional'
        assert rlemasklib.full is rlemasklib.ones

    def test_png_channel_validation(self):
        import io

        from PIL import Image

        img = np.zeros((5, 4, 3), np.uint8)
        img[:, :, 0] = 255
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        png_rgb = buf.getvalue()
        with pytest.raises(ValueError):
            RLEMask.from_png(data=png_rgb, channel=-2)
        with pytest.raises(ValueError):
            RLEMask.from_png(data=png_rgb, channel=-1)
        assert RLEMask.from_png(data=png_rgb, channel=0).area() == 20
        assert RLEMask.from_png(data=png_rgb, channel=1).area() == 0


class TestNonCanonicalConstruction:
    """Constructors must normalize interior zero-length runs (which violate the canonical-form
    invariant that run-walking algorithms rely on) instead of storing them and later crashing
    or hanging."""

    def test_from_counts_interior_zeros_transpose(self):
        m = RLEMask.from_counts([1, 0, 0, 0, 0, 5], (3, 2))
        assert m == RLEMask.from_counts([1, 5], (3, 2))
        t = m.T  # previously a heap-buffer overflow (SIGABRT)
        assert np.array_equal(np.array(t), np.array(m).T)

    def test_from_dict_coco_string_interior_zeros(self):
        d = RLEMask.from_counts([1, 0, 0, 0, 0, 5], (3, 2)).to_dict()
        m = RLEMask.from_dict(d)
        assert np.array_equal(np.array(m.rot90(1)), np.rot90(np.array(m), 1))

    def test_from_dict_ucounts_interior_zeros_contours(self):
        # a non-canonical mask used to make contours()/perimeter() hang forever
        d = {'size': [6, 6], 'ucounts': [0, 0, 0, 3, 0, 3, 0, 0, 0, 4, 0, 0,
                                         4, 3, 2, 4, 2, 0, 0, 2, 4, 3, 0, 1, 0, 1]}
        m = RLEMask.from_dict(d)
        c = m.contours()
        assert c.area() >= 0
        assert m.counts.tolist() == [x for x in m.counts.tolist() if True]  # canonical
        assert (m.counts[1:] > 0).all()  # no interior/trailing zero runs

    def test_functional_paths_canonicalize(self):
        d = {'size': [3, 2], 'ucounts': [1, 0, 0, 0, 0, 5]}
        # functional merge over a non-canonical dict must not crash
        out = rlemasklib.union([d, rlemasklib.zeros((3, 2))])
        assert rlemasklib.area(out) == 5


class TestMultiMaskDecodeAlignment:
    """decode([...]) of several masks must align every plane, including odd-m masks whose last
    pixel is background."""

    def test_decode_multi_mask_odd_m(self):
        m0 = np.zeros((3, 4), np.uint8)
        m0[0, 0] = 1
        m0[1, 1] = 1
        m1 = np.ones((3, 4), np.uint8)
        enc = [encode_arr(x) for x in [m0, m1, m0]]
        out = rlemasklib.decode(enc)
        assert all(np.array_equal(out[:, :, i], x) for i, x in enumerate([m0, m1, m0]))

    def test_decode_uncompressed_multi_mask(self):
        m0 = np.zeros((3, 4), np.uint8)
        m0[0, 0] = 1
        m1 = np.ones((3, 4), np.uint8)
        enc = [rlemasklib.encode(np.asfortranarray(x), compressed=False) for x in [m0, m1]]
        out = rlemasklib.decode(enc)
        assert np.array_equal(out[:, :, 0], m0) and np.array_equal(out[:, :, 1], m1)


class TestConcatDomainGuard:
    """hconcat/vconcat/tile must reject results exceeding the 2**32-1 pixel domain."""

    def test_hconcat_over_domain(self):
        a = RLEMask.ones((65535, 40000))
        with pytest.raises(ValueError):
            RLEMask.hconcat([a, a])

    def test_vconcat_over_domain(self):
        a = RLEMask.ones((65535, 40000))
        with pytest.raises(ValueError):
            RLEMask.vconcat([a, a])

    def test_tile_over_domain(self):
        a = RLEMask.ones((65535, 40000))
        with pytest.raises(ValueError):
            a.tile(1, 2)

    def test_normal_concat_ok(self):
        c = RLEMask.hconcat([RLEMask.ones((3, 2)), RLEMask.zeros((3, 2))])
        assert c.shape == (3, 4) and c.area() == 6


class TestFloatThreshold:
    def test_float_threshold_honored(self):
        d = np.array([[0.1, 0.4], [0.6, 0.9]], np.float32)
        assert np.array_equal(np.array(RLEMask.from_array(d, threshold=0.5)),
                              (d >= 0.5).astype(np.uint8))

    def test_uint8_noninteger_threshold_rejected(self):
        with pytest.raises(ValueError):
            RLEMask.from_array(np.array([[0, 200]], np.uint8), threshold=1.5)

    def test_uint8_integer_threshold_ok(self):
        assert RLEMask.from_array(np.array([[0, 200]], np.uint8), threshold=128).area() == 1


class TestUnsafeApiRaisesNotAborts:
    """RLEs made invalid through the explicitly unsafe APIs must raise a catchable ValueError
    on a later merge, not abort the process."""

    def test_validate_sum_false_merge(self):
        bad = RLEMask.from_counts([5, 5], (5, 5), validate_sum=False)
        with pytest.raises(ValueError):
            _ = bad | RLEMask.zeros((5, 5))

    def test_shape_setter_merge(self):
        m = RLEMask.ones((4, 4))
        m.cy.shape = (5, 5)
        with pytest.raises(ValueError):
            _ = m | RLEMask.zeros((5, 5))

    def test_counts_view_mutation_merge(self):
        m = RLEMask.from_counts([2, 1, 1], (4, 1))
        v = m.counts_view
        v[0] = 99
        with pytest.raises(ValueError):
            _ = m & RLEMask.ones((4, 1))

    def test_no_stale_error_after_caught(self):
        try:
            _ = RLEMask.from_counts([5, 5], (5, 5), validate_sum=False) | RLEMask.zeros((5, 5))
        except ValueError:
            pass
        # a subsequent well-formed merge must not spuriously raise
        assert (RLEMask.ones((3, 3)) | RLEMask.zeros((3, 3))).area() == 9


class TestDictSizeValidation:
    def test_non_integer_size_rejected(self):
        with pytest.raises(ValueError):
            RLEMask.from_dict({'size': [2.5, 2], 'counts': b'04'})


class TestMorphologyOpenClose:
    """opening/closing are advertised in the README/index; they must exist on RLEMask (not
    only in the functional API) and equal the erode-then-dilate / dilate-then-erode compositions."""

    def test_opening_equals_erode_then_dilate(self):
        arr = (np.random.default_rng(1).random((50, 50)) < 0.5).astype(np.uint8)
        m = RLEMask.from_array(arr)
        assert m.opening(kernel_shape='square', kernel_size=3) == \
            m.erode('square', 3).dilate('square', 3)

    def test_closing_equals_dilate_then_erode(self):
        arr = (np.random.default_rng(2).random((50, 50)) < 0.5).astype(np.uint8)
        m = RLEMask.from_array(arr)
        assert m.closing(kernel_shape='square', kernel_size=3) == \
            m.dilate('square', 3).erode('square', 3)

    def test_opening_inplace(self):
        arr = (np.random.default_rng(3).random((30, 30)) < 0.5).astype(np.uint8)
        m = RLEMask.from_array(arr)
        ref = m.opening('circle', 5)
        assert m.opening('circle', 5, inplace=True) is m
        assert m == ref

    def test_bad_kernel_shape_message(self):
        with pytest.raises(ValueError, match="circle"):
            RLEMask.ones((5, 5)).dilate("blob")
        with pytest.raises(TypeError, match="kernel_shape"):
            RLEMask.ones((5, 5)).dilate(1)


class TestNamespaceHygiene:
    def test_no_leaked_loop_globals(self):
        assert not hasattr(rlemasklib, 'x')
        assert not hasattr(rlemasklib, 'obj')

    def test_from_png_multichannel_error_mentions_channel(self):
        import io

        from PIL import Image

        img = np.zeros((5, 4, 3), np.uint8)
        img[:, :, 0] = 255
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        with pytest.raises(ValueError, match="channel="):
            RLEMask.from_png(data=buf.getvalue())


class TestNaiveUserFixes:
    """Fixes from the naive-user documentation review."""

    def test_resize_with_scale_factors_only(self):
        # docstring promised fx/fy alone is enough; output_imshape now defaults to None
        assert RLEMask.ones((100, 100)).resize(fx=0.5, fy=0.5).shape == (50, 50)
        assert RLEMask.ones((100, 100)).resize(fx=2.0, fy=0.5).shape == (50, 200)

    def test_from_polygon_rejects_list(self):
        outer = np.array([[0, 0], [10, 0], [10, 10]])
        hole = np.array([[3, 3], [7, 3], [7, 7]])
        with pytest.raises(ValueError):
            RLEMask.from_polygon([outer, hole], imshape=(20, 20))
        with pytest.raises(ValueError):
            rlemasklib.from_polygon(np.zeros((2, 4, 2)), imshape=(20, 20))

    def test_from_polygon_single_still_works(self):
        m = RLEMask.from_polygon(np.array([[2, 2], [10, 2], [10, 10], [2, 10]]), imshape=(20, 20))
        assert m.area() > 0

    def test_polygon_holes_via_subtract(self):
        # the documented replacement recipe for "polygon with holes"
        outer = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        hole = np.array([[30, 30], [70, 30], [70, 70], [30, 70]])
        full = RLEMask.from_polygon(outer, imshape=(200, 200))
        holed = full - RLEMask.from_polygon(hole, imshape=(200, 200))
        assert 0 < holed.area() < full.area()

    def test_from_png_missing_file_raises_filenotfound(self):
        with pytest.raises(FileNotFoundError):
            RLEMask.from_png('/nonexistent/definitely/not/here.png')

    def test_coco_compression_walkthrough_value(self):
        # the doc walkthrough now matches reality: [8,12,6,15] -> b'8<63'
        assert RLEMask.from_counts([8, 12, 6, 15], shape=(41, 1)).to_dict()['counts'] == b'8<63'

    def test_boolfunc_difference_value(self):
        # boolean-operations.rst: DIFFERENCE (A-B) is 0b0100 = 4
        assert int(BoolFunc.DIFFERENCE) == 4

    def test_decode_into_supports_integer_and_float_dtypes(self):
        m = RLEMask.from_array(np.eye(4, dtype=np.uint8))
        for dt in (np.uint8, np.int32, np.uint16, np.int64, np.float32, np.float64):
            arr = np.zeros((4, 4), dt)
            m.decode_into(arr, 7)
            assert arr.max() == 7

    def test_array_kernel_raises_typeerror(self):
        # cv2 habit: cv2.erode(img, kernel); previously died with a cryptic numpy
        # truth-value ValueError
        kernel = np.ones((7, 7), np.uint8)
        for method in ('erode', 'dilate', 'opening', 'closing'):
            with pytest.raises(TypeError, match="kernel_shape"):
                getattr(RLEMask.ones((9, 9)), method)(kernel)

    def test_erode_inplace_not_corrupted_by_bad_kernel(self):
        m = RLEMask.from_array(np.eye(5, dtype=np.uint8))
        ref = m.copy()
        with pytest.raises(TypeError):
            m.erode(np.ones((3, 3)), inplace=True)
        assert m == ref

    def test_iteration_raises_typeerror(self):
        # previously fell back to legacy __getitem__ iteration and failed with a
        # confusing "Only 2D slicing is supported" error
        m = RLEMask.ones((3, 3))
        with pytest.raises(TypeError, match="not iterable"):
            iter(m)
        with pytest.raises(TypeError, match="not iterable"):
            list(m)

    def test_shape_mismatch_error_mentions_shapes(self):
        a = RLEMask.zeros((4, 5))
        b = RLEMask.zeros((6, 7))
        with pytest.raises(ValueError, match=r"\(4, 5\).*\(6, 7\)"):
            a | b
        with pytest.raises(ValueError, match=r"\(4, 5\)"):
            RLEMask.union([a, b])

    def test_nonzero_matches_numpy(self):
        rng = np.random.default_rng(0)
        for shape in [(1, 1), (7, 3), (3, 7), (40, 60)]:
            arr = (rng.random(shape) < 0.4).astype(np.uint8)
            rows, cols = RLEMask.from_array(arr).nonzero()
            np_rows, np_cols = np.nonzero(arr)
            assert np.array_equal(rows, np_rows)
            assert np.array_equal(cols, np_cols)
            assert rows.dtype == np.intp and cols.dtype == np.intp

    def test_nonzero_empty_mask(self):
        rows, cols = RLEMask.zeros((5, 5)).nonzero()
        assert rows.shape == (0,) and cols.shape == (0,)
