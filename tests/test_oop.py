import os
import warnings

import cv2
import numpy as np
from rlemasklib.oop import RLEMask


def test_union():
    d1 = RLEMask.from_array(np.eye(3))
    d2 = RLEMask.from_array(np.eye(3)[::-1])
    d3 = d1 | d2
    assert np.all(np.array(d3) == np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]))


def test_intersection():
    d1 = RLEMask.from_array(np.eye(3))
    d2 = RLEMask.from_array(np.eye(3)[::-1])
    d3 = d1 & d2
    assert np.all(np.array(d3) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))


def test_difference():
    d1 = RLEMask.from_array(np.eye(3))
    d2 = RLEMask.from_array(np.eye(3)[::-1])
    d3 = d1 - d2
    assert np.all(np.array(d3) == np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]))


def test_slicing():
    d1 = RLEMask.from_array(np.eye(3))
    d2 = d1[1:3, 1:3]
    assert np.all(np.array(d2) == np.eye(2))


def test_set_rect():
    d1 = RLEMask.zeros((3, 3))
    d1[1:3, 1:3] = 1
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]))


def test_set_rect_other():
    d1 = RLEMask.zeros((3, 3))
    d1[1:3, 1:3] = RLEMask.from_array(np.eye(2))
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_set_rect_np():
    d1 = RLEMask.zeros((3, 3))
    d1[1:3, 1:3] = np.eye(2)
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_set_pixel():
    d1 = RLEMask.zeros((3, 3))
    d1[1, 1] = 1
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]))


def test_get_pixel():
    d1 = RLEMask.from_array(np.eye(3))
    assert d1[1, 1] == 1


def test_get_pixel_cython():
    d1 = RLEMask.from_array(np.eye(3))
    assert d1[1, 0] == 0
    assert d1[1, 1] == 1


def test_slicingc():
    d1 = RLEMask.from_array(np.eye(3))
    d2 = d1[1:3, 1:3]
    assert np.all(np.array(d2) == np.eye(2))


def test_from_dictc():
    d1 = RLEMask.from_dict({"ucounts": [0, 1, 2, 1], "size": [2, 2]})
    assert np.all(np.array(d1) == np.eye(2))

    d2 = RLEMask.from_array(np.eye(3))
    d3 = RLEMask.from_dict(d2.to_dict())
    assert np.all(np.array(d3) == np.eye(3))

    d2 = RLEMask.from_array(np.eye(3))
    d3 = RLEMask.from_dict(d2.to_dict(zlevel=-1))
    assert np.all(np.array(d3) == np.eye(3))


def test_set_pixelc():
    d1 = RLEMask.zeros((3, 3))
    d1[1, 1] = 1
    d1[1, 0] = 1
    d1[1, 0] = 0
    d1[1, 2] = 1
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]))

    d1 = RLEMask.zeros((3, 3))
    d1[0, 0] = 1
    d1[2, 2] = 1
    assert np.all(np.array(d1) == np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]))


def test_set_slicec():
    d1 = RLEMask.zeros((3, 3))
    d1[1:3, 1:3] = 1
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]]))

    d1 = RLEMask.from_array(np.zeros((3, 3)))
    d1[1:3, 1:3] = RLEMask.from_array(np.eye(2))
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]))

    d1 = RLEMask.from_array(np.zeros((3, 3)))
    d1[1:3, 1:3] = np.eye(2)
    assert np.all(np.array(d1) == np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]))


def test_slice_stride():
    mask = np.random.randint(0, 2, (3, 3))
    rle = RLEMask.from_array(mask)
    assert np.all(np.array(rle[1::2, 1::2]) == mask[1::2, 1::2])
    assert np.all(np.array(rle[::2, ::2]) == mask[::2, ::2])


def test_max_pool():
    for i in range(100):
        mask = np.random.randint(0, 2, (2, 2))
        rle = RLEMask.from_array(mask)
        assert np.all(np.array(rle.max_pool2x2()) == np.max(mask, keepdims=True))


def test_min_pool():
    for i in range(100):
        mask = np.random.randint(0, 2, (2, 2))
        rle = RLEMask.from_array(mask)
        assert np.all(np.array(rle.min_pool2x2()) == np.min(mask, keepdims=True))


def test_avg_pool():
    mask = np.array([[0, 1], [1, 1]])
    rle = RLEMask.from_array(mask)
    assert np.all(
        np.array(rle.avg_pool2x2()) == (np.mean(mask, keepdims=True) >= 0.5).astype(int)
    )

    mask = np.array([[0, 1], [1, 0]])
    rle = RLEMask.from_array(mask)
    assert np.all(
        np.array(rle.avg_pool2x2()) == (np.mean(mask, keepdims=True) >= 0.5).astype(int)
    )

    mask = np.array([[0, 1], [0, 0]])
    rle = RLEMask.from_array(mask)
    assert np.all(
        np.array(rle.avg_pool2x2()) == (np.mean(mask, keepdims=True) >= 0.5).astype(int)
    )


def test_remove_small_compc():
    mask = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]])
    rle = RLEMask.from_array(mask)
    rle.remove_small_components(connectivity=4, min_size=6, inplace=True)
    assert rle == RLEMask(np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))

    mask = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]])
    rle = RLEMask.from_array(mask)
    rle.remove_small_components(connectivity=4, min_size=2, inplace=True)
    assert rle == RLEMask(np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]]))

    mask = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    rle = RLEMask.from_array(mask)
    rle.fill_small_holes(connectivity=4, min_size=2, inplace=True)
    assert rle == RLEMask(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))


def test_largest_compc():
    mask = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]])
    rle = RLEMask.from_array(mask)
    rle.largest_connected_component(connectivity=4, inplace=True)
    assert rle == RLEMask(np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]]))

    mask = np.array([[0, 1, 1], [0, 1, 1], [1, 0, 0]])
    rle = RLEMask.from_array(mask)
    rle.largest_connected_component(connectivity=4, inplace=True)
    assert rle == RLEMask(np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]]))


def test_padc():
    for i in range(10000):
        h = np.random.randint(1, 10)
        w = np.random.randint(1, 10)
        mask = np.random.randint(0, 2, (h, w))
        p = np.random.randint(0, 2, 4)
        rle = RLEMask.from_array(mask)
        rle = rle.pad(p[0], p[1], p[2], p[3])
        assert np.all(np.array(rle) == np.pad(mask, ((p[0], p[1]), (p[2], p[3]))))

    mask = np.array(
        [
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        np.uint8,
    )
    rle = RLEMask.from_array(mask)
    rle = rle.pad(1, 1, 1, 1)
    assert np.all(np.array(rle) == np.pad(mask, 1))


def test_merge_multibool():
    mask1 = np.array([[0, 1, 1], [1, 1, 1], [0, 0, 0]])
    mask2 = np.array([[1, 0, 0], [0, 0, 0], [1, 1, 1]])
    mask3 = np.array([[1, 0, 0], [0, 0, 0], [0, 1, 0]])
    rle1 = RLEMask.from_array(mask1)
    rle2 = RLEMask.from_array(mask2)
    rle3 = RLEMask.from_array(mask3)

    rle = RLEMask.merge_many_custom(
        [rle1, rle2, rle3, rle1, rle2, rle3, rle1, rle2, rle3],
        lambda a1, a2, a3, a4, a5, a6, a7, a8, a9: a1
        | a2
        | a3
        | a4
        | a5
        | a6
        | a7
        | a8
        | a9,
    )
    assert np.all(np.array(rle) == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    rle = RLEMask.merge_many_custom([rle1, rle2, rle3], lambda a, b, c: (a | b | c))
    assert np.all(np.array(rle) == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))

    rle = RLEMask.merge_many_custom([rle1, rle2, rle3], lambda a, b, c: (a | b) & ~c)
    assert np.all(np.array(rle) == np.array([[0, 1, 1], [1, 1, 1], [1, 0, 1]]))

    mergefun = RLEMask.make_merge_function(lambda a, b, c: (a | b) & ~c)
    rle = mergefun(rle1, rle2, rle3)
    assert np.all(np.array(rle) == np.array([[0, 1, 1], [1, 1, 1], [1, 0, 1]]))

    mergefun = RLEMask.make_merge_function(
        lambda a1, a2, a3, a4, a5, a6, a7, a8, a9: a1
        | a2
        | a3
        | a4
        | a5
        | a6
        | a7
        | a8
        | a9
    )
    rle = mergefun(rle1, rle2, rle3, rle1, rle2, rle3, rle1, rle2, rle3)
    assert np.all(np.array(rle) == np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]))


def test_circle():
    import tempfile
    import imageio.v2 as imageio

    rle = RLEMask.from_circle([499 / 2, 499 / 2], 199, imshape=(500, 500))

    import cv2

    poly = cv2.ellipse2Poly((499 // 2, 499 // 2), (199, 199), 0, 0, 360, 1)
    rle2 = RLEMask.from_polygon(poly, imshape=(500, 500))

    mask = np.array(rle2 - rle)
    imageio.imwrite(os.path.join(tempfile.gettempdir(), "circle.png"), mask * 255)


def test_iou():
    mask1 = np.random.randint(0, 2, (3, 3))
    mask2 = np.random.randint(0, 2, (3, 3))
    mask3 = np.random.randint(0, 2, (3, 3))
    mask4 = np.random.randint(0, 2, (3, 3))
    rle1 = RLEMask.from_array(mask1)
    rle2 = RLEMask.from_array(mask2)
    rle3 = RLEMask.from_array(mask3)
    rle4 = RLEMask.from_array(mask4)

    iou = RLEMask.iou_matrix([rle1, rle2], [rle3, rle4])
    iou2 = np.array(
        [[rle1.iou(rle3), rle1.iou(rle4)], [rle2.iou(rle3), rle2.iou(rle4)]]
    )
    assert np.all(iou == iou2)

    iou = rle1.iou(rle1)
    assert iou == 1.0


def test_transpose():
    # mask = np.array([[0, 1, 1], [1, 1, 0], [1, 1, 0]])

    for i in range(1000):
        w = np.random.randint(0, 10)
        h = np.random.randint(0, 10)
        mask = np.random.randint(0, 2, (h, w))
        rle_transp = RLEMask.from_array(mask).transpose()
        rle_transp_correct = RLEMask.from_array(mask.T)
        assert rle_transp == rle_transp_correct
    # assert np.all(np.array(rle.transpose()) == mask.T)


def test_rot180():
    for i in range(100):
        w = np.random.randint(0, 10)
        h = np.random.randint(0, 10)
        mask = np.random.randint(0, 2, (h, w))
        rle_rot = RLEMask.from_array(mask).rot90(k=2)
        rle_rot_correct = RLEMask.from_array(mask[::-1, ::-1])
        assert rle_rot == rle_rot_correct


def test_rot90():
    for i in range(100):
        w = np.random.randint(0, 10)
        h = np.random.randint(0, 10)
        k = np.random.randint(0, 4)
        mask = np.random.randint(0, 2, (h, w))
        rle_rot = RLEMask.from_array(mask).rot90(k=k)
        rle_rot_correct = RLEMask.from_array(np.rot90(mask, k=k))
        assert rle_rot == rle_rot_correct


def test_flip():
    for i in range(1000):
        w = np.random.randint(0, 10)
        h = np.random.randint(0, 10)
        # mask
        # [[1 1]
        #  [1 0]
        #  [0 0]
        #  [0 1]
        #  [1 1]
        #  [1 0]
        #  [1 1]
        #  [0 1]
        #  [1 1]]

        # mask= np.array([[1, 1], [1, 0], [0, 0], [0, 1], [1, 1], [1, 0], [1, 1], [0, 1], [1, 1]])
        mask = np.random.randint(0, 2, (h, w))
        # print(mask)
        rle_flip = RLEMask.from_array(mask).flipud()
        rle_flip_correct = RLEMask.from_array(mask[::-1])
        assert rle_flip == rle_flip_correct

        rle_flip = RLEMask.from_array(mask).fliplr()
        rle_flip_correct = RLEMask.from_array(mask[:, ::-1])
        assert rle_flip == rle_flip_correct


def test_negative_step():
    for i in range(1000):
        w = np.random.randint(0, 10)
        h = np.random.randint(0, 10)
        mask = np.random.randint(0, 2, (h, w))

        start = np.random.randint(-w, 2 * w + 1)
        stop = np.random.randint(-w, 2 * w + 1)
        step = np.random.randint(1, 5) * np.random.choice([-1, 1])
        start2 = np.random.randint(-w, 2 * w + 1)
        stop2 = np.random.randint(-w, 2 * w + 1)
        step2 = np.random.randint(1, 5) * np.random.choice([-1, 1])

        rle_flip = RLEMask.from_array(mask)[start:stop:step]
        rle_flip_correct = RLEMask.from_array(mask[start:stop:step])
        assert rle_flip == rle_flip_correct, (
            f"Mismatch detected:\n"
            f"rle_flip: {np.array(rle_flip)}\n"
            f"rle_flip_correct: {np.array(rle_flip_correct)}\n"
            f"mask: {mask}\n"
            f"start: {start}, stop: {stop}, step: {step}\n"
            f"start2: {start2}, stop2: {stop2}, step2: {step2}\n"
        )

        rle_flip = RLEMask.from_array(mask)[:, start:stop:step]
        rle_flip_correct = RLEMask.from_array(mask[:, start:stop:step])
        assert rle_flip == rle_flip_correct, (
            f"Mismatch detected:\n"
            f"rle_flip: {np.array(rle_flip)}\n"
            f"rle_flip_correct: {np.array(rle_flip_correct)}\n"
            f"mask: {mask}\n"
            f"start: {start}, stop: {stop}, step: {step}\n"
            f"start2: {start2}, stop2: {stop2}, step2: {step2}\n"
        )

        start2 = np.random.randint(0, w + 1)
        stop2 = np.random.randint(0, w + 1)
        step2 = np.random.randint(1, 5) * np.random.choice([-1, 1])

        rle_flip = RLEMask.from_array(mask)[start:stop:step, start2:stop2:step2]
        rle_flip_correct = RLEMask.from_array(mask[start:stop:step, start2:stop2:step2])

        assert rle_flip == rle_flip_correct, (
            f"Mismatch detected:\n"
            f"rle_flip: {np.array(rle_flip)}\n"
            f"rle_flip_correct: {np.array(rle_flip_correct)}\n"
            f"mask: {mask}\n"
            f"start: {start}, stop: {stop}, step: {step}\n"
            f"start2: {start2}, stop2: {stop2}, step2: {step2}\n"
        )


def test_shift():
    for i in range(1000):
        w = np.random.randint(0, 10)
        h = np.random.randint(0, 10)
        mask = np.random.randint(0, 2, (h, w))
        shift = np.random.randint(-5, 6, 2)
        rle = RLEMask.from_array(mask)
        rle_shift = rle.shift(shift)
        shifted_mask = shift_arr(mask, shift)
        rle_shift_correct = RLEMask.from_array(shifted_mask)
        assert rle_shift == rle_shift_correct


def test_morph():
    kernel5x5 = np.array(
        [
            [0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )

    for i in range(1000):
        w = np.random.randint(1, 10)
        h = np.random.randint(1, 10)
        mask = np.random.randint(0, 2, (h, w)).astype(np.uint8)
        rle = RLEMask.from_array(mask)
        rle_dilated = rle.dilate3x3(connectivity=4)
        mask_dilate = cv2.dilate(
            mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        )
        rle_dilate_correct = RLEMask.from_array(mask_dilate)
        assert rle_dilated == rle_dilate_correct

        rle_eroded = rle.erode3x3(connectivity=4)
        mask_erode = cv2.erode(
            mask, np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        )
        rle_erode_correct = RLEMask.from_array(mask_erode)
        assert rle_eroded == rle_erode_correct

        rle_dilated = rle.dilate3x3(connectivity=8)
        mask_dilate = cv2.dilate(
            mask, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        )
        rle_dilate_correct = RLEMask.from_array(mask_dilate)
        assert rle_dilated == rle_dilate_correct

        rle_eroded = rle.erode3x3(connectivity=8)
        mask_erode = cv2.erode(
            mask, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
        )
        rle_erode_correct = RLEMask.from_array(mask_erode)
        assert rle_eroded == rle_erode_correct

        rle_dilated = rle.dilate5x5()
        mask_dilate = cv2.dilate(mask, kernel5x5)
        rle_dilate_correct = RLEMask.from_array(mask_dilate)
        assert rle_dilated == rle_dilate_correct

        rle_eroded = rle.erode5x5()
        mask_erode = cv2.erode(mask, kernel5x5)
        rle_erode_correct = RLEMask.from_array(mask_erode)
        assert rle_eroded == rle_erode_correct


def test_largest_interior_rectangle():
    grid = np.array(
        [
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0, 0],
        ]
    )
    rect = RLEMask.from_array(grid).largest_interior_rectangle()
    np.testing.assert_array_equal(rect, np.array([2, 2, 4, 7]))


def test_largest_interior_rectangle_around_center_integer():
    """Test that largest_interior_rectangle_around returns exact integer center."""
    mask = np.ones((20, 20), dtype=np.uint8)
    rle = RLEMask.from_array(mask)

    for center in [(5, 7), (10, 10), (3, 15), (8, 4)]:
        rect = rle.largest_interior_rectangle_around(center)
        x, y, w, h = rect
        result_cx = x + (w - 1) / 2
        result_cy = y + (h - 1) / 2
        np.testing.assert_almost_equal(result_cx, center[0], decimal=5)
        np.testing.assert_almost_equal(result_cy, center[1], decimal=5)


def test_largest_interior_rectangle_around_center_float():
    """Test that largest_interior_rectangle_around returns exact float center."""
    mask = np.ones((20, 20), dtype=np.uint8)
    rle = RLEMask.from_array(mask)

    for center in [(5.3, 7.8), (10.5, 10.5), (3.1, 15.9), (8.7, 4.2)]:
        rect = rle.largest_interior_rectangle_around(center)
        x, y, w, h = rect
        result_cx = x + (w - 1) / 2
        result_cy = y + (h - 1) / 2
        np.testing.assert_almost_equal(result_cx, center[0], decimal=5)
        np.testing.assert_almost_equal(result_cy, center[1], decimal=5)


def test_largest_interior_rectangle_around_aspect_ratio():
    """Test that largest_interior_rectangle_around returns exact aspect ratio."""
    mask = np.ones((30, 30), dtype=np.uint8)
    rle = RLEMask.from_array(mask)

    for aspect_ratio in [1.0, 1.5, 2.0, 0.5, 16 / 9, 4 / 3]:
        rect = rle.largest_interior_rectangle_around(
            (15, 15), aspect_ratio=aspect_ratio
        )
        x, y, w, h = rect
        if h > 0:
            result_aspect = w / h
            np.testing.assert_almost_equal(result_aspect, aspect_ratio, decimal=5)


def test_largest_interior_rectangle_around_center_and_aspect():
    """Test that both center and aspect ratio are exact."""
    mask = np.ones((30, 30), dtype=np.uint8)
    rle = RLEMask.from_array(mask)

    for center in [(10.3, 15.7), (14.5, 14.5), (8.2, 20.1)]:
        for aspect_ratio in [1.5, 2.0, 0.75]:
            rect = rle.largest_interior_rectangle_around(
                center, aspect_ratio=aspect_ratio
            )
            x, y, w, h = rect
            if h > 0:
                result_cx = x + (w - 1) / 2
                result_cy = y + (h - 1) / 2
                result_aspect = w / h
                np.testing.assert_almost_equal(result_cx, center[0], decimal=5)
                np.testing.assert_almost_equal(result_cy, center[1], decimal=5)
                np.testing.assert_almost_equal(result_aspect, aspect_ratio, decimal=5)


def test_largest_interior_rectangle_around_inside_mask():
    """Test that the returned rectangle is inside the foreground."""
    # Create a mask with some structure
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[5:25, 5:25] = 1

    rle = RLEMask.from_array(mask)

    for center in [(10, 10), (15.5, 15.5), (12.3, 18.7)]:
        for aspect_ratio in [None, 1.5, 2.0]:
            rect = rle.largest_interior_rectangle_around(
                center, aspect_ratio=aspect_ratio
            )
            x, y, w, h = rect
            if w > 0 and h > 0:
                # Check that rectangle is inside the foreground
                x_int, y_int = int(np.floor(x)), int(np.floor(y))
                w_ceil, h_ceil = (
                    int(np.ceil(x + w)) - x_int,
                    int(np.ceil(y + h)) - y_int,
                )
                rect_region = mask[y_int : y_int + h_ceil, x_int : x_int + w_ceil]
                assert rect_region.all(), f"Rectangle not inside mask for center={center}, aspect={aspect_ratio}"


def test_largest_interior_rectangle_aspect_ratio():
    """Test that largest_interior_rectangle with aspect_ratio returns exact aspect."""
    mask = np.ones((30, 30), dtype=np.uint8)
    rle = RLEMask.from_array(mask)

    for aspect_ratio in [1.0, 1.5, 2.0, 0.5, 16 / 9, 4 / 3]:
        rect = rle.largest_interior_rectangle(aspect_ratio=aspect_ratio)
        x, y, w, h = rect
        if h > 0:
            result_aspect = w / h
            np.testing.assert_almost_equal(result_aspect, aspect_ratio, decimal=5)


def shift_arr(arr, shifts):
    """
    Rolls a 2D array without wrapping, replacing rolled-in elements with zeros.

    Parameters:
        arr (np.ndarray): Input 2D array.
        shifts (tuple): A tuple of (vertical_shift, horizontal_shift).

    Returns:
        np.ndarray: Shifted 2D array with rolled-in elements replaced by zeros.
    """
    vertical, horizontal = shifts
    rolled = np.roll(np.roll(arr, vertical, axis=0), horizontal, axis=1)

    # Replace rolled-in elements with zeros
    if vertical > 0:
        rolled[:vertical, :] = 0
    elif vertical < 0:
        rolled[vertical:, :] = 0
    if horizontal > 0:
        rolled[:, :horizontal] = 0
    elif horizontal < 0:
        rolled[:, horizontal:] = 0

    return rolled


def test_decode_error():
    d1 = RLEMask.from_array(np.eye(3))
    d1.cy.shape = (2, 2)
    try:
        d1.to_array()
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError")


# --- to_array fg_value / bg_value / dtype tests ---

def test_to_array_default():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array()
    np.testing.assert_array_equal(result, mask)
    assert result.dtype == np.uint8


def test_to_array_fg_value():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=255)
    expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_to_array_bg_value():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(bg_value=128)
    expected = np.array([[128, 1], [1, 128]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_to_array_fg_and_bg():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=5, bg_value=10)
    expected = np.array([[10, 5], [5, 10]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_to_array_dtype_float32():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(dtype=np.float32)
    expected = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float32


def test_to_array_dtype_float32_with_nan_bg():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(bg_value=np.nan, dtype=np.float32)
    assert result.dtype == np.float32
    assert result[0, 1] == 1.0
    assert result[1, 0] == 1.0
    assert np.isnan(result[0, 0])
    assert np.isnan(result[1, 1])


def test_to_array_dtype_float64_custom_values():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=2.5, bg_value=-1.0, dtype=np.float64)
    expected = np.array([[-1.0, 2.5], [2.5, -1.0]], dtype=np.float64)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64


def test_to_array_deprecated_value():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = rle.to_array(value=255)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "fg_value" in str(w[0].message)
    expected = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_to_array_order_c():
    mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=1, order='C')
    np.testing.assert_array_equal(result, mask)
    assert result.flags.c_contiguous


def test_to_array_empty_mask():
    rle = RLEMask.zeros((0, 5))
    result = rle.to_array(fg_value=255, bg_value=128)
    assert result.shape == (0, 5)

    rle = RLEMask.zeros((5, 0))
    result = rle.to_array(fg_value=255, bg_value=128)
    assert result.shape == (5, 0)


def test_to_array_full_mask():
    rle = RLEMask.ones((3, 3))
    result = rle.to_array(fg_value=7, bg_value=3)
    expected = np.full((3, 3), 7, dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


def test_to_array_all_bg():
    rle = RLEMask.zeros((3, 3))
    result = rle.to_array(fg_value=7, bg_value=3)
    expected = np.full((3, 3), 3, dtype=np.uint8)
    np.testing.assert_array_equal(result, expected)


# --- decode_into tests ---

def test_decode_into_fg_only():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    canvas = np.full((2, 2), 99, dtype=np.uint8)
    rle.decode_into(canvas, fg_value=5)
    # fg pixels become 5, bg pixels stay 99
    expected = np.array([[99, 5], [5, 99]], dtype=np.uint8)
    np.testing.assert_array_equal(canvas, expected)


def test_decode_into_bg_only():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    canvas = np.full((2, 2), 99, dtype=np.uint8)
    rle.decode_into(canvas, bg_value=42)
    # bg pixels become 42, fg pixels stay 99
    expected = np.array([[42, 99], [99, 42]], dtype=np.uint8)
    np.testing.assert_array_equal(canvas, expected)


def test_decode_into_fg_and_bg():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    canvas = np.full((2, 2), 99, dtype=np.uint8)
    rle.decode_into(canvas, fg_value=5, bg_value=10)
    expected = np.array([[10, 5], [5, 10]], dtype=np.uint8)
    np.testing.assert_array_equal(canvas, expected)


def test_decode_into_both_none():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    canvas = np.full((2, 2), 99, dtype=np.uint8)
    rle.decode_into(canvas, fg_value=None, bg_value=None)
    # no-op, canvas unchanged
    expected = np.full((2, 2), 99, dtype=np.uint8)
    np.testing.assert_array_equal(canvas, expected)


def test_decode_into_deprecated_value():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    canvas = np.zeros((2, 2), dtype=np.uint8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        rle.decode_into(canvas, value=7)
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
    expected = np.array([[0, 7], [7, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(canvas, expected)


def test_decode_into_empty_mask():
    rle = RLEMask.zeros((3, 3))
    canvas = np.full((3, 3), 99, dtype=np.uint8)
    rle.decode_into(canvas, fg_value=5)
    # all-zero mask, no foreground pixels, canvas unchanged
    np.testing.assert_array_equal(canvas, np.full((3, 3), 99, dtype=np.uint8))


def test_decode_into_full_mask():
    rle = RLEMask.ones((3, 3))
    canvas = np.full((3, 3), 99, dtype=np.uint8)
    rle.decode_into(canvas, fg_value=5)
    np.testing.assert_array_equal(canvas, np.full((3, 3), 5, dtype=np.uint8))


# --- HWC (multi-channel) to_array tests ---

def test_to_array_hwc_fg_tuple():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=(255, 0, 0), bg_value=0)
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.uint8
    np.testing.assert_array_equal(result[0, 1], [255, 0, 0])  # fg pixel
    np.testing.assert_array_equal(result[0, 0], [0, 0, 0])  # bg pixel


def test_to_array_hwc_bg_tuple():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=255, bg_value=(128, 64, 32))
    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result[0, 1], [255, 255, 255])  # fg broadcast
    np.testing.assert_array_equal(result[0, 0], [128, 64, 32])  # bg pixel


def test_to_array_hwc_both_tuples():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=(255, 0, 0), bg_value=(0, 0, 255))
    assert result.shape == (2, 2, 3)
    np.testing.assert_array_equal(result[0, 1], [255, 0, 0])
    np.testing.assert_array_equal(result[1, 1], [0, 0, 255])


def test_to_array_hwc_numpy_array():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=np.array([10, 20, 30, 40]), bg_value=0)
    assert result.shape == (2, 2, 4)
    np.testing.assert_array_equal(result[0, 1], [10, 20, 30, 40])
    np.testing.assert_array_equal(result[0, 0], [0, 0, 0, 0])


def test_to_array_hwc_list():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=[200, 100], bg_value=[50, 25])
    assert result.shape == (2, 2, 2)
    np.testing.assert_array_equal(result[0, 1], [200, 100])
    np.testing.assert_array_equal(result[0, 0], [50, 25])


def test_to_array_hwc_full_mask():
    rle = RLEMask.ones((3, 3))
    result = rle.to_array(fg_value=(255, 128, 0), bg_value=(0, 0, 0))
    expected_pixel = [255, 128, 0]
    assert result.shape == (3, 3, 3)
    for r in range(3):
        for c in range(3):
            np.testing.assert_array_equal(result[r, c], expected_pixel)


def test_to_array_hwc_empty_mask():
    rle = RLEMask.zeros((2, 2))
    result = rle.to_array(fg_value=(255, 0, 0), bg_value=(0, 0, 255))
    assert result.shape == (2, 2, 3)
    for r in range(2):
        for c in range(2):
            np.testing.assert_array_equal(result[r, c], [0, 0, 255])


def test_to_array_hwc_float32():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(fg_value=(1.0, 0.0, 0.0), bg_value=(0.0, 0.0, 1.0), dtype=np.float32)
    assert result.shape == (2, 2, 3)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result[0, 1], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result[0, 0], [0.0, 0.0, 1.0])


def test_to_array_hwc_float64_nan_bg():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    result = rle.to_array(
        fg_value=(1.0, 2.0, 3.0), bg_value=(np.nan, np.nan, np.nan), dtype=np.float64)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result[0, 1], [1.0, 2.0, 3.0])
    assert all(np.isnan(result[0, 0]))


# --- decode_into with float arrays ---

def test_decode_into_float32_nan():
    """Write NaN into invalid pixels of a float32 array."""
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    depth = np.ones((2, 2), dtype=np.float32) * 5.0
    (~rle).decode_into(depth, fg_value=np.nan)
    assert depth[0, 1] == 5.0  # valid pixel unchanged
    assert depth[1, 0] == 5.0
    assert np.isnan(depth[0, 0])  # invalid pixel set to NaN
    assert np.isnan(depth[1, 1])


def test_decode_into_float64():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    arr = np.zeros((2, 2), dtype=np.float64)
    rle.decode_into(arr, fg_value=3.14)
    np.testing.assert_almost_equal(arr[0, 1], 3.14)
    np.testing.assert_almost_equal(arr[1, 0], 3.14)
    assert arr[0, 0] == 0.0
    assert arr[1, 1] == 0.0


def test_decode_into_float32_fg_and_bg():
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    rle = RLEMask.from_array(mask)
    arr = np.empty((2, 2), dtype=np.float32)
    rle.decode_into(arr, fg_value=1.0, bg_value=np.nan)
    assert arr[0, 1] == 1.0
    assert arr[1, 0] == 1.0
    assert np.isnan(arr[0, 0])
    assert np.isnan(arr[1, 1])
