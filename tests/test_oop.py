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
    import imageio.v2 as imageio

    rle = RLEMask.from_circle([499 / 2, 499 / 2], 199, imshape=(500, 500))

    import cv2

    poly = cv2.ellipse2Poly((499 // 2, 499 // 2), (199, 199), 0, 0, 360, 1)
    rle2 = RLEMask.from_polygon(poly, imshape=(500, 500))

    mask = np.array(rle2 - rle)
    imageio.imwrite("/tmp/circle.png", mask * 255)


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
