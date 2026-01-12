"""Test cases for documentation examples.

Each test verifies that the mask-demo examples in docstrings are correct.
Run with: pytest tests/test_for_docs.py -v
"""

import numpy as np
from rlemasklib import RLEMask


def mask_from_str(s):
    """Convert ASCII art string to RLEMask."""
    lines = [line for line in s.strip().split("\n") if line.strip()]
    rows = []
    for line in lines:
        row = []
        for char in line:
            if char in "#X1*":
                row.append(1)
            elif char in ".0":
                row.append(0)
        if row:
            rows.append(row)
    return RLEMask.from_array(np.array(rows, dtype=np.uint8))


def assert_mask_equal(mask, expected_str):
    """Assert that mask equals the expected ASCII art."""
    expected = mask_from_str(expected_str)
    assert (
        mask == expected
    ), f"\nGot:\n{np.array(mask)}\nExpected:\n{np.array(expected)}"


# =============================================================================
# Boolean Operations (magic methods)
# =============================================================================


class TestBooleanOps:
    def test_or_union(self):
        """Union combines both shapes."""
        A = mask_from_str("""
            ####....
            ####....
            ####....
            ........
        """)
        B = mask_from_str("""
            ..####..
            ..####..
            ..####..
            ..####..
        """)
        assert_mask_equal(
            A | B,
            """
            ######..
            ######..
            ######..
            ..####..
        """,
        )

    def test_and_intersection(self):
        """Intersection keeps only the overlap."""
        A = mask_from_str("""
            ####....
            ####....
            ####....
            ........
        """)
        B = mask_from_str("""
            ..####..
            ..####..
            ..####..
            ..####..
        """)
        assert_mask_equal(
            A & B,
            """
            ..##....
            ..##....
            ..##....
            ........
        """,
        )

    def test_sub_difference(self):
        """Difference removes B from A."""
        A = mask_from_str("""
            ####....
            ####....
            ####....
            ........
        """)
        B = mask_from_str("""
            ..####..
            ..####..
            ..####..
            ..####..
        """)
        assert_mask_equal(
            A - B,
            """
            ##......
            ##......
            ##......
            ........
        """,
        )

    def test_xor_symmetric_difference(self):
        """XOR keeps only non-overlapping parts."""
        A = mask_from_str("""
            ####....
            ####....
            ####....
            ........
        """)
        B = mask_from_str("""
            ..####..
            ..####..
            ..####..
            ..####..
        """)
        assert_mask_equal(
            A ^ B,
            """
            ##..##..
            ##..##..
            ##..##..
            ..####..
        """,
        )


class TestInvert:
    def test_invert(self):
        """~ operator inverts the mask (same as complement)."""
        # Use an interesting shape - a ring
        A = mask_from_str("""
            ..####..
            .######.
            ##....##
            ##....##
            .######.
            ..####..
        """)
        assert_mask_equal(
            ~A,
            """
            ##....##
            #......#
            ..####..
            ..####..
            #......#
            ##....##
        """,
        )


# =============================================================================
# Geometric Transformations
# =============================================================================


class TestShift:
    def test_shift_asymmetric(self):
        """Shift with different x and y offsets."""
        A = mask_from_str("""
            ###.....
            ###.....
            ###.....
            ........
            ........
        """)
        result = A.shift((1, 3))  # dy=1, dx=3
        assert_mask_equal(
            result,
            """
            ........
            ...###..
            ...###..
            ...###..
            ........
        """,
        )


class TestPad:
    def test_pad_asymmetric(self):
        """Pad with different amounts on each side."""
        A = mask_from_str("""
            ##
            ##
        """)
        result = A.pad(top=1, bottom=2, left=1, right=3, value=0)
        assert_mask_equal(
            result,
            """
            ......
            .##...
            .##...
            ......
            ......
        """,
        )


class TestCrop:
    def test_crop_partial(self):
        """Crop that includes some background."""
        A = mask_from_str("""
            ..........
            ..####....
            ..####....
            ..####....
            ..........
            ..........
        """)
        result = A.crop([1, 0, 6, 5])  # x=1, y=0, w=6, h=5
        assert_mask_equal(
            result,
            """
            ......
            .####.
            .####.
            .####.
            ......
        """,
        )


class TestTranspose:
    def test_transpose_complex(self):
        """Transpose an L-shaped figure."""
        A = mask_from_str("""
            ##...
            ##...
            ##...
            #####
            #####
        """)
        assert_mask_equal(
            A.T,
            """
            #####
            #####
            ...##
            ...##
            ...##
        """,
        )


class TestRot90:
    def test_rot90_complex(self):
        """90 degree counter-clockwise rotation of arrow shape."""
        A = mask_from_str("""
            ..#..
            .###.
            #####
            ..#..
            ..#..
        """)
        assert_mask_equal(
            A.rot90(1),
            """
            ..#..
            .##..
            #####
            .##..
            ..#..
        """,
        )

    def test_rot90_180(self):
        """180 degree rotation."""
        A = mask_from_str("""
            ##...
            ##...
            ##...
            #####
            #####
        """)
        assert_mask_equal(
            A.rot90(2),
            """
            #####
            #####
            ...##
            ...##
            ...##
        """,
        )


class TestFlip:
    def test_flip_axis0(self):
        """flip(axis=0) is vertical flip (same as flipud)."""
        A = mask_from_str("""
            #####
            ##...
            #....
        """)
        assert_mask_equal(
            A.flip(axis=0),
            """
            #....
            ##...
            #####
        """,
        )

    def test_flip_axis1(self):
        """flip(axis=1) is horizontal flip (same as fliplr)."""
        A = mask_from_str("""
            #####
            ##...
            #....
        """)
        assert_mask_equal(
            A.flip(axis=1),
            """
            #####
            ...##
            ....#
        """,
        )

    def test_flipud_complex(self):
        """Vertical flip of arrow shape."""
        A = mask_from_str("""
            ..#..
            .###.
            #####
            ..#..
            ..#..
        """)
        assert_mask_equal(
            A.flipud(),
            """
            ..#..
            ..#..
            #####
            .###.
            ..#..
        """,
        )

    def test_fliplr_complex(self):
        """Horizontal flip of L shape."""
        A = mask_from_str("""
            ##...
            ##...
            ##...
            #####
            #####
        """)
        assert_mask_equal(
            A.fliplr(),
            """
            ...##
            ...##
            ...##
            #####
            #####
        """,
        )


# =============================================================================
# Connected Components
# =============================================================================


class TestConnectedComponents:
    def test_connected_components_4(self):
        """With 4-connectivity, diagonal pixels are separate components."""
        A = mask_from_str("""
            #.#.#
            .#.#.
            #.#.#
        """)
        components = A.connected_components(connectivity=4)
        assert len(components) == 8  # Each pixel is separate with 4-connectivity

    def test_connected_components_8(self):
        """With 8-connectivity, diagonal pixels connect."""
        A = mask_from_str("""
            #.#.#
            .#.#.
            #.#.#
        """)
        components = A.connected_components(connectivity=8)
        assert len(components) == 1  # All connected diagonally


class TestLargestConnectedComponent:
    def test_largest_component_4(self):
        """Keep largest with 4-connectivity."""
        A = mask_from_str("""
            ##.....##
            ##.....##
            .........
            ...###...
            ...###...
            ...###...
        """)
        result = A.largest_connected_component(connectivity=4)
        assert_mask_equal(
            result,
            """
            .........
            .........
            .........
            ...###...
            ...###...
            ...###...
        """,
        )

    def test_largest_component_8(self):
        """With 8-connectivity, diagonal bridge connects regions."""
        A = mask_from_str("""
            ###......
            ###......
            ..#......
            ...#.....
            ....#....
            .....####
            .....####
        """)
        # All connected via diagonal, so whole thing is one component
        result = A.largest_connected_component(connectivity=8)
        assert result.area() == A.area()


class TestRemoveSmallComponents:
    def test_remove_small_4(self):
        """Remove small isolated pixels with 4-connectivity."""
        A = mask_from_str("""
            #.......#
            .........
            ..#####..
            ..#####..
            ..#####..
            .........
            #.......#
        """)
        result = A.remove_small_components(min_size=4, connectivity=4)
        assert_mask_equal(
            result,
            """
            .........
            .........
            ..#####..
            ..#####..
            ..#####..
            .........
            .........
        """,
        )

    def test_remove_small_8(self):
        """Diagonal pixels form one component with 8-connectivity."""
        A = mask_from_str("""
            #.....
            .#....
            ..#...
            ...#..
            ....##
        """)
        result = A.remove_small_components(min_size=4, connectivity=8)
        # With 8-conn, all 6 pixels are one component
        assert result.area() == 6


class TestFillSmallHoles:
    def test_fill_holes_4(self):
        """Fill small holes, 4-connectivity."""
        A = mask_from_str("""
            #######
            #.....#
            #.###.#
            #.#.#.#
            #.###.#
            #.....#
            #######
        """)
        # The single center hole is 1 pixel
        result = A.fill_small_holes(min_size=2, connectivity=4)
        assert_mask_equal(
            result,
            """
            #######
            #.....#
            #.###.#
            #.###.#
            #.###.#
            #.....#
            #######
        """,
        )

    def test_fill_holes_8(self):
        """Hole size determines if it gets filled."""
        A = mask_from_str("""
            ####
            #..#
            #..#
            ####
        """)
        # 4-pixel hole, min_size=5 means fill holes with size < 5
        result = A.fill_small_holes(min_size=5, connectivity=4)
        assert result.area() == 16  # 4-pixel hole is filled (4 < 5)

        # But with min_size=4, hole is NOT filled (4 is not < 4)
        result = A.fill_small_holes(min_size=4, connectivity=4)
        assert result == A  # Hole not filled


# =============================================================================
# Morphological Operations
# =============================================================================


class TestDilate:
    def test_dilate_circle_7(self):
        """Dilate single pixel with circular kernel."""
        A = mask_from_str("""
            .........
            .........
            .........
            .........
            ....#....
            .........
            .........
            .........
            .........
        """)
        result = A.dilate(kernel_shape="circle", kernel_size=7)
        # Circle with radius 3 around center
        assert result.area() > 30  # Rough check for circular shape
        # Center should be set
        assert np.array(result)[4, 4] == 1

    def test_dilate_square_5(self):
        """Dilate with square kernel."""
        A = mask_from_str("""
            .......
            .......
            .......
            ...#...
            .......
            .......
            .......
        """)
        result = A.dilate(kernel_shape="square", kernel_size=5)
        assert_mask_equal(
            result,
            """
            .......
            .#####.
            .#####.
            .#####.
            .#####.
            .#####.
            .......
        """,
        )


class TestErode:
    def test_erode_circle_5(self):
        """Erode with circular kernel."""
        A = mask_from_str("""
            ...........
            ..#######..
            .#########.
            .#########.
            .#########.
            .#########.
            .#########.
            ..#######..
            ...........
        """)
        result = A.erode(kernel_shape="circle", kernel_size=5)
        # Should shrink the shape
        assert result.area() < A.area()
        # Center should still be set
        assert np.array(result)[4, 5] == 1


class TestDilate3x3:
    def test_dilate3x3_conn4(self):
        """Single pixel dilates to cross shape."""
        A = mask_from_str("""
            .....
            .....
            ..#..
            .....
            .....
        """)
        assert_mask_equal(
            A.dilate3x3(connectivity=4),
            """
            .....
            ..#..
            .###.
            ..#..
            .....
        """,
        )

    def test_dilate3x3_conn8(self):
        """Single pixel dilates to 3x3 square."""
        A = mask_from_str("""
            .....
            .....
            ..#..
            .....
            .....
        """)
        assert_mask_equal(
            A.dilate3x3(connectivity=8),
            """
            .....
            .###.
            .###.
            .###.
            .....
        """,
        )


class TestErode3x3:
    def test_erode3x3_conn4_large(self):
        """Erosion of shape with protrusion shows connectivity difference."""
        A = mask_from_str("""
            .........
            ...###...
            ...###...
            .#####...
            .#####...
            ...###...
            ...###...
            .........
        """)
        assert_mask_equal(
            A.erode3x3(connectivity=4),
            """
            .........
            .........
            ....#....
            ...##....
            ...##....
            ....#....
            .........
            .........
        """,
        )

    def test_erode3x3_conn8_large(self):
        """With 8-connectivity, diagonal neighbors needed too."""
        A = mask_from_str("""
            .........
            ...###...
            ...###...
            .#####...
            .#####...
            ...###...
            ...###...
            .........
        """)
        assert_mask_equal(
            A.erode3x3(connectivity=8),
            """
            .........
            .........
            ....#....
            ....#....
            ....#....
            ....#....
            .........
            .........
        """,
        )

    def test_erode3x3_border(self):
        """Border pixels preserved with replicate padding."""
        A = mask_from_str("""
            ###...
            ###...
            ###...
        """)
        assert_mask_equal(
            A.erode3x3(connectivity=4),
            """
            ##....
            ##....
            ##....
        """,
        )


class TestDilate5x5:
    def test_dilate5x5(self):
        """Single pixel dilates to rounded 5x5 shape."""
        A = mask_from_str("""
            .......
            .......
            .......
            ...#...
            .......
            .......
            .......
        """)
        assert_mask_equal(
            A.dilate5x5(),
            """
            .......
            ..###..
            .#####.
            .#####.
            .#####.
            ..###..
            .......
        """,
        )


class TestErode5x5:
    def test_erode5x5(self):
        """Erosion shrinks by 2 pixels in each direction."""
        A = mask_from_str("""
            ...........
            ..#######..
            .#########.
            .#########.
            .#########.
            .#########.
            .#########.
            .#########.
            .#########.
            ..#######..
            ...........
        """)
        assert_mask_equal(
            A.erode5x5(),
            """
            ...........
            ...........
            ...........
            ...#####...
            ...#####...
            ...#####...
            ...#####...
            ...#####...
            ...........
            ...........
            ...........
        """,
        )


class TestContours:
    def test_contours_donut(self):
        """Contours of a donut shape."""
        A = mask_from_str("""
            .........
            ..#####..
            .#######.
            .###.###.
            .##...##.
            .###.###.
            .#######.
            ..#####..
            .........
        """)
        result = A.contours()
        # Contour should have outer and inner edge
        assert result.area() < A.area()
        # Center hole should create inner contour
        arr = np.array(result)
        assert arr[4, 4] == 0  # Center of hole stays empty


class TestDilateVertical:
    def test_dilate_vertical_asymmetric(self):
        """Asymmetric vertical dilation."""
        A = mask_from_str("""
            ........
            ........
            ........
            .######.
            ........
            ........
            ........
            ........
        """)
        result = A.dilate_vertical(up=1, down=3)
        assert_mask_equal(
            result,
            """
            ........
            ........
            .######.
            .######.
            .######.
            .######.
            .######.
            ........
        """,
        )


# =============================================================================
# Size and Shape Operations
# =============================================================================


class TestRepeat:
    def test_repeat(self):
        """Repeat upscales by repeating pixels."""
        A = mask_from_str("""
            #.
            .#
        """)
        result = A.repeat(2, 3)
        assert_mask_equal(
            result,
            """
            ###...
            ###...
            ...###
            ...###
        """,
        )


class TestTile:
    def test_tile(self):
        """Tile repeats the whole mask."""
        A = mask_from_str("""
            #.
            .#
        """)
        result = A.tile(2, 3)
        assert_mask_equal(
            result,
            """
            #.#.#.
            .#.#.#
            #.#.#.
            .#.#.#
        """,
        )


class TestConcat:
    def test_hconcat(self):
        """Horizontal concatenation."""
        A = mask_from_str("""
            ##.
            ##.
            ##.
        """)
        B = mask_from_str("""
            .#
            #.
            .#
        """)
        result = RLEMask.hconcat([A, B])
        assert_mask_equal(
            result,
            """
            ##..#
            ##.#.
            ##..#
        """,
        )

    def test_vconcat(self):
        """Vertical concatenation."""
        A = mask_from_str("""
            ###
            ###
        """)
        B = mask_from_str("""
            #.#
            .#.
        """)
        result = RLEMask.vconcat([A, B])
        assert_mask_equal(
            result,
            """
            ###
            ###
            #.#
            .#.
        """,
        )


# =============================================================================
# Bounding Box and Interior Rectangles
# =============================================================================


class TestBbox:
    def test_bbox(self):
        """Bounding box of a shape."""
        A = mask_from_str("""
            ..........
            ...####...
            ...####...
            ...####...
            ..........
        """)
        bbox = A.bbox()
        np.testing.assert_array_equal(bbox, [3, 1, 4, 3])


class TestTightCrop:
    def test_tight_crop(self):
        """Tight crop removes surrounding background."""
        A = mask_from_str("""
            ..........
            ...###....
            ...#.#....
            ...###....
            ..........
        """)
        cropped, bbox = A.tight_crop()
        assert_mask_equal(
            cropped,
            """
            ###
            #.#
            ###
        """,
        )
        np.testing.assert_array_equal(bbox, [3, 1, 3, 3])


class TestLargestInteriorRectangle:
    def test_largest_interior_rect(self):
        """Find largest rectangle inside L-shape."""
        A = mask_from_str("""
            ..........
            .#####....
            .#####....
            .#####....
            .###......
            .###......
            ..........
        """)
        rect = A.largest_interior_rectangle()
        # rect is [x, y, w, h]
        assert rect[2] * rect[3] > 0  # Non-empty
        # Check it's inside the shape
        x, y, w, h = rect.astype(int)
        arr = np.array(A)
        assert arr[y : y + h, x : x + w].all()

    def test_largest_interior_rect_aspect(self):
        """Find largest rectangle with specific aspect ratio."""
        A = mask_from_str("""
            .............
            .###########.
            .###########.
            .###########.
            .###########.
            .............
        """)
        rect = A.largest_interior_rectangle(aspect_ratio=2.0)
        # Width should be ~2x height
        assert abs(rect[2] / rect[3] - 2.0) < 0.1

    def test_largest_interior_rect_around(self):
        """Find largest rectangle around a specific center point."""
        # L-shaped mask - rectangle around different points gives different results
        A = mask_from_str("""
            ..........
            .#######..
            .#######..
            .#######..
            .###......
            .###......
            ..........
        """)
        # Get rect centered at (4, 3) - in the wide upper part
        rect = A.largest_interior_rectangle_around([4, 3])
        cx = rect[0] + (rect[2] - 1) / 2
        cy = rect[1] + (rect[3] - 1) / 2
        assert abs(cx - 4) < 0.5
        assert abs(cy - 3) < 0.5
        # Check it's inside the mask
        x, y, w, h = rect.astype(int)
        arr = np.array(A)
        assert arr[y : y + h, x : x + w].all()

    def test_largest_interior_rect_around_aspect(self):
        """Find largest rectangle with aspect ratio around center."""
        A = mask_from_str("""
            .............
            .###########.
            .###########.
            .###########.
            .###########.
            .###########.
            .............
        """)
        # Get 2:1 aspect ratio rectangle centered at (6, 3)
        rect = A.largest_interior_rectangle_around([6, 3], aspect_ratio=2.0)
        # Width should be ~2x height
        assert abs(rect[2] / rect[3] - 2.0) < 0.2


# =============================================================================
# Fill Operations
# =============================================================================


class TestFillRectangle:
    def test_fill_rectangle(self):
        """Fill a rectangle in the mask."""
        A = mask_from_str("""
            ........
            ........
            ........
            ........
        """)
        result = A.fill_rectangle([2, 1, 4, 2], value=1)
        assert_mask_equal(
            result,
            """
            ........
            ..####..
            ..####..
            ........
        """,
        )

    def test_fill_rectangle_clear(self):
        """Clear a rectangle in the mask."""
        A = mask_from_str("""
            ########
            ########
            ########
            ########
        """)
        result = A.fill_rectangle([2, 1, 4, 2], value=0)
        assert_mask_equal(
            result,
            """
            ########
            ##....##
            ##....##
            ########
        """,
        )


class TestFillCircle:
    def test_fill_circle_on_pattern(self):
        """Fill a circle on an existing pattern."""
        # Start with a checkerboard
        A = mask_from_str("""
            #.#.#.#.#
            .#.#.#.#.
            #.#.#.#.#
            .#.#.#.#.
            #.#.#.#.#
            .#.#.#.#.
            #.#.#.#.#
            .#.#.#.#.
            #.#.#.#.#
        """)
        result = A.fill_circle([4, 4], radius=3, value=1)
        arr = np.array(result)
        # Circle area should all be 1 now
        assert arr[4, 4] == 1  # Center
        assert arr[4, 3] == 1  # Inside circle, was 0 in checkerboard
        assert arr[4, 5] == 1  # Inside circle
        # Corners should preserve checkerboard
        assert arr[0, 0] == 1
        assert arr[0, 1] == 0

    def test_fill_circle_clear(self):
        """Clear a circle from a filled mask."""
        A = RLEMask.ones((9, 9))
        result = A.fill_circle([4, 4], radius=2, value=0)
        arr = np.array(result)
        # Circle should be 0
        assert arr[4, 4] == 0  # Center
        # Corners should still be 1
        assert arr[0, 0] == 1
        assert arr[8, 8] == 1


# =============================================================================
# Pooling and Convolution
# =============================================================================


class TestMaxPool:
    def test_max_pool2x2(self):
        """Max pooling - any 1 in 2x2 block produces 1."""
        A = mask_from_str("""
            #...
            ....
            ..#.
            ....
        """)
        result = A.max_pool2x2()
        assert_mask_equal(
            result,
            """
            #.
            .#
        """,
        )


class TestMinPool:
    def test_min_pool2x2(self):
        """Min pooling - all 1s in 2x2 block required for 1."""
        A = mask_from_str("""
            ##..
            ##..
            ..##
            ..#.
        """)
        result = A.min_pool2x2()
        assert_mask_equal(
            result,
            """
            #.
            ..
        """,
        )


class TestAvgPool:
    def test_avg_pool2x2(self):
        """Average pooling with threshold at 50%."""
        A = mask_from_str("""
            ##..
            #...
            ..##
            ..##
        """)
        result = A.avg_pool2x2()
        # First 2x2: 3/4 > 0.5 -> 1
        # Second 2x2: 0/4 -> 0
        # Third 2x2: 0/4 -> 0
        # Fourth 2x2: 4/4 -> 1
        assert_mask_equal(
            result,
            """
            #.
            .#
        """,
        )


class TestAvgPool2dValid:
    def test_avg_pool2d_valid_3x3_stride2(self):
        """Average pooling with 3x3 kernel and stride 2."""
        A = mask_from_str("""
            ########
            ########
            ########
            ....####
            ....####
            ....####
        """)
        # 3x3 kernel, stride 2, threshold default (majority)
        result = A.avg_pool2d_valid(kernel_size=(3, 3), stride=(2, 2))
        # Output shape: (6-3)//2+1 = 2, (8-3)//2+1 = 3
        assert result.shape == (2, 3)
        arr = np.array(result)
        # Top-left 3x3 all 1s -> 1
        assert arr[0, 0] == 1
        # Top-right 3x3 all 1s -> 1
        assert arr[0, 2] == 1

    def test_avg_pool2d_valid_threshold(self):
        """Custom threshold for pooling."""
        A = mask_from_str("""
            ##..
            #...
            ....
            ....
        """)
        # With high threshold, need more 1s to pass
        result_high = A.avg_pool2d_valid(kernel_size=(2, 2), stride=(2, 2), threshold=4)
        result_low = A.avg_pool2d_valid(kernel_size=(2, 2), stride=(2, 2), threshold=1)
        # High threshold: need all 4 pixels, top-left has only 3
        assert np.array(result_high)[0, 0] == 0
        # Low threshold: need only 1 pixel
        assert np.array(result_low)[0, 0] == 1


class TestConv2d:
    def test_conv2d_edge_detect(self):
        """Convolution with edge detection kernel."""
        A = mask_from_str("""
            .......
            .#####.
            .#####.
            .#####.
            .#####.
            .#####.
            .......
        """)
        # Sobel-like vertical edge kernel
        kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        result = A.conv2d_valid(kernel, threshold=0)
        # Should detect vertical edges
        assert result.area() > 0


# =============================================================================
# Resize and Warp
# =============================================================================


class TestResize:
    def test_resize_up(self):
        """Upscale an interesting shape by 2x."""
        # Checkerboard pattern
        A = mask_from_str("""
            #.#
            .#.
            #.#
        """)
        result = A.resize(output_imshape=(6, 6))
        assert result.shape == (6, 6)
        # Each pixel becomes 2x2 block
        arr = np.array(result)
        assert arr[0, 0] == 1  # Top-left was 1
        assert arr[0, 2] == 0  # Was 0
        assert arr[2, 2] == 1  # Center was 1

    def test_resize_down(self):
        """Downscale a pattern."""
        # Larger checkerboard
        A = mask_from_str("""
            ##..##
            ##..##
            ..##..
            ..##..
            ##..##
            ##..##
        """)
        result = A.resize(output_imshape=(3, 3))
        assert result.shape == (3, 3)
        # Should preserve checkerboard pattern
        arr = np.array(result)
        assert arr[0, 0] == 1
        assert arr[1, 1] == 1
        assert arr[0, 1] == 0


class TestWarpAffine:
    def test_warp_affine_translate(self):
        """Affine warp - translation."""
        A = mask_from_str("""
            ##....
            ##....
            ......
            ......
        """)
        # Translation matrix: move right by 2, down by 1
        M = np.array([[1, 0, 2], [0, 1, 1]], dtype=np.float32)
        result = A.warp_affine(M, output_imshape=(4, 6))
        assert_mask_equal(
            result,
            """
            ......
            ..##..
            ..##..
            ......
        """,
        )

    def test_warp_affine_scale(self):
        """Affine warp - scaling."""
        A = mask_from_str("""
            ##
            ##
        """)
        # Scale 2x
        M = np.array([[2, 0, 0], [0, 2, 0]], dtype=np.float32)
        result = A.warp_affine(M, output_imshape=(4, 4))
        assert result.shape == (4, 4)
        assert result.area() == 16


class TestWarpPerspective:
    def test_warp_perspective_scale_translate(self):
        """Perspective warp with scaling and translation."""
        A = mask_from_str("""
            ##
            ##
        """)
        # Scale 2x and translate by (2, 2)
        H = np.array([[2, 0, 2], [0, 2, 2], [0, 0, 1]], dtype=np.float32)
        result = A.warp_perspective(H, output_imshape=(8, 8))
        assert result.shape == (8, 8)
        arr = np.array(result)
        # Original (0,0) maps to (2,2)
        assert arr[2, 2] == 1
        assert arr[3, 3] == 1
        # Check corners are empty
        assert arr[0, 0] == 0
        assert arr[7, 7] == 0

    def test_warp_perspective_rotation(self):
        """Perspective warp with 90 degree rotation."""
        A = mask_from_str("""
            ###.
            ###.
            ....
            ....
        """)
        # 90 degree rotation around center (2, 2)
        # Rotate then translate to keep in frame
        import math

        theta = math.pi / 2  # 90 degrees
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # Rotation matrix around origin, then translate
        H = np.array(
            [[cos_t, -sin_t, 3], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32
        )
        result = A.warp_perspective(H, output_imshape=(4, 4))
        # Shape should be rotated
        assert result.area() > 0


# =============================================================================
# Merge Functions
# =============================================================================


class TestMergeCount:
    def test_merge_count_majority(self):
        """Majority vote across masks."""
        A = mask_from_str("""
            ####
            ####
            ....
            ....
        """)
        B = mask_from_str("""
            ##..
            ##..
            ##..
            ##..
        """)
        C = mask_from_str("""
            ....
            ####
            ####
            ....
        """)
        # Threshold 2 means at least 2 of 3 must agree
        result = RLEMask.merge_count([A, B, C], threshold=2)
        assert_mask_equal(
            result,
            """
            ##..
            ####
            ##..
            ....
        """,
        )


class TestMerge:
    def test_merge_custom_func(self):
        """Merge with custom boolean function."""
        from rlemasklib import BoolFunc

        A = mask_from_str("""
            ##..
            ##..
        """)
        B = mask_from_str("""
            .##.
            .##.
        """)
        # XOR
        result = A.merge(B, BoolFunc.XOR)
        assert_mask_equal(
            result,
            """
            #.#.
            #.#.
        """,
        )


class TestMergeMany:
    def test_merge_many_or(self):
        """Merge many with OR - chains operation across all masks."""
        from rlemasklib import BoolFunc

        A = mask_from_str("""
            ##......
            ##......
        """)
        B = mask_from_str("""
            ..##....
            ..##....
        """)
        C = mask_from_str("""
            ....##..
            ....##..
        """)
        D = mask_from_str("""
            ......##
            ......##
        """)
        result = RLEMask.merge_many([A, B, C, D], BoolFunc.OR)
        assert_mask_equal(
            result,
            """
            ########
            ########
        """,
        )

    def test_merge_many_and(self):
        """Merge many with AND - only full overlap survives."""
        from rlemasklib import BoolFunc

        A = mask_from_str("""
            ######..
            ######..
        """)
        B = mask_from_str("""
            ..######
            ..######
        """)
        C = mask_from_str("""
            ...####.
            ...####.
        """)
        result = RLEMask.merge_many([A, B, C], BoolFunc.AND)
        assert_mask_equal(
            result,
            """
            ...###..
            ...###..
        """,
        )


class TestMergeManyCustom:
    def test_merge_many_custom(self):
        """Merge with custom n-ary function."""
        A = mask_from_str("""
            ##..
            ##..
        """)
        B = mask_from_str("""
            .##.
            .##.
        """)
        C = mask_from_str("""
            ..##
            ..##
        """)
        # (A | B) & ~C
        result = RLEMask.merge_many_custom(
            [A, B, C], lambda a, b, c: (a or b) and not c
        )
        assert_mask_equal(
            result,
            """
            ##..
            ##..
        """,
        )


class TestMakeMergeFunction:
    def test_make_merge_function(self):
        """Create reusable merge function."""
        # Exactly one of three
        exactly_one = RLEMask.make_merge_function(
            lambda a, b, c: sum([a, b, c]) == 1, arity=3
        )
        A = mask_from_str("""
            ##..
            ##..
        """)
        B = mask_from_str("""
            .##.
            .##.
        """)
        C = mask_from_str("""
            ..##
            ..##
        """)
        result = exactly_one(A, B, C)
        assert_mask_equal(
            result,
            """
            #..#
            #..#
        """,
        )


# =============================================================================
# Static Constructors
# =============================================================================


class TestFromBbox:
    def test_from_bbox(self):
        """Create mask from bounding box."""
        mask = RLEMask.from_bbox([2, 1, 4, 3], imshape=(6, 8))
        assert_mask_equal(
            mask,
            """
            ........
            ..####..
            ..####..
            ..####..
            ........
            ........
        """,
        )


class TestFromCircle:
    def test_from_circle(self):
        """Create circular mask."""
        mask = RLEMask.from_circle([4, 4], radius=3, imshape=(9, 9))
        # Check it's roughly circular
        arr = np.array(mask)
        assert arr[4, 4] == 1  # Center
        assert arr[4, 1] == 0  # Outside left
        assert arr[4, 7] == 0  # Outside right
        assert arr[4, 2] == 1  # Inside


class TestFromPolygon:
    def test_from_polygon_triangle(self):
        """Create mask from triangle polygon."""
        # Triangle with vertices at (1,5), (4,1), (7,5) - xy coords
        poly = np.array([[1, 5], [4, 1], [7, 5]], dtype=np.float32)
        mask = RLEMask.from_polygon(poly, imshape=(7, 9))
        arr = np.array(mask)
        # Check some points inside triangle
        assert arr[3, 4] == 1  # Center area
        assert arr[4, 4] == 1  # Lower center (safely inside)
        # Check corners outside
        assert arr[0, 0] == 0
        assert arr[0, 8] == 0


class TestIntersectionStatic:
    def test_intersection_static(self):
        """Static intersection of multiple masks - only overlap survives."""
        A = mask_from_str("""
            ######..
            ######..
            ######..
            ........
        """)
        B = mask_from_str("""
            ..######
            ..######
            ..######
            ........
        """)
        C = mask_from_str("""
            ....####
            ....####
            ....####
            ....####
        """)
        result = RLEMask.intersection([A, B, C])
        assert_mask_equal(
            result,
            """
            ....##..
            ....##..
            ....##..
            ........
        """,
        )


class TestUnionStatic:
    def test_union_static(self):
        """Static union of multiple masks - combines all."""
        A = mask_from_str("""
            ##......
            ##......
            ........
            ........
        """)
        B = mask_from_str("""
            ........
            ...##...
            ...##...
            ........
        """)
        C = mask_from_str("""
            ........
            ........
            ......##
            ......##
        """)
        result = RLEMask.union([A, B, C])
        assert_mask_equal(
            result,
            """
            ##......
            ##.##...
            ...##.##
            ......##
        """,
        )
