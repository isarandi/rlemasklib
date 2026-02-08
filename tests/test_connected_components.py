import numpy as np
import pytest
from rlemasklib import RLEMask

cv2 = pytest.importorskip("cv2")
skimage_data = pytest.importorskip("skimage.data")


class TestConnectedComponentStats:
    def test_stats_basic(self):
        """Test that stats returns correct areas, bboxes, and centroids."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1  # 10x10 = 100 pixels, centroid at (14.5, 14.5)
        mask[50:60, 50:70] = 1  # 10x20 = 200 pixels, centroid at (59.5, 54.5)

        rle = RLEMask.from_array(mask)
        result = rle.connected_component_stats()

        assert result is not None
        areas, bboxes, centroids = result

        assert len(areas) == 2
        assert set(areas) == {100, 200}

        # Find which component is which by area
        idx_100 = 0 if areas[0] == 100 else 1
        idx_200 = 1 - idx_100

        # Check bboxes (x, y, w, h)
        assert list(bboxes[idx_100]) == [10, 10, 10, 10]
        assert list(bboxes[idx_200]) == [50, 50, 20, 10]

        # Check centroids (x, y)
        np.testing.assert_allclose(centroids[idx_100], [14.5, 14.5], atol=0.1)
        np.testing.assert_allclose(centroids[idx_200], [59.5, 54.5], atol=0.1)

    def test_stats_empty_mask(self):
        """Test stats on empty mask returns None."""
        rle = RLEMask.zeros((100, 100))
        result = rle.connected_component_stats()
        assert result is None

    def test_stats_single_pixel(self):
        """Test stats on single pixel component."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[5, 5] = 1

        rle = RLEMask.from_array(mask)
        result = rle.connected_component_stats()

        assert result is not None
        areas, bboxes, centroids = result

        assert len(areas) == 1
        assert areas[0] == 1
        assert list(bboxes[0]) == [5, 5, 1, 1]
        np.testing.assert_allclose(centroids[0], [5, 5], atol=0.1)

    def test_stats_with_min_size(self):
        """Test that min_size filters small components."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:12, 10:12] = 1  # 4 pixels
        mask[50:60, 50:60] = 1  # 100 pixels

        rle = RLEMask.from_array(mask)

        # Without min_size filter
        result = rle.connected_component_stats(min_size=1)
        assert result is not None
        assert len(result[0]) == 2

        # With min_size filter
        result = rle.connected_component_stats(min_size=10)
        assert result is not None
        assert len(result[0]) == 1
        assert result[0][0] == 100


class TestCountConnectedComponents:
    def test_count_basic(self):
        """Test counting connected components."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[50:60, 50:60] = 1
        mask[80:90, 80:90] = 1

        rle = RLEMask.from_array(mask)
        assert rle.count_connected_components() == 3

    def test_count_empty(self):
        """Test counting on empty mask."""
        rle = RLEMask.zeros((100, 100))
        assert rle.count_connected_components() == 0

    def test_count_with_min_size(self):
        """Test counting with min_size filter."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:12, 10:12] = 1  # 4 pixels
        mask[20:22, 20:22] = 1  # 4 pixels
        mask[50:60, 50:60] = 1  # 100 pixels

        rle = RLEMask.from_array(mask)
        assert rle.count_connected_components(min_size=1) == 3
        assert rle.count_connected_components(min_size=5) == 1
        assert rle.count_connected_components(min_size=101) == 0

    def test_count_connectivity(self):
        """Test that connectivity affects count."""
        # Diagonal pattern
        mask = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8)

        rle = RLEMask.from_array(mask)
        assert rle.count_connected_components(connectivity=4) == 5
        assert rle.count_connected_components(connectivity=8) == 1


class TestConnectedComponentsFiltered:
    def test_filter_by_area(self):
        """Test filtering components by area."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10:15] = 1  # 25 pixels
        mask[30:35, 30:35] = 1  # 25 pixels
        mask[50:70, 50:70] = 1  # 400 pixels

        rle = RLEMask.from_array(mask)

        # Get only large components
        large = rle.connected_components(filter_fn=lambda a, b, c: a > 100)
        assert len(large) == 1
        assert large[0].area() == 400

        # Get only small components
        small = rle.connected_components(filter_fn=lambda a, b, c: a < 100)
        assert len(small) == 2
        assert all(comp.area() == 25 for comp in small)

    def test_filter_by_position(self):
        """Test filtering components by centroid position."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1  # left side, centroid x ~ 14.5
        mask[10:20, 80:90] = 1  # right side, centroid x ~ 84.5

        rle = RLEMask.from_array(mask)

        # Get only left-side components
        left = rle.connected_components(
            filter_fn=lambda areas, bboxes, centroids: centroids[:, 0] < 50
        )
        assert len(left) == 1

        # Get only right-side components
        right = rle.connected_components(
            filter_fn=lambda areas, bboxes, centroids: centroids[:, 0] > 50
        )
        assert len(right) == 1

    def test_filter_by_bbox(self):
        """Test filtering components by bounding box properties."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10:50] = 1  # wide: 5x40
        mask[50:90, 50:55] = 1  # tall: 40x5

        rle = RLEMask.from_array(mask)

        # Get only wide components (width > height)
        wide = rle.connected_components(
            filter_fn=lambda a, bboxes, c: bboxes[:, 2] > bboxes[:, 3]
        )
        assert len(wide) == 1
        bbox = wide[0].bbox()
        assert bbox[2] > bbox[3]  # width > height

        # Get only tall components (height > width)
        tall = rle.connected_components(
            filter_fn=lambda a, bboxes, c: bboxes[:, 3] > bboxes[:, 2]
        )
        assert len(tall) == 1
        bbox = tall[0].bbox()
        assert bbox[3] > bbox[2]  # height > width

    def test_filter_none_selected(self):
        """Test filter that selects no components."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1

        rle = RLEMask.from_array(mask)
        result = rle.connected_components(filter_fn=lambda a, b, c: a > 10000)
        assert len(result) == 0

    def test_filter_all_selected(self):
        """Test filter that selects all components."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[50:60, 50:60] = 1

        rle = RLEMask.from_array(mask)

        # Select all
        result = rle.connected_components(filter_fn=lambda a, b, c: a > 0)
        assert len(result) == 2

        # Compare with unfiltered
        unfiltered = rle.connected_components()
        assert len(result) == len(unfiltered)

    def test_filter_with_min_size(self):
        """Test that filter_fn works together with min_size."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:12, 10:12] = 1  # 4 pixels
        mask[30:35, 30:35] = 1  # 25 pixels
        mask[50:70, 50:70] = 1  # 400 pixels

        rle = RLEMask.from_array(mask)

        # min_size filters first, then filter_fn
        result = rle.connected_components(
            min_size=10, filter_fn=lambda a, b, c: a < 100
        )
        # Should get only the 25-pixel component (4 filtered by min_size, 400 by filter_fn)
        assert len(result) == 1
        assert result[0].area() == 25

    def test_filter_empty_mask(self):
        """Test filter on empty mask."""
        rle = RLEMask.zeros((100, 100))
        result = rle.connected_components(filter_fn=lambda a, b, c: a > 0)
        assert len(result) == 0

    def test_no_filter_same_as_regular(self):
        """Test that not providing filter_fn gives same result as regular method."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[50:60, 50:60] = 1

        rle = RLEMask.from_array(mask)

        with_none = rle.connected_components(filter_fn=None)
        without = rle.connected_components()

        assert len(with_none) == len(without)
        for c1, c2 in zip(with_none, without):
            assert c1 == c2


class TestConnectedComponentsAgainstCV2:
    """Oracle tests comparing against OpenCV's connectedComponentsWithStats."""

    def test_stats_match_cv2_simple(self):
        """Test that stats match cv2 for simple shapes."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:30] = 1  # Rectangle
        mask[50:70, 50:70] = 1  # Square

        self._compare_with_cv2(mask, connectivity=4)
        self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_diagonal(self):
        """Test diagonal pattern where connectivity matters."""
        mask = np.array(
            [
                [1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1],
            ],
            dtype=np.uint8,
        )

        self._compare_with_cv2(mask, connectivity=4)
        self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_random(self):
        """Test random masks."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            mask = (rng.rand(50, 50) > 0.7).astype(np.uint8)
            self._compare_with_cv2(mask, connectivity=4)
            self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_sparse(self):
        """Test sparse masks with isolated pixels."""
        rng = np.random.RandomState(123)
        mask = np.zeros((100, 100), dtype=np.uint8)
        # Sprinkle some isolated pixels
        for _ in range(20):
            y, x = rng.randint(0, 100, 2)
            mask[y, x] = 1

        self._compare_with_cv2(mask, connectivity=4)
        self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_dense(self):
        """Test dense masks."""
        rng = np.random.RandomState(456)
        mask = (rng.rand(30, 30) > 0.3).astype(np.uint8)

        self._compare_with_cv2(mask, connectivity=4)
        self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_single_component(self):
        """Test mask with single connected component."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 1

        self._compare_with_cv2(mask, connectivity=4)

    def test_stats_match_cv2_l_shape(self):
        """Test L-shaped component."""
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[5:15, 5:8] = 1  # Vertical part
        mask[12:15, 5:15] = 1  # Horizontal part

        self._compare_with_cv2(mask, connectivity=4)

    def test_count_matches_cv2(self):
        """Test that component count matches cv2."""
        rng = np.random.RandomState(789)
        for _ in range(10):
            mask = (rng.rand(40, 40) > 0.6).astype(np.uint8)

            for connectivity in [4, 8]:
                rle = RLEMask.from_array(mask)
                rle_count = rle.count_connected_components(connectivity=connectivity)

                cv2_connectivity = 4 if connectivity == 4 else 8
                n_labels, _, _, _ = cv2.connectedComponentsWithStats(
                    mask, connectivity=cv2_connectivity
                )
                cv2_count = n_labels - 1  # Subtract background

                assert rle_count == cv2_count, (
                    f"Count mismatch: rle={rle_count}, cv2={cv2_count}, "
                    f"connectivity={connectivity}"
                )

    def test_stats_match_cv2_coins(self):
        """Test on thresholded coins image."""
        image = skimage_data.coins()
        for threshold in [50, 100, 150, 200]:
            mask = (image > threshold).astype(np.uint8)
            self._compare_with_cv2(mask, connectivity=4)
            self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_camera(self):
        """Test on thresholded camera image."""
        image = skimage_data.camera()
        for threshold in [50, 100, 150, 200]:
            mask = (image > threshold).astype(np.uint8)
            self._compare_with_cv2(mask, connectivity=4)
            self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_text(self):
        """Test on thresholded text image (many small components)."""
        image = skimage_data.text()
        for threshold in [100, 150, 200]:
            mask = (image > threshold).astype(np.uint8)
            self._compare_with_cv2(mask, connectivity=4)
            self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_binary_blobs(self):
        """Test on binary blobs image."""
        rng = np.random.default_rng(42)
        image = skimage_data.binary_blobs(length=128, rng=rng)
        mask = image.astype(np.uint8)
        self._compare_with_cv2(mask, connectivity=4)
        self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_checkerboard(self):
        """Test on checkerboard pattern."""
        image = skimage_data.checkerboard()
        mask = (image > 127).astype(np.uint8)
        self._compare_with_cv2(mask, connectivity=4)
        self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_brick(self):
        """Test on brick texture."""
        image = skimage_data.brick()
        for threshold in [100, 150, 200]:
            mask = (image > threshold).astype(np.uint8)
            self._compare_with_cv2(mask, connectivity=4)
            self._compare_with_cv2(mask, connectivity=8)

    def test_stats_match_cv2_grass(self):
        """Test on grass texture."""
        image = skimage_data.grass()
        for threshold in [80, 120, 160]:
            mask = (image > threshold).astype(np.uint8)
            self._compare_with_cv2(mask, connectivity=4)
            self._compare_with_cv2(mask, connectivity=8)

    def _compare_with_cv2(self, mask: np.ndarray, connectivity: int):
        """Compare RLEMask stats with cv2.connectedComponentsWithStats."""
        rle = RLEMask.from_array(mask)
        result = rle.connected_component_stats(connectivity=connectivity)

        cv2_connectivity = 4 if connectivity == 4 else 8
        n_labels, _, stats, centroids_cv2 = cv2.connectedComponentsWithStats(
            mask, connectivity=cv2_connectivity
        )

        # cv2 includes background as label 0, so actual components are 1..n_labels-1
        n_components_cv2 = n_labels - 1

        if n_components_cv2 == 0:
            assert result is None, "Expected None for empty mask"
            return

        assert result is not None, "Expected stats for non-empty mask"
        areas, bboxes, centroids = result

        assert (
            len(areas) == n_components_cv2
        ), f"Component count mismatch: rle={len(areas)}, cv2={n_components_cv2}"

        # Extract cv2 stats (skip background label 0)
        # cv2 stats columns: [x, y, width, height, area]
        cv2_areas = stats[1:, cv2.CC_STAT_AREA]
        cv2_bboxes = np.column_stack(
            [
                stats[1:, cv2.CC_STAT_LEFT],
                stats[1:, cv2.CC_STAT_TOP],
                stats[1:, cv2.CC_STAT_WIDTH],
                stats[1:, cv2.CC_STAT_HEIGHT],
            ]
        )
        cv2_centroids = centroids_cv2[1:]  # Skip background

        # Build lookup dict: (area, bbox) -> list of indices with that key
        cv2_lookup = {}
        for j in range(len(cv2_areas)):
            key = (cv2_areas[j], tuple(cv2_bboxes[j]))
            cv2_lookup.setdefault(key, []).append(j)

        # Match components by finding the cv2 component for each rle component
        for i in range(len(areas)):
            rle_area = areas[i]
            rle_bbox = tuple(bboxes[i])
            rle_centroid = centroids[i]

            key = (rle_area, rle_bbox)
            matches = cv2_lookup.get(key, [])

            assert (
                len(matches) >= 1
            ), f"No cv2 match for rle component {i}: area={rle_area}, bbox={rle_bbox}"

            # If multiple matches (same area and bbox), check centroid
            matched = False
            for j in matches:
                if np.allclose(rle_centroid, cv2_centroids[j], atol=1e-10):
                    matched = True
                    break

            assert matched, (
                f"Centroid mismatch for component {i}: "
                f"rle={rle_centroid}, cv2 candidates={[cv2_centroids[j] for j in matches]}"
            )


class TestConnectedComponentsConsistency:
    """Test that stats, count, and extraction are consistent with each other."""

    def test_count_matches_stats_length(self):
        """Count should match number of stats entries."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[50:60, 50:60] = 1
        mask[80:85, 80:85] = 1

        rle = RLEMask.from_array(mask)

        count = rle.count_connected_components()
        stats = rle.connected_component_stats()

        assert count == len(stats[0])

    def test_stats_areas_match_extracted_areas(self):
        """Stats areas should match areas of extracted components."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:20, 10:20] = 1
        mask[50:55, 50:60] = 1

        rle = RLEMask.from_array(mask)

        stats = rle.connected_component_stats()
        components = rle.connected_components()

        stats_areas = sorted(stats[0])
        comp_areas = sorted([c.area() for c in components])

        assert stats_areas == comp_areas

    def test_filtered_count_matches_filtered_extraction(self):
        """Filtered extraction count should match what filter would select."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:15, 10:15] = 1  # 25 pixels
        mask[50:70, 50:70] = 1  # 400 pixels

        rle = RLEMask.from_array(mask)

        # Get stats and manually count large ones
        stats = rle.connected_component_stats()
        expected_count = np.sum(stats[0] > 100)

        # Get filtered components
        filtered = rle.connected_components(filter_fn=lambda a, b, c: a > 100)

        assert len(filtered) == expected_count
