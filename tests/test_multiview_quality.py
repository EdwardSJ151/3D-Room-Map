import unittest

import numpy as np

from cutr_api import (
    MIN_RGBD_COVERAGE,
    MIN_VIEW_TRANSLATION_M,
    _make_intrinsics,
    _view_quality,
)


def metadata(position=(0.0, 0.0, 0.0)):
    return {
        "fx": 200.0,
        "fy": 150.0,
        "cx": 100.0,
        "cy": 75.0,
        "width": 400,
        "height": 300,
        "pose_R_wc": np.eye(3).tolist(),
        "pose_t_wc": list(position),
    }


class MultiviewQualityTests(unittest.TestCase):
    def test_intrinsics_are_scaled_when_encoded_image_changes_size(self):
        def defaults(width, height):
            return np.eye(3)

        matrix = _make_intrinsics(
            {"make_default_intrinsics": defaults}, metadata(), 800, 600
        )

        self.assertEqual(matrix[0, 0], 400.0)
        self.assertEqual(matrix[1, 1], 300.0)
        self.assertEqual(matrix[0, 2], 200.0)
        self.assertEqual(matrix[1, 2], 150.0)

    def test_quality_requires_metric_coverage_and_view_diversity(self):
        valid_depth = np.full((100, 100), 2000, dtype=np.uint16)
        accepted = _view_quality(metadata(), valid_depth, [])
        duplicate = _view_quality(metadata(), valid_depth, [metadata()])
        moved = _view_quality(
            metadata((MIN_VIEW_TRANSLATION_M, 0.0, 0.0)), valid_depth, [metadata()]
        )
        sparse = _view_quality(metadata(), np.zeros((100, 100), dtype=np.uint16), [])

        self.assertTrue(accepted.accepted)
        self.assertFalse(duplicate.accepted)
        self.assertIn("similar", duplicate.guidance)
        self.assertTrue(moved.accepted)
        self.assertFalse(sparse.accepted)
        self.assertLess(sparse.depth_coverage_ratio, MIN_RGBD_COVERAGE)


if __name__ == "__main__":
    unittest.main()
