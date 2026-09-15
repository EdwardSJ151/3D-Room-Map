import math
import unittest

import numpy as np

from multiview import (
    Cluster,
    Observation,
    build_fused_prediction,
    depth_coverage_ratio,
    fuse_observations,
    localize_detection_from_depth,
    predictions_to_world,
    scaled_intrinsics,
)


def rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )


def rotation_y(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ]
    )


def vertical_tilt(rotation: np.ndarray) -> float:
    alignment = np.max(np.abs(rotation.T @ np.asarray([0.0, 1.0, 0.0])))
    return math.degrees(math.acos(float(np.clip(alignment, -1.0, 1.0))))


def yaw_line_difference(a: np.ndarray, b: np.ndarray) -> float:
    yaw_a = math.atan2(float(a[2, 0]), float(a[0, 0]))
    yaw_b = math.atan2(float(b[2, 0]), float(b[0, 0]))
    delta = (yaw_a - yaw_b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return abs(math.degrees(delta))


def observation(
    frame_id: str,
    rotation: np.ndarray,
    *,
    center=(0.0, 1.0, 2.0),
    dims=(1.8, 2.0, 0.5),
    score=0.9,
    bbox=(100.0, 100.0, 700.0, 700.0),
) -> Observation:
    return Observation(
        frame_id=frame_id,
        detection={"bbox_xyxy": list(bbox), "score": score, "class_id": 1},
        center=np.asarray(center, dtype=np.float64),
        dims=np.asarray(dims, dtype=np.float64),
        rotation=np.asarray(rotation, dtype=np.float64),
        score=score,
        image_size=(1000, 1000),
        raw_tilt_deg=vertical_tilt(rotation),
        final_tilt_deg=vertical_tilt(rotation),
    )


class OrientationFusionTests(unittest.TestCase):
    def test_intrinsics_scale_to_encoded_image_resolution(self):
        intrinsics = scaled_intrinsics(
            {"fx": 200.0, "fy": 150.0, "cx": 100.0, "cy": 75.0, "width": 400, "height": 300},
            (800, 600),
        )

        self.assertEqual(intrinsics, (400.0, 300.0, 200.0, 150.0))

    def test_depth_localization_uses_central_foreground_not_background(self):
        depth = np.full((120, 120), 4500, dtype=np.uint16)
        # The surrounding wall is farther away; the object occupies the
        # central portion of its detection box.
        depth[35:85, 35:85] = 2000
        result = localize_detection_from_depth(
            {"bbox_xyxy": [20.0, 20.0, 100.0, 100.0]}, depth
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result.depth_m, 2.0, places=3)
        self.assertGreaterEqual(result.sample_count, 64)
        self.assertGreater(result.confidence, 0.5)
        self.assertAlmostEqual(depth_coverage_ratio(depth), 1.0)

    def test_metric_depth_replaces_model_translation_in_world_space(self):
        depth = np.full((100, 100), 2000, dtype=np.uint16)
        metadata = {
            "fx": 100.0,
            "fy": 100.0,
            "cx": 50.0,
            "cy": 50.0,
            "width": 100,
            "height": 100,
            "pose_R_wc": np.eye(3).tolist(),
            "pose_t_wc": [1.0, 0.0, 0.0],
        }
        prediction = {
            "detections": [{"bbox_xyxy": [10.0, 10.0, 90.0, 90.0], "score": 0.9}],
            "boxes_3d": {
                # Intentionally wrong CuTR translation; metric depth should
                # control the final centre instead.
                "gravity_center_xyz": [[0.0, 0.0, 9.0]],
                "dims_lhw": [[1.0, 1.0, 1.0]],
                "R_3x3": [np.eye(3).tolist()],
            },
        }

        converted = predictions_to_world(
            "metric", prediction, metadata, (100, 100), "1,0,0,0,-1,0,0,0,1", depth
        )

        self.assertEqual(len(converted), 1)
        self.assertEqual(converted[0].localization_mode, "metric_depth")
        np.testing.assert_allclose(converted[0].center, [1.0, 0.0, 2.5], atol=0.03)

    def test_sparse_depth_does_not_create_metric_observation(self):
        depth = np.zeros((100, 100), dtype=np.uint16)
        depth[48:52, 48:52] = 2000
        result = localize_detection_from_depth(
            {"bbox_xyxy": [10.0, 10.0, 90.0, 90.0]}, depth
        )

        self.assertIsNone(result)

    def test_metric_center_fusion_rejects_a_distant_view(self):
        cluster = Cluster(
            [
                Observation(
                    frame_id="a", detection={"bbox_xyxy": [0, 0, 800, 800], "score": 0.9},
                    center=np.asarray([0.0, 1.0, 2.0]), dims=np.ones(3), rotation=np.eye(3),
                    score=0.9, image_size=(1000, 1000), localization_mode="metric_depth",
                    depth_sample_count=500, depth_coverage_ratio=1.0, localization_confidence=1.0,
                ),
                Observation(
                    frame_id="b", detection={"bbox_xyxy": [0, 0, 800, 800], "score": 0.9},
                    center=np.asarray([0.08, 1.0, 2.0]), dims=np.ones(3), rotation=np.eye(3),
                    score=0.9, image_size=(1000, 1000), localization_mode="metric_depth",
                    depth_sample_count=500, depth_coverage_ratio=1.0, localization_confidence=1.0,
                ),
                Observation(
                    frame_id="outlier", detection={"bbox_xyxy": [0, 0, 800, 800], "score": 0.9},
                    center=np.asarray([2.0, 1.0, 2.0]), dims=np.ones(3), rotation=np.eye(3),
                    score=0.9, image_size=(1000, 1000), localization_mode="metric_depth",
                    depth_sample_count=500, depth_coverage_ratio=1.0, localization_confidence=1.0,
                ),
            ]
        )

        fused = cluster.fused()

        self.assertAlmostEqual(float(fused.center[0]), 0.04, places=2)
        self.assertEqual(fused.localization_accepted_frames, ["a", "b"])
        self.assertEqual(fused.localization_rejected_frames, ["outlier"])

    def test_upright_consensus_removes_camera_tilt(self):
        expected = rotation_y(22.0)
        cluster = Cluster(
            [
                observation("down", expected @ rotation_x(-20.0)),
                observation("level", expected @ rotation_x(2.0)),
                observation("up", expected @ rotation_x(18.0)),
            ]
        )

        fused = cluster.fused()

        self.assertEqual(fused.orientation_mode, "gravity_aligned")
        self.assertLessEqual(vertical_tilt(fused.rotation), 1e-6)
        self.assertLess(yaw_line_difference(fused.rotation, expected), 3.0)
        self.assertEqual(fused.orientation_observations, 3)

    def test_axis_swap_and_180_degree_ambiguity_are_equivalent(self):
        expected = rotation_y(-31.0)
        dims = np.asarray([2.2, 1.7, 0.6])
        # new X=old Z, new Y=old Y, new Z=-old X: same physical OBB.
        swap = np.asarray(
            [
                [0.0, 0.0, -1.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        swapped_dims = np.abs(swap).T @ dims
        reversed_axes = np.diag([-1.0, 1.0, -1.0])
        cluster = Cluster(
            [
                observation("base", expected, dims=dims),
                observation(
                    "swapped",
                    expected @ swap,
                    dims=swapped_dims,
                ),
                observation(
                    "reversed",
                    expected @ reversed_axes,
                    dims=dims,
                ),
            ]
        )

        fused = cluster.fused()

        self.assertEqual(fused.orientation_mode, "gravity_aligned")
        self.assertLess(yaw_line_difference(fused.rotation, expected), 1e-5)
        np.testing.assert_allclose(fused.dims, dims, atol=1e-5)

    def test_upright_yaw_outlier_is_rejected(self):
        cluster = Cluster(
            [
                observation("a", rotation_y(10.0)),
                observation("b", rotation_y(13.0)),
                observation("outlier", rotation_y(82.0), score=0.99),
            ]
        )

        fused = cluster.fused()

        self.assertEqual(fused.orientation_mode, "gravity_aligned")
        self.assertEqual(fused.orientation_observations, 2)
        self.assertEqual(fused.orientation_rejected, 1)
        self.assertEqual(fused.orientation_rejected_frames, ["outlier"])
        self.assertNotEqual(fused.frame_id, "outlier")
        self.assertLess(yaw_line_difference(fused.rotation, rotation_y(11.5)), 3.0)

    def test_overlapping_top_and_bottom_views_share_one_cluster(self):
        observations_by_frame = {
            "bottom": [
                observation(
                    "bottom",
                    rotation_y(18.0) @ rotation_x(-15.0),
                    center=(0.0, 0.85, 2.0),
                    dims=(1.8, 1.7, 0.5),
                )
            ],
            "top": [
                observation(
                    "top",
                    rotation_y(20.0) @ rotation_x(15.0),
                    center=(0.0, 1.15, 2.0),
                    dims=(1.8, 1.7, 0.5),
                )
            ],
        }

        clusters = fuse_observations(observations_by_frame)

        self.assertEqual(len(clusters), 1)
        fused = clusters[0].fused()
        self.assertEqual(fused.orientation_mode, "gravity_aligned")
        self.assertEqual(fused.orientation_observations, 2)
        self.assertAlmostEqual(float(fused.center[1]), 1.0, delta=0.1)

    def test_consistent_inclined_object_remains_inclined(self):
        observations = [
            observation("a", rotation_y(15.0) @ rotation_x(36.0)),
            observation("b", rotation_y(17.0) @ rotation_x(34.0)),
            observation("c", rotation_y(14.0) @ rotation_x(38.0)),
        ]

        fused = Cluster(observations).fused()

        self.assertEqual(fused.orientation_mode, "free_3d")
        self.assertGreater(vertical_tilt(fused.rotation), 30.0)
        self.assertLess(vertical_tilt(fused.rotation), 42.0)
        self.assertEqual(fused.orientation_observations, 3)

    def test_single_view_falls_back_to_raw_rotation(self):
        raw = rotation_y(20.0) @ rotation_x(14.0)

        fused = Cluster([observation("only", raw)]).fused()

        self.assertEqual(fused.orientation_mode, "single_view")
        np.testing.assert_allclose(fused.rotation, raw, atol=1e-8)

    def test_output_keeps_schema_and_adds_orientation_diagnostics(self):
        cluster = Cluster(
            [
                observation("a", rotation_y(5.0) @ rotation_x(-10.0)),
                observation("b", rotation_y(7.0) @ rotation_x(10.0)),
            ]
        )

        prediction, representatives = build_fused_prediction([cluster], "rgbd")

        self.assertEqual(prediction["coordinate_space"], "unity_world")
        self.assertIn("R_3x3", prediction["boxes_3d"])
        detection = prediction["detections"][0]
        self.assertEqual(detection["orientation_mode"], "gravity_aligned")
        self.assertEqual(detection["orientation_observations"], 2)
        self.assertIn("orientation_dispersion_deg", detection)
        self.assertLessEqual(representatives[0].final_tilt_deg, 1e-6)

    def test_camera_pitch_and_raw_world_tilt_are_recorded(self):
        pose_rotation = rotation_x(20.0)
        metadata = {
            "pose_R_wc": pose_rotation.tolist(),
            "pose_t_wc": [0.0, 0.0, 0.0],
        }
        prediction = {
            "detections": [
                {
                    "bbox_xyxy": [10.0, 10.0, 90.0, 90.0],
                    "score": 0.8,
                }
            ],
            "boxes_3d": {
                "gravity_center_xyz": [[0.0, 0.0, 2.0]],
                "dims_lhw": [[1.0, 2.0, 0.5]],
                "R_3x3": [np.eye(3).tolist()],
            },
        }

        converted = predictions_to_world(
            "pitched",
            prediction,
            metadata,
            (100, 100),
        )

        self.assertEqual(len(converted), 1)
        self.assertAlmostEqual(abs(converted[0].camera_pitch_deg), 20.0, places=5)
        self.assertAlmostEqual(converted[0].raw_tilt_deg, 20.0, places=5)


if __name__ == "__main__":
    unittest.main()
