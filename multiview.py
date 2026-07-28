from __future__ import annotations

import base64
import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image


def _matrix3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return array


def _matrix4(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (4, 4) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    return array


def _vector3(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return array


def camera_to_world(meta: Dict[str, Any]) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _matrix3(meta.get("pose_R_wc"), "pose_R_wc")
    result[:3, 3] = _vector3(meta.get("pose_t_wc"), "pose_t_wc")
    return result


def decode_depth_mm(depth_base64: str, depth_meta: Dict[str, Any]) -> np.ndarray:
    width = int(depth_meta.get("width", 0))
    height = int(depth_meta.get("height", 0))
    if width <= 0 or height <= 0 or width * height > 16_777_216:
        raise ValueError("invalid depth dimensions")
    try:
        raw = gzip.decompress(base64.b64decode(depth_base64, validate=True))
    except Exception as exc:
        raise ValueError(f"invalid gzip/base64 depth payload: {exc}") from exc
    expected = width * height * 2
    if len(raw) != expected:
        raise ValueError(f"depth payload has {len(raw)} bytes; expected {expected}")
    depth = np.frombuffer(raw, dtype="<u2").reshape(height, width)
    if bool(depth_meta.get("flip_y", False)):
        depth = np.flipud(depth)
    return depth.copy()


def validate_depth(depth_mm: np.ndarray) -> bool:
    valid = depth_mm[(depth_mm >= 100) & (depth_mm <= 20_000)]
    return valid.size >= max(64, int(depth_mm.size * 0.01))


def reproject_depth_to_rgb(
    depth_mm: np.ndarray,
    depth_meta: Dict[str, Any],
    rgb_meta: Dict[str, Any],
    rgb_size: Tuple[int, int],
) -> np.ndarray:
    """Reproject metric eye depth into the passthrough RGB camera.

    `projection_matrix` and `camera_to_world` use Unity's left-handed camera
    convention. The RGB intrinsics are pixel-space intrinsics for the image
    delivered by PassthroughCameraAccess.
    """
    projection = _matrix4(depth_meta.get("projection_matrix"), "projection_matrix")
    depth_to_world = _matrix4(depth_meta.get("camera_to_world"), "camera_to_world")
    rgb_to_world = camera_to_world(rgb_meta)
    world_to_rgb = np.linalg.inv(rgb_to_world)
    inv_projection = np.linalg.inv(projection)

    height, width = depth_mm.shape
    ys, xs = np.nonzero((depth_mm >= 100) & (depth_mm <= 20_000))
    if xs.size == 0:
        return np.zeros((rgb_size[1], rgb_size[0]), dtype=np.uint16)

    # Texture row zero is normalized by `flip_y` before this point. Unity NDC
    # is [-1, 1], with +Y at the top for the logical camera image.
    ndc_x = ((xs.astype(np.float64) + 0.5) / width) * 2.0 - 1.0
    ndc_y = 1.0 - ((ys.astype(np.float64) + 0.5) / height) * 2.0
    clip = np.stack((ndc_x, ndc_y, np.ones_like(ndc_x), np.ones_like(ndc_x)), axis=0)
    rays = inv_projection @ clip
    rays /= np.where(np.abs(rays[3:4]) > 1e-9, rays[3:4], 1.0)
    ray_z = rays[2]
    usable = np.abs(ray_z) > 1e-6
    rays = rays[:, usable]
    metric_depth = depth_mm[ys[usable], xs[usable]].astype(np.float64) / 1000.0
    points_depth = rays[:3] * (metric_depth / rays[2])

    points_depth_h = np.vstack((points_depth, np.ones((1, points_depth.shape[1]))))
    points_rgb = world_to_rgb @ (depth_to_world @ points_depth_h)
    z = points_rgb[2]
    in_front = z > 0.05
    points_rgb = points_rgb[:, in_front]
    z = z[in_front]

    fx = float(rgb_meta["fx"])
    fy = float(rgb_meta["fy"])
    cx = float(rgb_meta["cx"])
    cy = float(rgb_meta["cy"])
    rgb_width, rgb_height = rgb_size
    u = np.rint(fx * points_rgb[0] / z + cx).astype(np.int64)
    v = np.rint(fy * points_rgb[1] / z + cy).astype(np.int64)
    inside = (u >= 0) & (u < rgb_width) & (v >= 0) & (v < rgb_height)
    u, v, z = u[inside], v[inside], z[inside]

    flat = np.full(rgb_width * rgb_height, np.inf, dtype=np.float64)
    indices = v * rgb_width + u
    np.minimum.at(flat, indices, z)
    aligned = flat.reshape(rgb_height, rgb_width)
    aligned[~np.isfinite(aligned)] = 0.0
    return np.clip(np.rint(aligned * 1000.0), 0, 65535).astype(np.uint16)


def save_depth_png(depth_mm: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_mm, mode="I;16").save(path)


def _basis_from_env(value: Optional[str] = None) -> np.ndarray:
    # CuTR predictions follow an image-camera convention. The default maps
    # OpenCV (+X right, +Y down, +Z forward) into Unity camera coordinates.
    raw = value or "1,0,0,0,-1,0,0,0,1"
    parts = [float(item.strip()) for item in raw.split(",")]
    if len(parts) != 9:
        raise ValueError("CUTR_TO_UNITY_BASIS must contain 9 comma-separated numbers")
    basis = np.asarray(parts, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(basis).all() or abs(abs(np.linalg.det(basis)) - 1.0) > 1e-3:
        raise ValueError("CUTR_TO_UNITY_BASIS must be an orthonormal basis")
    return basis


@dataclass
class Observation:
    frame_id: str
    detection: Dict[str, Any]
    center: np.ndarray
    dims: np.ndarray
    rotation: np.ndarray
    score: float
    image_size: Tuple[int, int]
    camera_pitch_deg: float = 0.0
    orientation_mode: str = "raw"
    orientation_observations: int = 1
    orientation_rejected: int = 0
    orientation_dispersion_deg: float = 0.0
    raw_tilt_deg: float = 0.0
    final_tilt_deg: float = 0.0
    orientation_accepted_frames: List[str] = field(default_factory=list)
    orientation_rejected_frames: List[str] = field(default_factory=list)

    @property
    def diagonal(self) -> float:
        return float(np.linalg.norm(self.dims))

    @property
    def bbox_area_ratio(self) -> float:
        bbox = self.detection.get("bbox_xyxy") or [0, 0, 0, 0]
        x1, y1, x2, y2 = (float(x) for x in bbox)
        width, height = self.image_size
        return max(0.0, x2 - x1) * max(0.0, y2 - y1) / max(1.0, width * height)


def predictions_to_world(
    frame_id: str,
    pred: Dict[str, Any],
    rgb_meta: Dict[str, Any],
    image_size: Tuple[int, int],
    basis_value: Optional[str] = None,
) -> List[Observation]:
    detections = pred.get("detections") or []
    boxes = pred.get("boxes_3d") or {}
    centers = boxes.get("gravity_center_xyz") or []
    dimensions = boxes.get("dims_lhw") or []
    rotations = boxes.get("R_3x3") or []
    count = min(len(detections), len(centers), len(dimensions), len(rotations))
    pose = camera_to_world(rgb_meta)
    basis = _basis_from_env(basis_value)
    camera_forward = pose[:3, 2]
    camera_pitch_deg = math.degrees(
        math.asin(float(np.clip(camera_forward[1], -1.0, 1.0)))
    )
    observations: List[Observation] = []
    for index in range(count):
        center_camera = _vector3(centers[index], f"center[{index}]")
        dims = _vector3(dimensions[index], f"dims[{index}]")
        rotation_camera = _matrix3(rotations[index], f"rotation[{index}]")
        if np.any(dims <= 0):
            continue
        center_unity_camera = basis @ center_camera
        rotation_unity_camera = basis @ rotation_camera @ basis.T
        center_world = pose[:3, :3] @ center_unity_camera + pose[:3, 3]
        rotation_world = pose[:3, :3] @ rotation_unity_camera
        u, _, vt = np.linalg.svd(rotation_world)
        rotation_world = u @ vt
        observations.append(
            Observation(
                frame_id=frame_id,
                detection=dict(detections[index]),
                center=center_world,
                dims=dims,
                rotation=rotation_world,
                score=float(detections[index].get("score", 0.0)),
                image_size=image_size,
                camera_pitch_deg=camera_pitch_deg,
                raw_tilt_deg=_vertical_tilt_deg(rotation_world),
                final_tilt_deg=_vertical_tilt_deg(rotation_world),
            )
        )
    return observations


_WORLD_UP = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
_UPRIGHT_TILT_DEG = 25.0
_UPRIGHT_CONSENSUS = 0.60
_ORIENTATION_OUTLIER_DEG = 30.0


def _rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _vertical_tilt_deg(rotation: np.ndarray) -> float:
    alignment = float(np.max(np.abs(rotation.T @ _WORLD_UP)))
    return math.degrees(math.acos(float(np.clip(alignment, -1.0, 1.0))))


def _observation_weight(observation: Observation) -> float:
    # A close, high-confidence observation should count more, without allowing
    # an extreme close-up to completely dominate the remaining views.
    area = float(np.clip(observation.bbox_area_ratio, 1e-4, 0.50))
    return max(0.01, observation.score) * math.sqrt(area)


def _upright_geometry(observation: Observation) -> Tuple[np.ndarray, np.ndarray]:
    """Return an equivalent OBB with +Y as its vertical local axis.

    The longest of the two horizontal axes becomes local X. An OBB is
    invariant to axis sign and to swapping equivalent local axes, so this
    canonical form removes the most common 90/180-degree ambiguities.
    """
    rotation = observation.rotation
    dims = observation.dims
    vertical_index = int(np.argmax(np.abs(rotation.T @ _WORLD_UP)))
    horizontal_indices = [index for index in range(3) if index != vertical_index]
    x_index = max(horizontal_indices, key=lambda index: float(dims[index]))

    x_axis = rotation[:, x_index].copy()
    x_axis -= float(x_axis @ _WORLD_UP) * _WORLD_UP
    norm = float(np.linalg.norm(x_axis))
    if norm < 1e-8:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        x_axis /= norm
    z_axis = np.cross(x_axis, _WORLD_UP)
    z_axis /= max(1e-8, float(np.linalg.norm(z_axis)))
    upright_rotation = np.column_stack((x_axis, _WORLD_UP, z_axis))

    z_index = next(
        index for index in horizontal_indices if index != x_index
    )
    upright_dims = np.asarray(
        [dims[x_index], dims[vertical_index], dims[z_index]],
        dtype=np.float64,
    )
    return upright_rotation, upright_dims


def _line_angle_difference_deg(a: float, b: float) -> float:
    delta = (a - b + math.pi / 2.0) % math.pi - math.pi / 2.0
    return abs(math.degrees(delta))


def _weighted_line_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    x = float(np.sum(weights * np.cos(2.0 * angles)))
    y = float(np.sum(weights * np.sin(2.0 * angles)))
    if abs(x) + abs(y) < 1e-12:
        return float(angles[int(np.argmax(weights))])
    return 0.5 * math.atan2(y, x)


def _proper_axis_transforms() -> List[np.ndarray]:
    transforms: List[np.ndarray] = []
    for permutation in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        for signs in (
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ):
            transform = np.zeros((3, 3), dtype=np.float64)
            for new_axis, old_axis in enumerate(permutation):
                transform[old_axis, new_axis] = signs[new_axis]
            if np.linalg.det(transform) > 0.5:
                transforms.append(transform)
    return transforms


_AXIS_TRANSFORMS = _proper_axis_transforms()


def _align_geometry(
    rotation: np.ndarray,
    dims: np.ndarray,
    reference: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    for transform in _AXIS_TRANSFORMS:
        candidate_rotation = rotation @ transform
        angle = _rotation_angle_deg(candidate_rotation, reference)
        candidate_dims = np.abs(transform).T @ dims
        if best is None or angle < best[2]:
            best = (candidate_rotation, candidate_dims, angle)
    assert best is not None
    return best


def _projected_bounds(
    observation: Observation,
    final_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    signs = np.asarray(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    corners = observation.center + (signs * (observation.dims / 2.0)) @ observation.rotation.T
    local_corners = corners @ final_rotation
    return np.min(local_corners, axis=0), np.max(local_corners, axis=0)


def _refit_geometry(
    observations: List[Observation],
    weights: np.ndarray,
    final_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    weights = weights / weights.sum()
    minima: List[np.ndarray] = []
    maxima: List[np.ndarray] = []
    for observation in observations:
        minimum, maximum = _projected_bounds(observation, final_rotation)
        minima.append(minimum)
        maxima.append(maximum)
    lower = np.average(np.asarray(minima), axis=0, weights=weights)
    upper = np.average(np.asarray(maxima), axis=0, weights=weights)
    local_center = (lower + upper) / 2.0
    return final_rotation @ local_center, np.maximum(upper - lower, 1e-4)


def _fuse_upright(
    observations: List[Observation],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    canonical = [_upright_geometry(item) for item in observations]
    angles = np.asarray(
        [math.atan2(rotation[2, 0], rotation[0, 0]) for rotation, _ in canonical]
    )
    weights = np.asarray([_observation_weight(item) for item in observations])

    medoid_index = min(
        range(len(observations)),
        key=lambda index: float(
            np.sum(
                weights
                * np.asarray(
                    [
                        _line_angle_difference_deg(angles[index], angle)
                        for angle in angles
                    ]
                )
            )
        ),
    )
    accepted = [
        index
        for index, angle in enumerate(angles)
        if _line_angle_difference_deg(angle, angles[medoid_index])
        <= _ORIENTATION_OUTLIER_DEG
    ]
    accepted_angles = angles[accepted]
    accepted_weights = weights[accepted]
    yaw = _weighted_line_mean(accepted_angles, accepted_weights)
    x_axis = np.asarray([math.cos(yaw), 0.0, math.sin(yaw)])
    final_rotation = np.column_stack(
        (x_axis, _WORLD_UP, np.cross(x_axis, _WORLD_UP))
    )

    accepted_observations = [observations[index] for index in accepted]
    center, dims = _refit_geometry(
        accepted_observations,
        accepted_weights,
        final_rotation,
    )
    differences = np.asarray(
        [_line_angle_difference_deg(angle, yaw) for angle in accepted_angles]
    )
    dispersion = float(
        math.sqrt(np.average(differences * differences, weights=accepted_weights))
    )
    return center, dims, final_rotation, accepted, dispersion


def _fuse_free_3d(
    observations: List[Observation],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], float]:
    weights = np.asarray([_observation_weight(item) for item in observations])
    pairwise = np.zeros((len(observations), len(observations)), dtype=np.float64)
    for row, observation in enumerate(observations):
        for column, reference in enumerate(observations):
            _, _, pairwise[row, column] = _align_geometry(
                observation.rotation,
                observation.dims,
                reference.rotation,
            )
    medoid_index = min(
        range(len(observations)),
        key=lambda index: float(np.sum(weights * pairwise[:, index])),
    )
    reference = observations[medoid_index].rotation
    aligned = [
        _align_geometry(item.rotation, item.dims, reference)
        for item in observations
    ]
    accepted = [
        index
        for index, (_, _, angle) in enumerate(aligned)
        if angle <= _ORIENTATION_OUTLIER_DEG
    ]
    accepted_weights = weights[accepted]
    rotation_sum = sum(
        (
            accepted_weights[offset] * aligned[index][0]
            for offset, index in enumerate(accepted)
        ),
        np.zeros((3, 3), dtype=np.float64),
    )
    u, _, vt = np.linalg.svd(rotation_sum)
    final_rotation = u @ vt
    if np.linalg.det(final_rotation) < 0:
        u[:, -1] *= -1
        final_rotation = u @ vt

    normalized_observations: List[Observation] = []
    for index in accepted:
        item = observations[index]
        normalized_observations.append(
            Observation(
                frame_id=item.frame_id,
                detection=item.detection,
                center=item.center,
                dims=aligned[index][1],
                rotation=aligned[index][0],
                score=item.score,
                image_size=item.image_size,
            )
        )
    center, dims = _refit_geometry(
        normalized_observations,
        accepted_weights,
        final_rotation,
    )
    differences = np.asarray(
        [
            _rotation_angle_deg(aligned[index][0], final_rotation)
            for index in accepted
        ]
    )
    dispersion = float(
        math.sqrt(np.average(differences * differences, weights=accepted_weights))
    )
    return center, dims, final_rotation, accepted, dispersion


def _aabb(obs: Observation) -> Tuple[np.ndarray, np.ndarray]:
    half_extents = np.abs(obs.rotation) @ (obs.dims / 2.0)
    return obs.center - half_extents, obs.center + half_extents


def _aabb_iou(a: Observation, b: Observation) -> float:
    a_min, a_max = _aabb(a)
    b_min, b_max = _aabb(b)
    intersection = np.maximum(0.0, np.minimum(a_max, b_max) - np.maximum(a_min, b_min))
    intersection_volume = float(np.prod(intersection))
    if intersection_volume <= 0:
        return 0.0
    a_volume = float(np.prod(a_max - a_min))
    b_volume = float(np.prod(b_max - b_min))
    return intersection_volume / max(1e-9, a_volume + b_volume - intersection_volume)


@dataclass
class Cluster:
    observations: List[Observation] = field(default_factory=list)

    def fused(self) -> Observation:
        representative = max(
            self.observations,
            key=lambda item: (item.score, item.bbox_area_ratio),
        )
        if len(self.observations) == 1:
            return Observation(
                frame_id=representative.frame_id,
                detection=dict(representative.detection),
                center=representative.center.copy(),
                dims=representative.dims.copy(),
                rotation=representative.rotation.copy(),
                score=representative.score,
                image_size=representative.image_size,
                camera_pitch_deg=representative.camera_pitch_deg,
                orientation_mode="single_view",
                orientation_observations=1,
                raw_tilt_deg=_vertical_tilt_deg(representative.rotation),
                final_tilt_deg=_vertical_tilt_deg(representative.rotation),
                orientation_accepted_frames=[representative.frame_id],
            )

        upright_count = sum(
            _vertical_tilt_deg(item.rotation) <= _UPRIGHT_TILT_DEG
            for item in self.observations
        )
        upright_consensus = (
            upright_count / len(self.observations) >= _UPRIGHT_CONSENSUS
        )
        if upright_consensus:
            eligible = [
                item
                for item in self.observations
                if _vertical_tilt_deg(item.rotation) <= _UPRIGHT_TILT_DEG
            ]
            center, dims, rotation, accepted_indices, dispersion = _fuse_upright(
                eligible
            )
            accepted = [eligible[index] for index in accepted_indices]
            mode = "gravity_aligned"
            rejected = len(self.observations) - len(accepted)
        else:
            center, dims, rotation, accepted_indices, dispersion = _fuse_free_3d(
                self.observations
            )
            accepted = [self.observations[index] for index in accepted_indices]
            mode = "free_3d"
            rejected = len(self.observations) - len(accepted)

        representative = max(
            accepted,
            key=lambda item: (item.score, item.bbox_area_ratio),
        )
        raw_weights = np.asarray([_observation_weight(item) for item in accepted])
        raw_tilt = float(
            np.average(
                [_vertical_tilt_deg(item.rotation) for item in accepted],
                weights=raw_weights,
            )
        )
        accepted_frame_ids = {item.frame_id for item in accepted}
        return Observation(
            frame_id=representative.frame_id,
            detection=dict(representative.detection),
            center=center,
            dims=dims,
            rotation=rotation,
            score=representative.score,
            image_size=representative.image_size,
            camera_pitch_deg=representative.camera_pitch_deg,
            orientation_mode=mode,
            orientation_observations=len(accepted),
            orientation_rejected=rejected,
            orientation_dispersion_deg=dispersion,
            raw_tilt_deg=raw_tilt,
            final_tilt_deg=_vertical_tilt_deg(rotation),
            orientation_accepted_frames=[item.frame_id for item in accepted],
            orientation_rejected_frames=[
                item.frame_id
                for item in self.observations
                if item.frame_id not in accepted_frame_ids
            ],
        )


def _matching_observation(observation: Observation) -> Observation:
    if _vertical_tilt_deg(observation.rotation) > _UPRIGHT_TILT_DEG:
        return observation
    rotation, dims = _upright_geometry(observation)
    return Observation(
        frame_id=observation.frame_id,
        detection=observation.detection,
        center=observation.center,
        dims=dims,
        rotation=rotation,
        score=observation.score,
        image_size=observation.image_size,
    )


def _compatible(candidate: Observation, reference: Observation) -> Tuple[bool, float]:
    candidate_match = _matching_observation(candidate)
    reference_match = _matching_observation(reference)
    candidate_dims = np.sort(candidate_match.dims)
    reference_dims = np.sort(reference_match.dims)
    ratios = np.maximum(
        candidate_dims / reference_dims,
        reference_dims / candidate_dims,
    )
    if np.any(ratios > 2.0):
        return False, math.inf
    distance = float(np.linalg.norm(candidate.center - reference.center))
    threshold = max(0.35, 0.30 * min(candidate.diagonal, reference.diagonal))
    iou = _aabb_iou(candidate_match, reference_match)
    accepted = iou >= 0.05 or distance <= threshold
    cost = distance / max(threshold, 1e-6) + (1.0 - iou)
    return accepted, cost


def fuse_observations(observations_by_frame: Dict[str, List[Observation]]) -> List[Cluster]:
    clusters: List[Cluster] = []
    for frame_id in sorted(observations_by_frame):
        frame_observations = sorted(
            observations_by_frame[frame_id],
            key=lambda item: item.score,
            reverse=True,
        )
        available = set(range(len(clusters)))
        for observation in frame_observations:
            choices: List[Tuple[float, int]] = []
            for cluster_index in available:
                reference = clusters[cluster_index].fused()
                accepted, cost = _compatible(observation, reference)
                if accepted:
                    choices.append((cost, cluster_index))
            if choices:
                _, selected = min(choices)
                clusters[selected].observations.append(observation)
                available.remove(selected)
            else:
                clusters.append(Cluster([observation]))
    return clusters


def build_fused_prediction(
    clusters: Iterable[Cluster],
    inference_mode: str,
) -> Tuple[Dict[str, Any], List[Observation]]:
    detections: List[Dict[str, Any]] = []
    centers: List[List[float]] = []
    dims: List[List[float]] = []
    rotations: List[List[List[float]]] = []
    representatives: List[Observation] = []
    for cluster in clusters:
        fused = cluster.fused()
        representatives.append(fused)
        detection = dict(fused.detection)
        detection["score"] = fused.score
        detection["source_frame_id"] = fused.frame_id
        detection["source_bbox_xyxy"] = list(detection.get("bbox_xyxy") or [])
        detection["observation_count"] = len(cluster.observations)
        detection["inference_mode"] = inference_mode
        detection["orientation_mode"] = fused.orientation_mode
        detection["orientation_observations"] = fused.orientation_observations
        detection["orientation_rejected"] = fused.orientation_rejected
        detection["orientation_dispersion_deg"] = round(
            fused.orientation_dispersion_deg, 3
        )
        detection["orientation_raw_tilt_deg"] = round(fused.raw_tilt_deg, 3)
        detection["orientation_final_tilt_deg"] = round(fused.final_tilt_deg, 3)
        detection["orientation_accepted_frames"] = list(
            fused.orientation_accepted_frames
        )
        detection["orientation_rejected_frames"] = list(
            fused.orientation_rejected_frames
        )
        detections.append(detection)
        centers.append(fused.center.astype(float).tolist())
        dims.append(fused.dims.astype(float).tolist())
        rotations.append(fused.rotation.astype(float).tolist())
    return (
        {
            "coordinate_space": "unity_world",
            "inference_mode": inference_mode,
            "detections": detections,
            "boxes_3d": {
                "gravity_center_xyz": centers,
                "dims_lhw": dims,
                "R_3x3": rotations,
            },
        },
        representatives,
    )


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
