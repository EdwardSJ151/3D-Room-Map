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
            )
        )
    return observations


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
        weights = np.asarray([max(0.01, item.score) for item in self.observations])
        weights /= weights.sum()
        center = sum((weight * item.center for weight, item in zip(weights, self.observations)), np.zeros(3))
        dims = sum((weight * item.dims for weight, item in zip(weights, self.observations)), np.zeros(3))
        representative = max(
            self.observations,
            key=lambda item: (item.score, item.bbox_area_ratio),
        )
        return Observation(
            frame_id=representative.frame_id,
            detection=dict(representative.detection),
            center=center,
            dims=dims,
            rotation=representative.rotation,
            score=representative.score,
            image_size=representative.image_size,
        )


def _compatible(candidate: Observation, reference: Observation) -> Tuple[bool, float]:
    ratios = np.maximum(candidate.dims / reference.dims, reference.dims / candidate.dims)
    if np.any(ratios > 2.0):
        return False, math.inf
    distance = float(np.linalg.norm(candidate.center - reference.center))
    threshold = max(0.35, 0.30 * min(candidate.diagonal, reference.diagonal))
    iou = _aabb_iou(candidate, reference)
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
