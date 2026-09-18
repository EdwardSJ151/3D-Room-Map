from __future__ import annotations

import base64
import io
import json
import math
import os
import random
import re
import string
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from crop_utils import crop_for_detection, save_detection_crops
from multiview import (
    build_fused_prediction,
    camera_to_world,
    decode_depth_mm,
    fuse_observations,
    predictions_to_world,
    reproject_depth_to_rgb,
    save_depth_png,
    validate_depth,
    write_manifest,
)

# Paths / cubify repo
def _in_colab() -> bool:
    # Colab sets COLAB_RELEASE_TAG and ships a google.colab module.
    if os.environ.get("COLAB_RELEASE_TAG"):
        return True
    try:
        import google.colab  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


_DEFAULT_BASE = Path("/content") if _in_colab() else Path.cwd()
DEFAULT_REPO_DIR = _DEFAULT_BASE / "ml-cubifyanything"

# Path to the cloned ml-cubifyanything repo.
# Override with env var CUBIFY_REPO if needed.
REPO_DIR = Path(os.environ.get("CUBIFY_REPO", str(DEFAULT_REPO_DIR)))

# Default CuTR checkpoint path.
# Override with env var CUTR_MODEL_PATH if needed.
DEFAULT_MODEL_PATH = Path(
    os.environ.get(
        "CUTR_MODEL_PATH",
        str(REPO_DIR / "models" / "cutr_rgb.pth")
    )
)
DEFAULT_RGBD_MODEL_PATH = Path(
    os.environ.get(
        "CUTR_RGBD_MODEL_PATH",
        str(REPO_DIR / "models" / "cutr_rgbd.pth"),
    )
)
TOOLS_DIR = REPO_DIR / "tools"

if TOOLS_DIR.exists() and str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

# Lazy imports — pulled in at first inference so the API can still boot
# even if the cubify repo isn't on this machine yet.
_cutr_runner = None
_cutr_imports: Dict[str, Any] = {}


def _load_cutr():
    global _cutr_runner
    if _cutr_imports:
        return _cutr_imports
    if not TOOLS_DIR.exists():
        raise RuntimeError(
            f"Cubify tools dir not found: {TOOLS_DIR}. "
            f"Set CUBIFY_REPO env var to point at your ml-cubifyanything clone."
        )
    from cutr_runtime import CutrRunner, make_default_intrinsics  # type: ignore
    from infer_image import (  # type: ignore
        _meta_intrinsic,
        load_meta_json,
        save_pred_json,
    )

    _cutr_imports.update(
        CutrRunner=CutrRunner,
        make_default_intrinsics=make_default_intrinsics,
        _meta_intrinsic=_meta_intrinsic,
        load_meta_json=load_meta_json,
        save_pred_json=save_pred_json,
    )
    return _cutr_imports


def _get_runner(model_path: str, device: str):
    """Singleton CuTR runner (one per process). Re-created if model_path changes."""
    global _cutr_runner
    key = (model_path, device)
    if _cutr_runner is None or _cutr_runner[0] != key:
        mods = _load_cutr()
        runner = mods["CutrRunner"](model_path=model_path, device=device)
        _cutr_runner = (key, runner)
    return _cutr_runner[1]


# Job infra
JOBS_DIR = Path(os.environ.get("CUTR_JOBS_DIR", "cutr_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_ERROR = "error"
JOB_COLLECTING = "collecting"

_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=1)


def _new_job_id() -> str:
    # 10-char id: 6 letters + 2 digits + 2 letters
    letters_a = "".join(random.choices(string.ascii_lowercase, k=6))
    digits    = "".join(random.choices(string.digits, k=2))
    letters_b = "".join(random.choices(string.ascii_lowercase, k=2))
    return f"{letters_a}{digits}{letters_b}"


def _unique_job_id() -> str:
    for _ in range(20):
        jid = _new_job_id()
        if not (JOBS_DIR / jid).exists() and jid not in _jobs:
            return jid
    raise RuntimeError("Could not allocate unique job id")


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _job_set(job_id: str, **fields):
    with _jobs_lock:
        job = _jobs.setdefault(job_id, {})
        job.update(fields)


def _job_get(job_id: str) -> Optional[Dict[str, Any]]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job:
            return dict(job)
    # Fallback to disk so the API can answer after a restart.
    d = _job_dir(job_id)
    if not d.exists():
        return None
    status_file = d / "status.json"
    if status_file.exists():
        try:
            return json.loads(status_file.read_text())
        except Exception:
            pass
    return {"status": JOB_DONE if (d / "pred.json").exists() else JOB_ERROR}


def _persist_status(job_id: str, status: str, error: Optional[str] = None, **extra):
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {"status": status}
    if error:
        payload["error"] = error
    payload.update(extra)
    (d / "status.json").write_text(json.dumps(payload, indent=2))


def _write_metrics(path: Path, metrics: Dict[str, Any]) -> None:
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in metrics.items()),
        encoding="utf-8",
    )


# API models
class CutrRunRequest(BaseModel):
    image_base64: str = Field(..., description="PNG/JPEG image, base64-encoded.")
    meta_json: Dict[str, Any] = Field(..., description="Passthrough meta JSON with intrinsics.")
    score_thresh: float = 0.35
    max_edge: int = 0  # 0 disables resize
    device: str = "cuda"
    model_path: str = str(DEFAULT_MODEL_PATH)


class JobStartResponse(BaseModel):
    job_id: str


class MultiFrameRequest(BaseModel):
    image_base64: str = Field(..., description="JPEG/PNG RGB frame, base64-encoded.")
    meta_json: Dict[str, Any] = Field(..., description="RGB intrinsics and camera-to-world pose.")
    depth_base64: Optional[str] = Field(
        default=None,
        description="Optional gzip-compressed little-endian uint16 millimeter depth.",
    )
    depth_meta: Optional[Dict[str, Any]] = None


class MultiFinalizeRequest(BaseModel):
    score_thresh: float = 0.35
    max_edge: int = 0
    device: str = "cuda"
    rgb_model_path: str = str(DEFAULT_MODEL_PATH)
    rgbd_model_path: str = str(DEFAULT_RGBD_MODEL_PATH)


class JobStatus(BaseModel):
    status: str
    error: Optional[str] = None
    uploaded_frames: Optional[int] = None
    processed_frames: Optional[int] = None
    num_detections: Optional[int] = None


# App
app = FastAPI(title="CuTR Inference API")


_FRAME_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _decode_image(image_base64: str):
    from PIL import Image

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image.load()
    except Exception as exc:
        raise ValueError(f"invalid RGB image: {exc}") from exc
    if image.width <= 0 or image.height <= 0 or image.width * image.height > 33_554_432:
        raise ValueError("invalid RGB dimensions")
    return image


def _make_intrinsics(mods: Dict[str, Any], meta: Dict[str, Any], width: int, height: int):
    values = [meta.get(key) for key in ("fx", "fy", "cx", "cy")]
    if not any(value is not None for value in values):
        return None
    if not all(value is not None and float(value) == float(value) for value in values):
        raise ValueError("fx, fy, cx and cy must all be finite when supplied")
    matrix = mods["make_default_intrinsics"](width, height)
    matrix[0, 0] = float(meta["fx"])
    matrix[1, 1] = float(meta["fy"])
    matrix[0, 2] = float(meta["cx"])
    matrix[1, 2] = float(meta["cy"])
    return matrix


def _frame_dirs(job_dir: Path) -> List[Path]:
    frames_dir = job_dir / "frames"
    if not frames_dir.exists():
        return []
    return sorted(
        path
        for path in frames_dir.iterdir()
        if path.is_dir() and (path / "input.jpg").exists() and (path / "meta.json").exists()
    )


def _load_frame(frame_dir: Path) -> Dict[str, Any]:
    from PIL import Image

    image_path = frame_dir / "input.jpg"
    meta_path = frame_dir / "meta.json"
    image = Image.open(image_path).convert("RGB")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frame: Dict[str, Any] = {
        "id": frame_dir.name,
        "dir": frame_dir,
        "image_path": image_path,
        "image": image,
        "meta": meta,
        "depth_mm": None,
        "depth_meta": None,
    }
    depth_path = frame_dir / "depth.bin.gz"
    depth_meta_path = frame_dir / "depth_meta.json"
    if depth_path.exists() and depth_meta_path.exists():
        depth_meta = json.loads(depth_meta_path.read_text(encoding="utf-8"))
        encoded = base64.b64encode(depth_path.read_bytes()).decode("ascii")
        depth_mm = decode_depth_mm(encoded, depth_meta)
        if validate_depth(depth_mm):
            frame["depth_mm"] = depth_mm
            frame["depth_meta"] = depth_meta
    return frame


def _run_multiview_job(job_id: str, req: MultiFinalizeRequest) -> None:
    from PIL import Image

    job_dir = _job_dir(job_id)
    log_path = job_dir / "run.log"
    map_started = time.perf_counter()
    try:
        _job_set(job_id, status=JOB_RUNNING, error=None, processed_frames=0)
        frame_dirs = _frame_dirs(job_dir)
        _persist_status(
            job_id,
            JOB_RUNNING,
            uploaded_frames=len(frame_dirs),
            processed_frames=0,
        )
        frames = [_load_frame(path) for path in frame_dirs]

        rgbd_frames: List[Dict[str, Any]] = []
        for frame in frames:
            if frame["depth_mm"] is None:
                continue
            image: Image.Image = frame["image"]
            try:
                aligned = reproject_depth_to_rgb(
                    frame["depth_mm"],
                    frame["depth_meta"],
                    frame["meta"],
                    image.size,
                )
            except Exception as exc:
                frame["depth_error"] = str(exc)
                continue
            if not validate_depth(aligned):
                frame["depth_error"] = "reprojected depth has insufficient valid coverage"
                continue
            frame["aligned_depth_mm"] = aligned
            save_depth_png(aligned, frame["dir"] / "depth_aligned.png")
            rgbd_frames.append(frame)

        use_rgbd = len(rgbd_frames) >= 3
        selected_frames = rgbd_frames if use_rgbd else frames
        inference_mode = "rgbd" if use_rgbd else "rgb"
        model_path = req.rgbd_model_path if use_rgbd else req.rgb_model_path
        runner = _get_runner(model_path, req.device)
        mods = _load_cutr()
        max_edge = None if int(req.max_edge or 0) <= 0 else int(req.max_edge)
        observations_by_frame: Dict[str, Any] = {}
        total_inference_time = 0.0
        frame_inference_times: List[float] = []
        transform_time = 0.0
        raw_detections = 0

        for processed, frame in enumerate(selected_frames, start=1):
            image = frame["image"]
            intrinsics = _make_intrinsics(mods, frame["meta"], image.width, image.height)
            depth_m = None
            if use_rgbd:
                depth_m = frame["aligned_depth_mm"].astype("float32") / 1000.0
            started = time.perf_counter()
            pred = runner.infer(
                image=image,
                K=intrinsics,
                depth_m=depth_m,
                score_thresh=float(req.score_thresh),
                max_edge=max_edge,
            )
            frame_time = time.perf_counter() - started
            frame_inference_times.append(frame_time)
            total_inference_time += frame_time
            raw_detections += len(pred.get("detections") or [])
            mods["save_pred_json"](
                pred,
                image_path=frame["image_path"],
                out_path=frame["dir"] / "pred.json",
            )
            transform_started = time.perf_counter()
            observations_by_frame[frame["id"]] = predictions_to_world(
                frame["id"],
                pred,
                frame["meta"],
                image.size,
                os.environ.get("CUTR_TO_UNITY_BASIS"),
            )
            transform_time += time.perf_counter() - transform_started
            _job_set(job_id, processed_frames=processed)
            _persist_status(
                job_id,
                JOB_RUNNING,
                uploaded_frames=len(frames),
                processed_frames=processed,
            )

        fusion_started = time.perf_counter()
        clusters = fuse_observations(observations_by_frame)
        fused_pred, representatives = build_fused_prediction(clusters, inference_mode)
        fusion_deduplication_time = transform_time + time.perf_counter() - fusion_started
        total_map_time = time.perf_counter() - map_started
        pred_path = job_dir / "pred.json"
        pred_path.write_text(json.dumps(fused_pred, indent=2), encoding="utf-8")

        representatives_dir = job_dir / "representatives"
        representatives_dir.mkdir(exist_ok=True)
        frames_by_id = {frame["id"]: frame for frame in frames}
        crop_started = time.perf_counter()
        for index, observation in enumerate(representatives):
            source = frames_by_id[observation.frame_id]["image"]
            crop = crop_for_detection(source, observation.detection)
            if crop is not None:
                crop.convert("RGB").save(
                    representatives_dir / f"{index}.jpg",
                    format="JPEG",
                    quality=90,
                )
        crop_generation_time = time.perf_counter() - crop_started

        num_fused_objects = len(clusters)
        mean_observations = (
            sum(len(cluster.observations) for cluster in clusters) / num_fused_objects
            if num_fused_objects else 0.0
        )
        mean_frame_time = (
            sum(frame_inference_times) / len(frame_inference_times)
            if frame_inference_times else 0.0
        )
        variance = (
            sum((value - mean_frame_time) ** 2 for value in frame_inference_times)
            / len(frame_inference_times)
            if frame_inference_times else 0.0
        )
        _write_metrics(job_dir / "metrics.txt", {
            "num_frames": len(selected_frames),
            "num_raw_detections": raw_detections,
            "num_fused_objects": num_fused_objects,
            "mean_observations_per_fused_object": round(mean_observations, 4),
            "cutr_inference_time_per_frame_s": round(mean_frame_time, 4),
            "cutr_inference_time_per_frame_std_s": round(math.sqrt(variance), 4),
            "cutr_inference_time_total_s": round(total_inference_time, 4),
            "fusion_deduplication_time_s": round(fusion_deduplication_time, 4),
            "total_map_construction_time_s": round(total_map_time, 4),
            "crop_generation_time_s": round(crop_generation_time, 4),
        })

        manifest = {
            "job_id": job_id,
            "inference_mode": inference_mode,
            "uploaded_frames": len(frames),
            "processed_frames": len(selected_frames),
            "valid_depth_frames": len(rgbd_frames),
            "frames": [
                {
                    "frame_id": frame["id"],
                    "has_depth": frame["depth_mm"] is not None,
                    "used_for_inference": frame["id"]
                    in {selected["id"] for selected in selected_frames},
                    "depth_error": frame.get("depth_error"),
                }
                for frame in frames
            ],
        }
        write_manifest(job_dir / "manifest.json", manifest)
        log_path.write_text(
            "\n".join(
                [
                    f"inference_mode={inference_mode}",
                    f"uploaded_frames={len(frames)}",
                    f"processed_frames={len(selected_frames)}",
                    f"valid_depth_frames={len(rgbd_frames)}",
                    f"detections={len(fused_pred['detections'])}",
                    f"inference_time_s={total_inference_time:.4f}",
                    *[
                        f"object_{index}_observations={len(cluster.observations)}"
                        for index, cluster in enumerate(clusters)
                    ],
                    *[
                        (
                            f"object_{index}_orientation="
                            f"mode:{representative.orientation_mode},"
                            f"accepted:{representative.orientation_observations},"
                            f"rejected:{representative.orientation_rejected},"
                            f"dispersion_deg:{representative.orientation_dispersion_deg:.3f},"
                            f"raw_tilt_deg:{representative.raw_tilt_deg:.3f},"
                            f"final_tilt_deg:{representative.final_tilt_deg:.3f},"
                            f"accepted_frames:{'|'.join(representative.orientation_accepted_frames)},"
                            f"rejected_frames:{'|'.join(representative.orientation_rejected_frames)}"
                        )
                        for index, representative in enumerate(representatives)
                    ],
                    *[
                        (
                            f"object_{object_index}_frame_{observation.frame_id}="
                            f"camera_pitch_deg:{observation.camera_pitch_deg:.3f},"
                            f"raw_tilt_deg:{observation.raw_tilt_deg:.3f},"
                            f"score:{observation.score:.4f}"
                        )
                        for object_index, cluster in enumerate(clusters)
                        for observation in cluster.observations
                    ],
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        zip_path = job_dir / f"{job_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in (pred_path, job_dir / "manifest.json", log_path, job_dir / "metrics.txt"):
                archive.write(path, arcname=path.name)
            for path in sorted(representatives_dir.glob("*.jpg")):
                archive.write(path, arcname=f"representatives/{path.name}")
            for frame in frames:
                aligned = frame["dir"] / "depth_aligned.png"
                if aligned.exists():
                    archive.write(aligned, arcname=f"depth_aligned/{frame['id']}.png")

        num_detections = len(fused_pred["detections"])
        _job_set(
            job_id,
            status=JOB_DONE,
            result_zip=str(zip_path),
            pred_path=str(pred_path),
            processed_frames=len(selected_frames),
            num_detections=num_detections,
            inference_mode=inference_mode,
        )
        _persist_status(
            job_id,
            JOB_DONE,
            uploaded_frames=len(frames),
            processed_frames=len(selected_frames),
            num_detections=num_detections,
            inference_mode=inference_mode,
            cutr_inference_time_s=round(total_inference_time, 4),
        )
    except Exception as exc:
        traceback_text = traceback.format_exc()
        log_path.write_text(traceback_text, encoding="utf-8")
        _job_set(job_id, status=JOB_ERROR, error=str(exc))
        _persist_status(job_id, JOB_ERROR, error=str(exc))


@app.post("/cutr/multiview/sessions", response_model=JobStartResponse)
def create_multiview_session():
    job_id = _unique_job_id()
    job_dir = _job_dir(job_id)
    (job_dir / "frames").mkdir(parents=True, exist_ok=True)
    _job_set(job_id, status=JOB_COLLECTING, error=None, uploaded_frames=0)
    _persist_status(job_id, JOB_COLLECTING, uploaded_frames=0, processed_frames=0)
    return JobStartResponse(job_id=job_id)


@app.put("/cutr/multiview/sessions/{job_id}/frames/{frame_id}")
def upload_multiview_frame(job_id: str, frame_id: str, req: MultiFrameRequest):
    if not _FRAME_ID_RE.fullmatch(frame_id):
        raise HTTPException(status_code=400, detail="Invalid frame_id")
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    if job.get("status") != JOB_COLLECTING:
        raise HTTPException(status_code=409, detail="Session is not collecting frames")
    frame_dir = _job_dir(job_id) / "frames" / frame_id
    if not frame_dir.exists() and len(_frame_dirs(_job_dir(job_id))) >= 12:
        raise HTTPException(status_code=422, detail="A session accepts at most 12 frames")
    try:
        image = _decode_image(req.image_base64)
        camera_to_world(req.meta_json)
        for key in ("fx", "fy", "cx", "cy"):
            value = float(req.meta_json[key])
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if req.depth_base64 is not None:
            if req.depth_meta is None:
                raise ValueError("depth_meta is required with depth_base64")
            depth = decode_depth_mm(req.depth_base64, req.depth_meta)
            if not validate_depth(depth):
                raise ValueError("depth has insufficient valid metric samples")
            _ = req.depth_meta["projection_matrix"]
            _ = req.depth_meta["camera_to_world"]
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    frame_dir.mkdir(parents=True, exist_ok=True)
    image.save(frame_dir / "input.jpg", format="JPEG", quality=90)
    (frame_dir / "meta.json").write_text(
        json.dumps(req.meta_json, indent=2),
        encoding="utf-8",
    )
    if req.depth_base64 is not None and req.depth_meta is not None:
        (frame_dir / "depth.bin.gz").write_bytes(
            base64.b64decode(req.depth_base64, validate=True)
        )
        (frame_dir / "depth_meta.json").write_text(
            json.dumps(req.depth_meta, indent=2),
            encoding="utf-8",
        )
    else:
        (frame_dir / "depth.bin.gz").unlink(missing_ok=True)
        (frame_dir / "depth_meta.json").unlink(missing_ok=True)
    uploaded_frames = len(_frame_dirs(_job_dir(job_id)))
    _job_set(job_id, uploaded_frames=uploaded_frames)
    _persist_status(
        job_id,
        JOB_COLLECTING,
        uploaded_frames=uploaded_frames,
        processed_frames=0,
    )
    return {"job_id": job_id, "frame_id": frame_id, "uploaded_frames": uploaded_frames}


@app.post("/cutr/multiview/sessions/{job_id}/finalize", response_model=JobStartResponse)
def finalize_multiview_session(job_id: str, req: MultiFinalizeRequest):
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Session not found")
    if job.get("status") != JOB_COLLECTING:
        raise HTTPException(status_code=409, detail="Session is not collecting frames")
    frame_count = len(_frame_dirs(_job_dir(job_id)))
    if frame_count < 3 or frame_count > 12:
        raise HTTPException(status_code=422, detail="A session must contain 3 to 12 frames")
    _job_set(job_id, status=JOB_QUEUED, uploaded_frames=frame_count)
    _persist_status(
        job_id,
        JOB_QUEUED,
        uploaded_frames=frame_count,
        processed_frames=0,
    )
    _executor.submit(_run_multiview_job, job_id, req)
    return JobStartResponse(job_id=job_id)


@app.post("/cutr/jobs", response_model=JobStartResponse)
def create_job(req: CutrRunRequest):
    job_id = _unique_job_id()
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    # Persist inputs immediately so the job is reproducible from disk.
    image_path = d / "input.png"
    meta_path  = d / "meta.json"
    image_path.write_bytes(base64.b64decode(req.image_base64))
    meta_path.write_text(json.dumps(req.meta_json, indent=2))

    _job_set(job_id, status=JOB_QUEUED, error=None)
    _persist_status(job_id, JOB_QUEUED)

    def worker():
        from PIL import Image

        _job_set(job_id, status=JOB_RUNNING)
        _persist_status(job_id, JOB_RUNNING)
        log_path = d / "run.log"
        try:
            mods = _load_cutr()
            meta = mods["load_meta_json"](meta_path)

            runner = _get_runner(req.model_path, req.device)
            img = Image.open(image_path).convert("RGB")
            w, h = img.size

            fx = mods["_meta_intrinsic"](meta, "fx")
            fy = mods["_meta_intrinsic"](meta, "fy")
            cx = mods["_meta_intrinsic"](meta, "cx")
            cy = mods["_meta_intrinsic"](meta, "cy")

            K_user = None
            if any(v is not None for v in (fx, fy, cx, cy)):
                K_user = mods["make_default_intrinsics"](w, h)
                if fx is not None: K_user[0, 0] = float(fx)
                if fy is not None: K_user[1, 1] = float(fy)
                if cx is not None: K_user[0, 2] = float(cx)
                if cy is not None: K_user[1, 2] = float(cy)

            max_edge = None if req.max_edge is None or int(req.max_edge) <= 0 else int(req.max_edge)

            _t0 = time.perf_counter()
            pred = runner.infer(
                image=img,
                K=K_user,
                depth_m=None,
                score_thresh=float(req.score_thresh),
                max_edge=max_edge,
            )
            cutr_inference_time_s = time.perf_counter() - _t0

            pred_path = d / "pred.json"
            mods["save_pred_json"](pred, image_path=image_path, out_path=pred_path)

            crops_dir = d / "crops"
            save_detection_crops(img, pred.get("detections") or [], crops_dir)

            num_detections = len(pred.get("detections", []))
            with open(log_path, "w") as f:
                f.write(f"detections={num_detections}\n")

            # Zip everything in the job folder for /download.
            zip_path = d / f"{job_id}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in (pred_path, image_path, meta_path, log_path):
                    if p.exists():
                        zf.write(p, arcname=p.name)
                if crops_dir.exists():
                    for p in sorted(crops_dir.glob("*.jpg")):
                        zf.write(p, arcname=f"crops/{p.name}")

            _job_set(job_id, status=JOB_DONE, result_zip=str(zip_path), pred_path=str(pred_path))
            _persist_status(
                job_id, JOB_DONE,
                num_detections=num_detections,
                cutr_inference_time_s=round(cutr_inference_time_s, 4),
            )
        except Exception as e:
            tb = traceback.format_exc()
            try:
                log_path.write_text(tb)
            except Exception:
                pass
            _job_set(job_id, status=JOB_ERROR, error=str(e))
            _persist_status(job_id, JOB_ERROR, error=str(e))

    _executor.submit(worker)
    return JobStartResponse(job_id=job_id)


@app.get("/cutr/jobs/{job_id}", response_model=JobStatus)
def job_status(job_id: str):
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(status=str(job.get("status", JOB_ERROR)), error=job.get("error"))


@app.get("/cutr/jobs/{job_id}/download")
def download(job_id: str):
    job = _job_get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != JOB_DONE:
        raise HTTPException(status_code=409, detail=f"Job not finished (status={job.get('status')})")
    zip_path = job.get("result_zip") or str(_job_dir(job_id) / f"{job_id}.zip")
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=500, detail="Result ZIP missing")
    return StreamingResponse(
        open(zip_path, "rb"),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )


"""
uvicorn cutr_api:app --host 0.0.0.0 --port 8090 --workers 1
"""
