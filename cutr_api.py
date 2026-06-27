from __future__ import annotations

import base64
import json
import os
import random
import string
import sys
import threading
import time
import traceback
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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


class JobStatus(BaseModel):
    status: str
    error: Optional[str] = None


# App
app = FastAPI(title="CuTR Inference API")


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

            num_detections = len(pred.get("detections", []))
            with open(log_path, "w") as f:
                f.write(f"detections={num_detections}\n")

            # Zip everything in the job folder for /download.
            zip_path = d / f"{job_id}.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in (pred_path, image_path, meta_path, log_path):
                    if p.exists():
                        zf.write(p, arcname=p.name)

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
