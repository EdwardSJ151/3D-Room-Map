"""Run the full mapping pipeline (CuTR → Qwen → vectorstore) for each scene.

Usage:
    python eval/01_run_mapping.py [--scene-id scene_01] [--force]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.config import (
    CUTR_BASE,
    POLL_INTERVAL_S,
    POLL_TIMEOUT_S,
    QWEN_BASE,
    SCENES_JSON,
)
from eval.lib.paths import CUTR_JOBS, QWEN_JOBS, scene_result_file
from eval.lib.poll import poll_until_done


def load_scenes(scene_id: str | None = None) -> list:
    scenes = json.loads(SCENES_JSON.read_text())
    if scene_id:
        scenes = [s for s in scenes if s["scene_id"] == scene_id]
        if not scenes:
            raise ValueError(f"scene_id '{scene_id}' not found in scenes.json")
    return scenes


def run_cutr(image_b64: str, meta_json: dict) -> tuple[str, dict]:
    resp = requests.post(
        f"{CUTR_BASE}/cutr/jobs",
        json={"image_base64": image_b64, "meta_json": meta_json, "score_thresh": 0.25},
        timeout=60,
    )
    resp.raise_for_status()
    job_id = resp.json()["job_id"]
    print(f"  [cutr] job_id={job_id}", flush=True)
    poll_until_done(
        f"{CUTR_BASE}/cutr/jobs/{job_id}",
        "cutr",
        interval_s=POLL_INTERVAL_S,
        timeout_s=POLL_TIMEOUT_S,
    )
    status = json.loads((CUTR_JOBS / job_id / "status.json").read_text())
    return job_id, status


def run_qwen(job_id: str, image_b64: str, pred: dict) -> dict:
    resp = requests.post(
        f"{QWEN_BASE}/qwen/jobs",
        json={"job_id": job_id, "image_base64": image_b64, "pred": pred},
        timeout=600,
    )
    resp.raise_for_status()
    vs = poll_until_done(
        f"{QWEN_BASE}/qwen/jobs/{job_id}/vectorstore",
        "vectorstore",
        interval_s=POLL_INTERVAL_S,
        timeout_s=POLL_TIMEOUT_S,
    )
    timing_path = QWEN_JOBS / job_id / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    return {"n": vs.get("n", 0), "timing": timing}


def process_scene(scene: dict, force: bool = False) -> None:
    scene_id = scene["scene_id"]
    out_path  = scene_result_file(scene_id, "mapping.json")
    if out_path.exists() and not force:
        print(f"[{scene_id}] mapping.json exists, skipping (use --force to redo)")
        return

    image_path = Path(__file__).resolve().parent.parent / scene["image_path"]
    meta_path  = Path(__file__).resolve().parent.parent / scene["meta_path"]

    if not image_path.exists():
        print(f"[{scene_id}] ERROR: image not found at {image_path}", file=sys.stderr)
        return
    if not meta_path.exists():
        print(f"[{scene_id}] ERROR: meta not found at {meta_path}", file=sys.stderr)
        return

    print(f"\n=== [{scene_id}] Running mapping pipeline ===")
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    meta_json = json.loads(meta_path.read_text())

    job_id, cutr_status = run_cutr(image_b64, meta_json)
    num_detections = cutr_status.get("num_detections", 0)
    cutr_time = cutr_status.get("cutr_inference_time_s")
    print(f"  [cutr] detections={num_detections} inference_time={cutr_time}s")

    pred_path = CUTR_JOBS / job_id / "pred.json"
    pred = json.loads(pred_path.read_text())

    qwen_result = run_qwen(job_id, image_b64, pred)
    num_records = qwen_result["n"]
    timing = qwen_result["timing"]
    print(f"  [qwen] records={num_records}")

    crop_time  = timing.get("crop_generation_time_s")
    label_time = timing.get("qwen_labeling_time_s")
    embed_time = timing.get("embedding_indexing_time_s")

    timed = [t for t in [cutr_time, crop_time, label_time, embed_time] if t is not None]
    total_time = round(sum(timed), 4) if timed else None

    mapping = {
        "scene_id": scene_id,
        "job_id": job_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_detections": num_detections,
        "num_generated_records": num_records,
        "cutr_inference_time_s": cutr_time,
        "crop_generation_time_s": crop_time,
        "qwen_labeling_time_s": label_time,
        "embedding_indexing_time_s": embed_time,
        "total_capture_to_memory_time_s": total_time,
    }
    out_path.write_text(json.dumps(mapping, indent=2))
    print(f"  [done] mapping.json written → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    scenes = load_scenes(args.scene_id)
    for scene in scenes:
        process_scene(scene, force=args.force)


if __name__ == "__main__":
    main()
