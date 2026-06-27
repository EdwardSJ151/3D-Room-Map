"""Generate ground-truth annotations for each scene using Gemini.

Usage:
    python eval/02_generate_ground_truth.py [--scene-id scene_01] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.config import (
    GEMINI_GT_MODEL,
    GENERATION_BACKEND,
    GT_BBOX_PROMPT,
    GT_ROOM_PROMPT,
    PROMPT_VERSION,
    SCENES_JSON,
    VLLM_GT_MODEL,
)
from eval.lib.generation_client import call_gt
from eval.lib.image_utils import crop_bbox
from eval.lib.paths import CUTR_JOBS, scene_result_file


def load_scenes(scene_id: str | None = None) -> list:
    scenes = json.loads(SCENES_JSON.read_text())
    if scene_id:
        scenes = [s for s in scenes if s["scene_id"] == scene_id]
        if not scenes:
            raise ValueError(f"scene_id '{scene_id}' not found in scenes.json")
    return scenes


def process_scene(scene: dict, force: bool = False) -> None:
    scene_id = scene["scene_id"]
    mapping_path = scene_result_file(scene_id, "mapping.json")
    out_path = scene_result_file(scene_id, "ground_truth.json")

    if not mapping_path.exists():
        print(f"[{scene_id}] mapping.json not found, skipping")
        return
    if out_path.exists() and not force:
        print(f"[{scene_id}] ground_truth.json exists, skipping (--force to redo)")
        return

    mapping = json.loads(mapping_path.read_text())
    job_id = mapping["job_id"]
    num_records = mapping.get("num_generated_records", 0)
    model_name = VLLM_GT_MODEL if GENERATION_BACKEND == "vllm" else GEMINI_GT_MODEL

    if num_records == 0:
        print(f"[{scene_id}] WARNING: num_generated_records=0, writing empty ground_truth")
        out_path.write_text(json.dumps({
            "scene_id": scene_id, "job_id": job_id,
            "generation_backend": GENERATION_BACKEND,
            "model": model_name, "prompt_version": PROMPT_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "room_type": "unknown", "room_description": "",
            "objects_present": [], "objects_absent_examples": [],
            "bbox_descriptions": [],
        }, indent=2))
        return

    print(f"\n=== [{scene_id}] Generating ground truth (job={job_id}) ===")

    image_path = CUTR_JOBS / job_id / "input.png"
    pred_path  = CUTR_JOBS / job_id / "pred.json"

    if not image_path.exists():
        print(f"[{scene_id}] ERROR: input.png not found at {image_path}", file=sys.stderr)
        return
    if not pred_path.exists():
        print(f"[{scene_id}] ERROR: pred.json not found at {pred_path}", file=sys.stderr)
        return

    pred = json.loads(pred_path.read_text())
    detections = pred.get("detections") or []

    # Whole-room call
    print(f"  [{GENERATION_BACKEND}] whole-room analysis...", flush=True)
    room_data = call_gt(GT_ROOM_PROMPT, [image_path])

    # Per-bbox calls
    bbox_descriptions = []
    print(f"  [{GENERATION_BACKEND}] per-bbox ({len(detections)} detections)...", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, det in enumerate(detections):
            bbox = det.get("bbox_xyxy")
            if not bbox or len(bbox) != 4:
                continue
            crop_path = Path(tmp) / f"crop_{i}.jpg"
            try:
                crop_bbox(image_path, tuple(bbox), out_path=crop_path)
                bbox_data = call_gt(GT_BBOX_PROMPT, [crop_path])
            except Exception as e:
                print(f"  [gemini] bbox {i} failed: {e}", flush=True)
                bbox_data = {
                    "object_name": "(error)", "rich_description": "",
                    "attributes": {}, "location_hint": "",
                }
            bbox_descriptions.append({
                "idx": i,
                "object_name": bbox_data.get("object_name", ""),
                "rich_description": bbox_data.get("rich_description", ""),
                "attributes": bbox_data.get("attributes", {}),
                "location_hint": bbox_data.get("location_hint", ""),
            })

    ground_truth = {
        "scene_id": scene_id,
        "job_id": job_id,
        "generation_backend": GENERATION_BACKEND,
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "room_type": room_data.get("room_type", "unknown"),
        "room_description": room_data.get("room_description", ""),
        "objects_present": room_data.get("objects_present", []),
        "objects_absent_examples": room_data.get("objects_absent_examples", []),
        "bbox_descriptions": bbox_descriptions,
    }
    out_path.write_text(json.dumps(ground_truth, indent=2))
    print(f"  [done] ground_truth.json written → {out_path}")


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
