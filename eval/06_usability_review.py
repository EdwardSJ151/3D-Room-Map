"""Interactive CLI for human usability review of object records (optional, sampled).

Usage:
    python eval/06_usability_review.py --scene-id scene_01 [--sample-size 10] [--sample-rate 0.2]

Keys: u=usable  p=partial  x=unusable  s=skip  q=quit
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.config import SCENES_JSON
from eval.lib.image_utils import crop_bbox, show_crop_cli
from eval.lib.paths import CUTR_JOBS, QWEN_JOBS, scene_result_file


VALID_KEYS = {"u", "p", "x", "s", "q"}
JUDGMENT_MAP = {"u": "usable", "p": "partial", "x": "unusable"}


def load_scenes(scene_id: str | None = None) -> list:
    scenes = json.loads(SCENES_JSON.read_text())
    if scene_id:
        scenes = [s for s in scenes if s["scene_id"] == scene_id]
        if not scenes:
            raise ValueError(f"scene_id '{scene_id}' not found in scenes.json")
    return scenes


def process_scene(
    scene: dict,
    sample_size: int | None = None,
    sample_rate: float | None = None,
) -> None:
    scene_id     = scene["scene_id"]
    mapping_path = scene_result_file(scene_id, "mapping.json")
    out_path     = scene_result_file(scene_id, "usability.json")

    if not mapping_path.exists():
        print(f"[{scene_id}] mapping.json not found, skipping")
        return

    mapping  = json.loads(mapping_path.read_text())
    job_id   = mapping["job_id"]
    num_records = mapping.get("num_generated_records", 0)

    if num_records == 0:
        print(f"[{scene_id}] no records to review")
        return

    objects_path = QWEN_JOBS / job_id / "objects_full.json"
    pred_path    = CUTR_JOBS / job_id / "pred.json"
    image_path   = CUTR_JOBS / job_id / "input.png"

    if not objects_path.exists():
        print(f"[{scene_id}] objects_full.json not found", file=sys.stderr)
        return

    records    = json.loads(objects_path.read_text())
    pred       = json.loads(pred_path.read_text()) if pred_path.exists() else {}
    detections = pred.get("detections") or []

    # Load existing reviews to support resume
    existing: dict[int, dict] = {}
    if out_path.exists():
        saved = json.loads(out_path.read_text())
        for r in saved.get("reviews", []):
            existing[r["idx"]] = r

    # Select sample
    unreviewed = [r for r in records if r["idx"] not in existing]
    if not unreviewed:
        print(f"[{scene_id}] all {len(records)} records already reviewed")
        return

    if sample_size is not None:
        sample = random.sample(unreviewed, min(sample_size, len(unreviewed)))
    elif sample_rate is not None:
        n = max(1, int(len(unreviewed) * sample_rate))
        sample = random.sample(unreviewed, n)
    else:
        sample = unreviewed

    print(f"\n=== [{scene_id}] Usability review: {len(sample)} records ===")
    print("Keys: u=usable  p=partial  x=unusable  s=skip  q=quit\n")

    reviews = list(existing.values())

    for i, rec in enumerate(sample):
        idx = rec["idx"]
        print(f"[{i+1}/{len(sample)}] idx={idx}")
        print(f"  Object:      {rec.get('object', '?')}")
        print(f"  Description: {rec.get('description', '?')}")
        print(f"  Tags:        {', '.join(rec.get('tags') or [])}")

        if idx < len(detections):
            bbox = detections[idx].get("bbox_xyxy")
            if bbox and len(bbox) == 4 and image_path.exists():
                try:
                    crop = crop_bbox(image_path, tuple(bbox))
                    show_crop_cli(crop, label=f"idx={idx} {rec.get('object', '')}")
                except Exception as e:
                    print(f"  (crop failed: {e})")

        while True:
            key = input("  [u/p/x/s/q] ").strip().lower()
            if key in VALID_KEYS:
                break
            print("  Invalid key. Use u/p/x/s/q.")

        if key == "q":
            print("Quitting.")
            break
        if key == "s":
            continue

        notes = input("  Notes (optional): ").strip()
        reviews.append({
            "idx": idx,
            "object_name": rec.get("object", ""),
            "judgment": JUDGMENT_MAP[key],
            "notes": notes,
        })

        # Write incrementally
        output = {
            "scene_id": scene_id,
            "job_id": job_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "reviewed_count": len(reviews),
            "sampled": True,
            "reviews": reviews,
        }
        out_path.write_text(json.dumps(output, indent=2))

    # Summary
    total = len(reviews)
    if total > 0:
        u = sum(1 for r in reviews if r["judgment"] == "usable")
        p = sum(1 for r in reviews if r["judgment"] == "partial")
        x = sum(1 for r in reviews if r["judgment"] == "unusable")
        print(f"\n[{scene_id}] reviewed={total}: usable={u} partial={p} unusable={x}")
    print(f"  Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-rate", type=float, default=None)
    args = parser.parse_args()

    scenes = load_scenes(args.scene_id)
    for scene in scenes:
        process_scene(scene, sample_size=args.sample_size, sample_rate=args.sample_rate)


if __name__ == "__main__":
    main()
