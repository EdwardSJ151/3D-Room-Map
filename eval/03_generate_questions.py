"""Generate 60 evaluation questions per scene using Gemini.

Usage:
    python eval/03_generate_questions.py [--scene-id scene_01] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.config import (
    GEMINI_QUESTION_MODEL,
    GENERATION_BACKEND,
    PROMPT_VERSION,
    QUESTION_GENERATION_PROMPT,
    SCENES_JSON,
    VLLM_QUESTION_MODEL,
)
from eval.lib.generation_client import call_questions
from eval.lib.paths import CUTR_JOBS, scene_result_file

EXPECTED_CATEGORIES = [
    "positive_object_existence",
    "negative_object_existence",
    "attribute_grounding",
    "category_retrieval",
    "affordance_retrieval",
    "spatial_or_local_relation",
]
QUESTIONS_PER_CAT = 10
TOTAL_QUESTIONS = len(EXPECTED_CATEGORIES) * QUESTIONS_PER_CAT


def load_scenes(scene_id: str | None = None) -> list:
    scenes = json.loads(SCENES_JSON.read_text())
    if scene_id:
        scenes = [s for s in scenes if s["scene_id"] == scene_id]
        if not scenes:
            raise ValueError(f"scene_id '{scene_id}' not found in scenes.json")
    return scenes


def validate_questions(questions: list) -> list[str]:
    errors = []
    texts = set()
    cat_counts: dict[str, int] = {c: 0 for c in EXPECTED_CATEGORIES}
    for q in questions:
        cat = q.get("category", "")
        if cat not in cat_counts:
            errors.append(f"Unknown category: {cat!r}")
        else:
            cat_counts[cat] += 1
        text = q.get("question", "")
        if text in texts:
            errors.append(f"Duplicate question: {text!r}")
        texts.add(text)
    for cat, count in cat_counts.items():
        if count != QUESTIONS_PER_CAT:
            errors.append(f"Category {cat!r}: expected {QUESTIONS_PER_CAT}, got {count}")
    return errors


def build_prompt(gt: dict) -> str:
    bbox_lines = "\n".join(
        f"  idx={b['idx']}: {b['object_name']} — {b['rich_description']} "
        f"(location: {b['location_hint']})"
        for b in gt.get("bbox_descriptions", [])
    )
    context = (
        f"Room type: {gt.get('room_type', 'unknown')}\n"
        f"Room description: {gt.get('room_description', '')}\n\n"
        f"Objects present: {', '.join(gt.get('objects_present', []))}\n\n"
        f"Objects absent (plausible but not present): "
        f"{', '.join(gt.get('objects_absent_examples', []))}\n\n"
        f"Detected bbox objects:\n{bbox_lines}"
    )
    return QUESTION_GENERATION_PROMPT + "\n\n---\n\nSCENE CONTEXT:\n" + context


def process_scene(scene: dict, force: bool = False) -> None:
    scene_id = scene["scene_id"]
    gt_path  = scene_result_file(scene_id, "ground_truth.json")
    out_path = scene_result_file(scene_id, "questions.json")

    if not gt_path.exists():
        print(f"[{scene_id}] ground_truth.json not found, skipping")
        return
    if out_path.exists() and not force:
        print(f"[{scene_id}] questions.json exists, skipping (--force to redo)")
        return

    gt = json.loads(gt_path.read_text())
    if not gt.get("bbox_descriptions"):
        print(f"[{scene_id}] WARNING: no bbox_descriptions, skipping question generation")
        return

    model_name = VLLM_QUESTION_MODEL if GENERATION_BACKEND == "vllm" else GEMINI_QUESTION_MODEL
    print(f"\n=== [{scene_id}] Generating questions ({GENERATION_BACKEND}/{model_name}) ===", flush=True)

    image_path = CUTR_JOBS / gt["job_id"] / "input.png"
    prompt = build_prompt(gt)

    result = call_questions(prompt, [image_path])
    questions_raw = result.get("questions") or []

    errors = validate_questions(questions_raw)
    if errors:
        print(f"  [warn] Validation errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")

    questions = []
    for idx, q in enumerate(questions_raw):
        questions.append({
            "question_id": f"{scene_id}_q{idx+1:03d}",
            "category": q.get("category", ""),
            "question": q.get("question", ""),
            "expected_answer": q.get("expected_answer", ""),
            "expected_visible_evidence": q.get("expected_visible_evidence", ""),
            "acceptable_alternatives": q.get("acceptable_alternatives") or [],
            "requires_absent_object": bool(q.get("requires_absent_object", False)),
            "target_idx": q.get("target_idx"),
        })

    output = {
        "scene_id": scene_id,
        "job_id": gt["job_id"],
        "generation_backend": GENERATION_BACKEND,
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "questions": questions,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  [done] {len(questions)} questions → {out_path}")


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
