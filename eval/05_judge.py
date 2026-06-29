"""Judge each QA answer using the configured judge (Gemini or vLLM).

Usage:
    python eval/05_judge.py [--scene-id scene_01] [--force] [--failed-only]
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
    JUDGE_BACKEND,
    JUDGE_MODEL,
    JUDGE_PROMPT_TEMPLATE,
    PROMPT_VERSION,
    SCENES_JSON,
    VLLM_JUDGE_MODEL,
)
from eval.lib.image_utils import crop_bbox
from eval.lib.judge_client import call_judge
from eval.lib.paths import CUTR_JOBS, scene_result_file


def load_scenes(scene_id: str | None = None) -> list:
    scenes = json.loads(SCENES_JSON.read_text())
    if scene_id:
        scenes = [s for s in scenes if s["scene_id"] == scene_id]
        if not scenes:
            raise ValueError(f"scene_id '{scene_id}' not found in scenes.json")
    return scenes


def build_judge_prompt(
    question: dict,
    qa_result: dict,
    gt: dict,
    gt_bbox_map: dict,
) -> str:
    retrieved_idxs = [r.get("idx") for r in qa_result.get("retrieved_records", []) if r.get("idx") is not None]
    bbox_gt_descs = []
    for idx in retrieved_idxs:
        b = gt_bbox_map.get(idx)
        if b:
            bbox_gt_descs.append(
                f"idx={idx}: {b['object_name']} — {b['rich_description']} "
                f"(location: {b['location_hint']})"
            )

    return JUDGE_PROMPT_TEMPLATE.format(
        room_description=gt.get("room_description", ""),
        bbox_gt_descriptions="\n".join(bbox_gt_descs) or "(none)",
        category=question.get("category", ""),
        question=question.get("question", ""),
        expected_answer=question.get("expected_answer", ""),
        assistant_answer=qa_result.get("assistant_answer", ""),
        tool_was_called=qa_result.get("tool_was_called", False),
        retrieved_idx_list=retrieved_idxs,
        target_idx=question.get("target_idx"),
    )


def collect_images(qa_result: dict, image_path: Path, pred: dict, tmp_dir: Path) -> list[Path]:
    images = [image_path]
    detections = pred.get("detections") or []
    for r in qa_result.get("retrieved_records", []):
        idx = r.get("idx")
        if idx is None or idx >= len(detections):
            continue
        bbox = detections[idx].get("bbox_xyxy")
        if not bbox or len(bbox) != 4:
            continue
        crop_path = tmp_dir / f"crop_{idx}.jpg"
        try:
            crop_bbox(image_path, tuple(bbox), out_path=crop_path)
            images.append(crop_path)
        except Exception:
            pass
    return images


def process_scene(scene: dict, force: bool = False, failed_only: bool = False) -> None:
    scene_id   = scene["scene_id"]
    gt_path    = scene_result_file(scene_id, "ground_truth.json")
    qa_path    = scene_result_file(scene_id, "qa_results.json")
    q_path     = scene_result_file(scene_id, "questions.json")
    out_path   = scene_result_file(scene_id, "judgments.json")
    mapping_path = scene_result_file(scene_id, "mapping.json")

    for p, name in [(gt_path, "ground_truth.json"), (qa_path, "qa_results.json"),
                    (q_path, "questions.json"), (mapping_path, "mapping.json")]:
        if not p.exists():
            print(f"[{scene_id}] {name} not found, skipping")
            return

    # Load existing judgments to find which question IDs previously failed.
    prev_judgments: dict[str, dict] = {}
    failed_ids: set[str] = set()
    if failed_only:
        if not out_path.exists():
            print(f"[{scene_id}] judgments.json not found, skipping (--failed-only requires prior phase 5 output)")
            return
        prev_data = json.loads(out_path.read_text())
        prev_judgments = {j["question_id"]: j for j in prev_data.get("judgments", [])}
        failed_ids = {qid for qid, j in prev_judgments.items() if j.get("judgment") != "success"}
        if not failed_ids:
            print(f"[{scene_id}] no failures found, nothing to re-judge")
            return
        print(f"[{scene_id}] re-judging {len(failed_ids)} failed question(s)")
    elif out_path.exists() and not force:
        print(f"[{scene_id}] judgments.json exists, skipping (--force to redo)")
        return

    gt            = json.loads(gt_path.read_text())
    qa_data       = json.loads(qa_path.read_text())
    questions_raw = json.loads(q_path.read_text()).get("questions", [])
    mapping       = json.loads(mapping_path.read_text())
    job_id        = mapping["job_id"]

    gt_bbox_map = {b["idx"]: b for b in gt.get("bbox_descriptions", [])}
    q_map       = {q["question_id"]: q for q in questions_raw}

    image_path = CUTR_JOBS / job_id / "input.png"
    pred_path  = CUTR_JOBS / job_id / "pred.json"
    pred       = json.loads(pred_path.read_text()) if pred_path.exists() else {}

    judge_model_name = VLLM_JUDGE_MODEL if JUDGE_BACKEND == "vllm" else JUDGE_MODEL
    print(f"\n=== [{scene_id}] Judging ({JUDGE_BACKEND}/{judge_model_name}) ===", flush=True)

    judgments = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, qa_result in enumerate(qa_data.get("results", [])):
            qid = qa_result["question_id"]
            question = q_map.get(qid, {})

            if failed_only and qid not in failed_ids:
                judgments.append(prev_judgments[qid])
                continue

            print(f"  [{i+1}] {qid}", flush=True)

            # Auto-fail when tool wasn't called
            if not qa_result.get("tool_was_called", False):
                judgments.append({
                    "question_id": qid,
                    "judgment": "failure",
                    "failure_reason": "retrieval_error",
                    "judge_explanation": "Agent did not call the object memory tool.",
                    "retrieved_idx_correct": None,
                })
                continue

            prompt = build_judge_prompt(question, qa_result, gt, gt_bbox_map)
            images = collect_images(qa_result, image_path, pred, tmp_dir)

            try:
                verdict = call_judge(prompt, images)
            except Exception as e:
                print(f"    judge error: {e}", flush=True)
                verdict = {
                    "judgment": "failure",
                    "failure_reason": "ambiguous_question",
                    "judge_explanation": f"Judge call failed: {e}",
                    "retrieved_idx_correct": None,
                }

            target_idx = question.get("target_idx")
            judgments.append({
                "question_id": qid,
                "judgment": verdict.get("judgment", "failure"),
                "failure_reason": verdict.get("failure_reason"),
                "judge_explanation": verdict.get("judge_explanation", ""),
                "retrieved_idx_correct": verdict.get("retrieved_idx_correct") if target_idx is not None else None,
            })

    output = {
        "scene_id": scene_id,
        "judge_backend": JUDGE_BACKEND,
        "judge_model": judge_model_name,
        "prompt_version": PROMPT_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "judgments": judgments,
    }
    if failed_only and prev_data:
        output["eval_run_id"] = prev_data.get("eval_run_id", output.get("eval_run_id"))
    out_path.write_text(json.dumps(output, indent=2))
    successes = sum(1 for j in judgments if j["judgment"] == "success")
    print(f"  [done] {successes}/{len(judgments)} success → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--failed-only", action="store_true")
    args = parser.parse_args()

    scenes = load_scenes(args.scene_id)
    for scene in scenes:
        process_scene(scene, force=args.force, failed_only=args.failed_only)


if __name__ == "__main__":
    main()
