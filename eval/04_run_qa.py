"""Run all QA questions through the ADK agent and collect results.

Usage:
    python eval/04_run_qa.py [--scene-id scene_01] [--force] [--failed-only]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.config import SCENES_JSON
from eval.lib.adk_client import create_session, parse_events, run_turn
from eval.lib.paths import scene_result_file


def load_scenes(scene_id: str | None = None) -> list:
    scenes = json.loads(SCENES_JSON.read_text())
    if scene_id:
        scenes = [s for s in scenes if s["scene_id"] == scene_id]
        if not scenes:
            raise ValueError(f"scene_id '{scene_id}' not found in scenes.json")
    return scenes


def get_failed_question_ids(scene_id: str) -> set[str] | None:
    judgments_path = scene_result_file(scene_id, "judgments.json")
    if not judgments_path.exists():
        return None
    judgments = json.loads(judgments_path.read_text())
    return {j["question_id"] for j in judgments.get("judgments", []) if j.get("judgment") == "failure"}


def process_scene(scene: dict, force: bool = False, failed_only: bool = False) -> None:
    scene_id = scene["scene_id"]
    questions_path = scene_result_file(scene_id, "questions.json")
    mapping_path   = scene_result_file(scene_id, "mapping.json")
    out_path       = scene_result_file(scene_id, "qa_results.json")

    if not questions_path.exists():
        print(f"[{scene_id}] questions.json not found, skipping")
        return
    if not mapping_path.exists():
        print(f"[{scene_id}] mapping.json not found, skipping")
        return

    failed_ids: set[str] | None = None
    if failed_only:
        failed_ids = get_failed_question_ids(scene_id)
        if failed_ids is None:
            print(f"[{scene_id}] judgments.json not found, skipping (--failed-only requires phase 5 output)")
            return
        if not failed_ids:
            print(f"[{scene_id}] no failed questions, skipping")
            return
        if not out_path.exists():
            print(f"[{scene_id}] qa_results.json not found, skipping (--failed-only requires existing results)")
            return
        print(f"[{scene_id}] retrying {len(failed_ids)} failed questions")
    elif out_path.exists() and not force:
        print(f"[{scene_id}] qa_results.json exists, skipping (--force to redo)")
        return

    questions_data = json.loads(questions_path.read_text())
    mapping        = json.loads(mapping_path.read_text())
    job_id         = mapping["job_id"]
    num_records    = mapping.get("num_generated_records", 0)
    questions      = questions_data.get("questions", [])

    if failed_only:
        questions = [q for q in questions if q["question_id"] in failed_ids]
        existing  = {r["question_id"]: r for r in json.loads(out_path.read_text()).get("results", [])}
    else:
        existing = {}

    run_id = uuid.uuid4().hex[:8]
    print(f"\n=== [{scene_id}] Running QA (job={job_id}, run_id={run_id}) ===", flush=True)

    new_results = {}
    for i, q in enumerate(questions):
        qid     = q["question_id"]
        text    = q["question"]
        sess_id = f"eval_{qid}_{run_id}"

        print(f"  [{i+1}/{len(questions)}] {qid}: {text[:60]}...", flush=True)

        result_entry = {
            "question_id": qid,
            "category": q["category"],
            "question": text,
            "adk_session_id": sess_id,
            "assistant_answer": "",
            "retrieved_records": [],
            "best_idx": None,
            "retrieval_time_ms": None,
            "response_generation_time_ms": None,
            "total_query_to_answer_time_ms": None,
            "tool_was_called": False,
            "error": None,
        }

        try:
            create_session(sess_id, job_id)
            t0 = time.perf_counter()
            events = run_turn(sess_id, text)
            total_ms = (time.perf_counter() - t0) * 1000.0

            parsed = parse_events(events)
            retrieval_ms = parsed.get("retrieval_time_ms")
            gen_ms = (total_ms - retrieval_ms) if retrieval_ms is not None else None

            result_entry.update({
                "assistant_answer": parsed["assistant_answer"],
                "retrieved_records": parsed["retrieved_records"],
                "best_idx": parsed["best_idx"],
                "retrieval_time_ms": retrieval_ms,
                "response_generation_time_ms": round(gen_ms, 2) if gen_ms is not None else None,
                "total_query_to_answer_time_ms": round(total_ms, 2),
                "tool_was_called": parsed["tool_was_called"],
            })
        except Exception as e:
            result_entry["error"] = str(e)
            print(f"    ERROR: {e}", flush=True)

        new_results[qid] = result_entry

    if failed_only:
        existing.update(new_results)
        all_questions = json.loads(questions_path.read_text()).get("questions", [])
        results = [existing[q["question_id"]] for q in all_questions if q["question_id"] in existing]
    else:
        results = list(new_results.values())

    prev = json.loads(out_path.read_text()) if out_path.exists() else {}
    output = {
        "scene_id": scene_id,
        "job_id": job_id,
        "eval_run_id": prev.get("eval_run_id", run_id) if failed_only else run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_records_in_scene": num_records,
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  [done] {len(new_results)} questions rerun, {len(results)} total → {out_path}")


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
