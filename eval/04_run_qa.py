"""Run all QA questions through the ADK agent and collect results.

Usage:
    python eval/04_run_qa.py [--scene-id scene_01] [--force]
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


def process_scene(scene: dict, force: bool = False) -> None:
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
    if out_path.exists() and not force:
        print(f"[{scene_id}] qa_results.json exists, skipping (--force to redo)")
        return

    questions_data = json.loads(questions_path.read_text())
    mapping        = json.loads(mapping_path.read_text())
    job_id         = mapping["job_id"]
    num_records    = mapping.get("num_generated_records", 0)
    questions      = questions_data.get("questions", [])

    run_id = uuid.uuid4().hex[:8]
    print(f"\n=== [{scene_id}] Running QA (job={job_id}, run_id={run_id}) ===", flush=True)

    results = []
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

        results.append(result_entry)

    output = {
        "scene_id": scene_id,
        "job_id": job_id,
        "eval_run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_records_in_scene": num_records,
        "results": results,
    }
    out_path.write_text(json.dumps(output, indent=2))
    print(f"  [done] {len(results)} QA results → {out_path}")


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
