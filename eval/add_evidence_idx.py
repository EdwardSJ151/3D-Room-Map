"""Enrich expected_visible_evidence in questions.json with idx references.

For each scene, loads ground_truth.json (bbox descriptions with idxs) and
questions.json, then asks an LLM to match each evidence string to the
relevant bbox idx(es). Writes the updated questions.json in-place.

Skips:
  - negative_object_existence questions (absent objects have no idx)
  - questions where expected_visible_evidence already contains "idx="
  - questions with no expected_visible_evidence

Usage:
    python eval/add_evidence_idx.py [--scene-id room01] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — mirrors eval/config.py but self-contained
# ---------------------------------------------------------------------------
REPO_ROOT   = Path(__file__).resolve().parent.parent
EVAL_DIR    = REPO_ROOT / "eval"
SCENES_JSON = EVAL_DIR / "scenes" / "scenes.json"
RESULTS_DIR = EVAL_DIR / "results"

VLLM_BASE  = os.environ.get("VLLM_JUDGE_BASE_URL", "http://localhost:8035/v1")
VLLM_MODEL = os.environ.get("VLLM_JUDGE_MODEL",    "QuantTrio/Qwen3.5-397B-A17B-AWQ")

SKIP_CATEGORIES = {"negative_object_existence"}

# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key="EMPTY", base_url=VLLM_BASE)
    return _client


def call_llm(prompt: str) -> dict:
    resp = _get_client().chat.completions.create(
        model=VLLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
PROMPT_TEMPLATE = """\
You are an evidence mapper for an indoor scene evaluation dataset.

You are given:
- A list of detected bounding-box objects with their idx numbers
- An evidence string describing which objects should be visible to answer a question

Your task: for each distinct object mentioned in the evidence string, find the \
bbox idx(es) where that object is the PRIMARY subject of the detection — not merely \
mentioned in the description as background or context. If multiple bboxes have that \
object as their primary subject, list all valid alternatives.

BBOX OBJECTS:
{bbox_list}

EVIDENCE STRING:
{evidence}

QUESTION (for context):
{question}

RULES:
- Only assign an idx if the object is the CENTRAL/PRIMARY subject of that bbox. \
Do NOT assign an idx just because the object appears in the description or background \
of a scene-level bbox whose main subject is something else.
- Only assign idx values that exist in the bbox list above.
- For evidence items that match a bbox: append the idx in parentheses, e.g. "laptop (idx=2)".
- For evidence items that match no bbox as primary subject: write the plain object name \
with NO parentheses and NO idx reference of any kind. Never write "idx" for an unmatched item.
- Group alternatives for the same object together (e.g. idx=2 or idx=9).
- Return a JSON object with a single key "updated_evidence" containing the \
rewritten evidence string. Matched and unmatched items are separated by ", ".
  Format matched items as: object name (idx=N) or object name (idx=N or idx=M)
  Format unmatched items as: object name

Example output:
{{"updated_evidence": "laptop (idx=2 or idx=9), bed (idx=6), keyboard"}}

Return ONLY valid JSON."""


# ---------------------------------------------------------------------------
# Scene processing
# ---------------------------------------------------------------------------

def scene_result_file(scene_id: str, filename: str) -> Path:
    return RESULTS_DIR / scene_id / filename


def build_bbox_list(gt: dict) -> str:
    lines = []
    for b in gt.get("bbox_descriptions", []):
        lines.append(
            f"  idx={b['idx']}: {b['object_name']} — {b.get('rich_description', '')} "
            f"(location: {b.get('location_hint', '')})"
        )
    return "\n".join(lines) or "(none)"


def process_scene(scene_id: str, dry_run: bool, force: bool = False) -> None:
    gt_path = scene_result_file(scene_id, "ground_truth.json")
    q_path  = scene_result_file(scene_id, "questions.json")

    if not gt_path.exists():
        print(f"[{scene_id}] ground_truth.json not found, skipping")
        return
    if not q_path.exists():
        print(f"[{scene_id}] questions.json not found, skipping")
        return

    gt        = json.loads(gt_path.read_text())
    q_data    = json.loads(q_path.read_text())
    questions = q_data.get("questions", [])
    bbox_list = build_bbox_list(gt)

    changed = 0
    for q in questions:
        category = q.get("category", "")
        evidence = q.get("expected_visible_evidence", "") or ""

        if category in SKIP_CATEGORIES:
            continue
        if not evidence.strip():
            continue
        if "idx=" in evidence and not force:
            continue

        prompt = PROMPT_TEMPLATE.format(
            bbox_list=bbox_list,
            evidence=evidence,
            question=q.get("question", ""),
        )

        try:
            result = call_llm(prompt)
            updated = result.get("updated_evidence", "").strip()
        except Exception as e:
            print(f"  [{scene_id}] {q['question_id']} LLM error: {e}", flush=True)
            continue

        if not updated or "idx=" not in updated:
            print(f"  [{scene_id}] {q['question_id']} — no idx found, keeping original: {evidence!r}")
            continue

        print(f"  [{scene_id}] {q['question_id']}")
        print(f"    before: {evidence}")
        print(f"    after:  {updated}")
        q["expected_visible_evidence"] = updated
        changed += 1

    if changed == 0:
        print(f"[{scene_id}] nothing to update")
        return

    if dry_run:
        print(f"[{scene_id}] dry-run: {changed} question(s) would be updated")
    else:
        q_path.write_text(json.dumps(q_data, indent=2, ensure_ascii=False))
        print(f"[{scene_id}] updated {changed} question(s) → {q_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene-id", default=None, help="Process a single scene")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    parser.add_argument("--force", action="store_true", help="Re-run even if idx= already present")
    args = parser.parse_args()

    scenes = json.loads(SCENES_JSON.read_text())
    if args.scene_id:
        scenes = [s for s in scenes if s["scene_id"] == args.scene_id]
        if not scenes:
            print(f"scene_id '{args.scene_id}' not found")
            sys.exit(1)

    for scene in scenes:
        process_scene(scene["scene_id"], dry_run=args.dry_run, force=args.force)


if __name__ == "__main__":
    main()
