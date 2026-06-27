from __future__ import annotations

from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parent.parent.parent
EVAL_DIR    = REPO_ROOT / "eval"
SCENES_DIR  = EVAL_DIR / "scenes"
RESULTS_DIR = EVAL_DIR / "results"
CUTR_JOBS   = REPO_ROOT / "cutr_jobs"
QWEN_JOBS   = REPO_ROOT / "qwen_jobs"


def scene_result_dir(scene_id: str) -> Path:
    d = RESULTS_DIR / scene_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def scene_result_file(scene_id: str, filename: str) -> Path:
    return scene_result_dir(scene_id) / filename
