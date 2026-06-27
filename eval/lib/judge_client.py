from __future__ import annotations

import json
from pathlib import Path
from typing import List

from eval.config import (
    JUDGE_BACKEND,
    JUDGE_MODEL,
    VLLM_JUDGE_MODEL,
)
from eval.lib.gemini_client import call_gemini_json
from eval.lib.vllm_client import call_vllm_json


def call_judge(
    prompt: str,
    images: List[Path],
    json_mode: bool = True,
) -> dict:
    if JUDGE_BACKEND == "vllm":
        return call_vllm_json(prompt, images, model=VLLM_JUDGE_MODEL)
    else:
        return call_gemini_json(prompt, images, model=JUDGE_MODEL)
