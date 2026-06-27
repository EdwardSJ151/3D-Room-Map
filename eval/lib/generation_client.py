from __future__ import annotations

import json
from pathlib import Path
from typing import List

from eval.config import (
    GENERATION_BACKEND,
    GEMINI_GT_MODEL,
    GEMINI_QUESTION_MODEL,
    VLLM_GT_MODEL,
    VLLM_QUESTION_MODEL,
    VLLM_GENERATION_BASE_URL,
)
from eval.lib.gemini_client import call_gemini_json
from eval.lib.vllm_client import call_vllm_json


def _vllm_gt_json(prompt: str, images: List[Path]) -> dict:
    from openai import OpenAI
    import base64

    client = OpenAI(api_key="EMPTY", base_url=VLLM_GENERATION_BASE_URL)
    content: list = [{"type": "text", "text": prompt}]
    for img_path in images:
        suffix = img_path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    resp = client.chat.completions.create(
        model=VLLM_GT_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=2048,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")


def _vllm_question_json(prompt: str, images: List[Path]) -> dict:
    from openai import OpenAI
    import base64

    client = OpenAI(api_key="EMPTY", base_url=VLLM_GENERATION_BASE_URL)
    content: list = [{"type": "text", "text": prompt}]
    for img_path in images:
        suffix = img_path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    resp = client.chat.completions.create(
        model=VLLM_QUESTION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=8192,
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content or "{}")


def call_gt(prompt: str, images: List[Path]) -> dict:
    if GENERATION_BACKEND == "vllm":
        return _vllm_gt_json(prompt, images)
    return call_gemini_json(prompt, images, model=GEMINI_GT_MODEL)


def call_questions(prompt: str, images: List[Path]) -> dict:
    if GENERATION_BACKEND == "vllm":
        return _vllm_question_json(prompt, images)
    return call_gemini_json(prompt, images, model=GEMINI_QUESTION_MODEL)
