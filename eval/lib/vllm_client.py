from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from eval.config import VLLM_JUDGE_BASE_URL

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key="EMPTY", base_url=VLLM_JUDGE_BASE_URL)
    return _client


def _encode_image_b64(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def call_vllm(
    prompt: str,
    images: List[Path],
    model: str,
    json_mode: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    client = _get_client()
    content: list = [{"type": "text", "text": prompt}]
    for img_path in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": _encode_image_b64(img_path)},
        })

    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"} if json_mode else None,
    )
    return resp.choices[0].message.content or ""


def call_vllm_json(
    prompt: str,
    images: List[Path],
    model: str,
    temperature: float = 0.0,
) -> dict:
    raw = call_vllm(prompt, images, model, json_mode=True, temperature=temperature)
    return json.loads(raw)
