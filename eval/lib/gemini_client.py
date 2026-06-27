from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import List, Optional

from google import genai
from google.genai import types

_client: Optional[genai.Client] = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()
    return _client


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def call_gemini(
    prompt: str,
    images: List[Path],
    model: str,
    json_mode: bool = True,
    temperature: float = 0.0,
) -> str:
    client = _get_client()
    parts = []
    for img_path in images:
        suffix = img_path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else "image/png"
        parts.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type=mime))
    parts.append(types.Part.from_text(text=prompt))

    config_kwargs: dict = {"temperature": temperature}
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(**config_kwargs),
    )
    return response.text or ""


def call_gemini_json(
    prompt: str,
    images: List[Path],
    model: str,
    temperature: float = 0.0,
) -> dict:
    raw = call_gemini(prompt, images, model, json_mode=True, temperature=temperature)
    return json.loads(raw)
