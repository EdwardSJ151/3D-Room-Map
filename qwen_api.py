from __future__ import annotations

import asyncio
import base64
import json
import os
import pickle
import random
import re
import string
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, Field

# ----------------------------
# Config (mirrors inference_timing.ipynb)
# ----------------------------

VLLM_BASE       = os.environ.get("VLLM_BASE", "http://localhost:8010/v1")
VLLM_MODEL      = os.environ.get("VLLM_MODEL", "MY_MODEL")
MAX_NEW_TOKENS  = int(os.environ.get("QWEN_MAX_NEW_TOKENS", "256"))
CONCURRENCY     = int(os.environ.get("QWEN_CONCURRENCY", "16"))
REQUEST_TIMEOUT = float(os.environ.get("QWEN_TIMEOUT_S", "20.0"))
EMBED_MODEL_ID     = os.environ.get("QWEN_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBED_DEVICE       = os.environ.get("QWEN_EMBED_DEVICE", "cpu")   # "cpu" or "cuda"

JOBS_DIR = Path(os.environ.get("QWEN_JOBS_DIR", "qwen_jobs"))
JOBS_DIR.mkdir(parents=True, exist_ok=True)

PROMPT_TAGS = """You are an image tagger.

The image is a crop. List only what is clearly visible in the crop.

Rules:
- English, short nouns (1\u20134 words).
- Do not guess. If unsure, omit.
- No duplicates. Maximum 10 items.

OUTPUT FORMAT (required):
Respond with ONE single line starting exactly with:
Image Tags:
followed by tags separated by " | " (space-pipe-space).

Example:
Image Tags: pen | desk | paper | wood

No JSON. No markdown. No explanation. No reasoning."""

PROMPT_DESCRIBE = """You are an object identification assistant.

The image is a crop from a bounding box detected in a room scene.

Your task:
1. Identify the MAIN object in the crop (the primary subject of the bounding box).
2. Only if relevant, briefly mention the most important secondary details (e.g. notable items on or around the main object).

Rules:
- English only.
- Be concise. One or two sentences maximum.
- Do not list every visible item \u2014 focus on what matters most.
- Do not guess. If the main object is unclear, say so.

OUTPUT FORMAT (required):
Respond with exactly two lines:
Object: <main object name>
Description: <one or two sentence description>

No JSON. No markdown. No extra text. No reasoning."""


# ----------------------------
# Vectorstore state
# ----------------------------

VS_PENDING  = "pending"
VS_BUILDING = "building"
VS_DONE     = "done"
VS_ERROR    = "error"

_vs_lock = threading.Lock()
_vs_state: Dict[str, Dict[str, Any]] = {}
_vs_executor = ThreadPoolExecutor(max_workers=1)

# In-memory caches (FAISS indexes + metadata) for fast queries.
_faiss_cache: Dict[str, Tuple[Any, List[Dict[str, Any]]]] = {}
_embed_model = None
_embed_model_lock = threading.Lock()


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        with _embed_model_lock:
            if _embed_model is None:
                import torch
                from sentence_transformers import SentenceTransformer
                _embed_model = SentenceTransformer(
                    EMBED_MODEL_ID,
                    device=EMBED_DEVICE,
                    model_kwargs={"torch_dtype": torch.float16} if EMBED_DEVICE == "cuda" else {},
                    tokenizer_kwargs={"padding_side": "left"},
                )
    return _embed_model


def _vs_set(job_id: str, **fields):
    with _vs_lock:
        st = _vs_state.setdefault(job_id, {})
        st.update(fields)
        _persist_vs(job_id, st)


def _vs_get(job_id: str) -> Optional[Dict[str, Any]]:
    with _vs_lock:
        st = _vs_state.get(job_id)
        if st:
            return dict(st)
    # Fallback to disk.
    p = _job_dir(job_id) / "vectorstore.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return None


def _persist_vs(job_id: str, st: Dict[str, Any]):
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "vectorstore.json").write_text(json.dumps(st, indent=2))


# ----------------------------
# Helpers
# ----------------------------

def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _new_job_id() -> str:
    letters_a = "".join(random.choices(string.ascii_lowercase, k=6))
    digits    = "".join(random.choices(string.digits, k=2))
    letters_b = "".join(random.choices(string.ascii_lowercase, k=2))
    return f"{letters_a}{digits}{letters_b}"


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _crop_for_detection(img: Image.Image, det: Dict[str, Any]) -> Optional[Image.Image]:
    bbox = det.get("bbox_xyxy")
    if not bbox or len(bbox) != 4:
        return None
    w, h = img.size
    x1, y1, x2, y2 = [float(v) for v in bbox]
    x1 = _clamp(x1, 0.0, w - 1.0); y1 = _clamp(y1, 0.0, h - 1.0)
    x2 = _clamp(x2, 0.0, w - 1.0); y2 = _clamp(y2, 0.0, h - 1.0)
    x1_i = max(0, int(round(x1))); y1_i = max(0, int(round(y1)))
    x2_i = max(0, int(round(x2))); y2_i = max(0, int(round(y2)))
    if x2_i - x1_i < 2 or y2_i - y1_i < 2:
        return None
    return img.crop((x1_i, y1_i, x2_i, y2_i))


def _encode_image_b64(img: Image.Image, quality: int = 85) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def _build_messages(img: Image.Image, prompt: str) -> List[Dict[str, Any]]:
    b64 = _encode_image_b64(img)
    return [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ],
    }]


def _parse_tags(text: str) -> List[str]:
    m = re.search(r"Image Tags:\s*(.+)", text, re.IGNORECASE)
    if not m:
        return []
    line = m.group(1).split("\n")[0]
    return [t.strip() for t in line.split("|") if t.strip()]


def _parse_description(text: str) -> Tuple[str, str]:
    obj  = re.search(r"Object:\s*(.+)",      text, re.IGNORECASE)
    desc = re.search(r"Description:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return (
        obj.group(1).strip()  if obj  else "(unknown)",
        desc.group(1).strip() if desc else text.strip(),
    )


async def _vllm_call(client: AsyncOpenAI, messages: List[Dict[str, Any]]) -> str:
    resp = await client.chat.completions.create(
        model=VLLM_MODEL,
        messages=messages,
        max_tokens=MAX_NEW_TOKENS,
        temperature=0.0,
        extra_body={"repetition_penalty": 1.05},
    )
    return resp.choices[0].message.content or ""


async def _infer_one(client, sem, idx: int, crop: Image.Image, prompt: str) -> Tuple[int, str]:
    messages = _build_messages(crop, prompt)
    async with sem:
        for attempt in range(2):
            try:
                text = await asyncio.wait_for(_vllm_call(client, messages), timeout=REQUEST_TIMEOUT)
                return idx, text
            except asyncio.TimeoutError:
                if attempt == 1:
                    return idx, "[TIMEOUT]"


async def _run_prompt_on_crops(
    crops: List[Tuple[int, Image.Image]], prompt: str
) -> Dict[int, str]:
    client = AsyncOpenAI(base_url=VLLM_BASE, api_key="none")
    sem    = asyncio.Semaphore(CONCURRENCY)
    tasks  = [_infer_one(client, sem, i, c, prompt) for i, c in crops]
    results: Dict[int, str] = {}
    for coro in asyncio.as_completed(tasks):
        idx, text = await coro
        results[idx] = text
    return results


def _build_vectorstore(job_id: str, items: List[Dict[str, Any]]):
    """Background task. items: [{idx, tag, tags, object, description}]."""
    import faiss

    _vs_set(job_id, status=VS_BUILDING, error=None)
    try:
        model = _get_embed_model()
        texts = []
        metadata: List[Dict[str, Any]] = []
        for it in items:
            tags_str = " | ".join(it.get("tags") or [])
            embedding_text = f"{it['object']}: {it['description']}\nTags: {tags_str}"
            texts.append(embedding_text)
            metadata.append({
                "idx": it["idx"],
                "tag": it["tag"],
                "tags": it.get("tags") or [],
                "object": it["object"],
                "description": it["description"],
            })

        embs = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        d = _job_dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(d / "objects.index"))
        with open(d / "objects_meta.pkl", "wb") as f:
            pickle.dump(metadata, f)

        _faiss_cache[job_id] = (index, metadata)
        _vs_set(job_id, status=VS_DONE, n=len(metadata), error=None)
    except Exception as e:
        tb = traceback.format_exc()
        try:
            (_job_dir(job_id) / "vectorstore.log").write_text(tb)
        except Exception:
            pass
        _vs_set(job_id, status=VS_ERROR, error=str(e))


def _load_vectorstore(job_id: str) -> Tuple[Any, List[Dict[str, Any]]]:
    cached = _faiss_cache.get(job_id)
    if cached:
        return cached
    import faiss
    d = _job_dir(job_id)
    idx_path  = d / "objects.index"
    meta_path = d / "objects_meta.pkl"
    if not idx_path.exists() or not meta_path.exists():
        raise HTTPException(status_code=409, detail="Vectorstore not ready")
    index = faiss.read_index(str(idx_path))
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
    _faiss_cache[job_id] = (index, metadata)
    return index, metadata


# ----------------------------
# API models
# ----------------------------

class QwenRunRequest(BaseModel):
    job_id: str = Field(..., description="Job id from the CuTR run (reused here).")
    image_base64: str = Field(..., description="Original full image, base64.")
    pred: Dict[str, Any] = Field(..., description="The pred dict returned by CuTR (contains 'detections').")


class VectorStoreStatus(BaseModel):
    status: str
    n: Optional[int] = None
    error: Optional[str] = None


class QueryRequest(BaseModel):
    job_id: str
    query: str
    top_k: int = 5


class QueryHit(BaseModel):
    idx: int
    tag: str
    object: str
    description: str
    tags: List[str]
    score: float


class QueryResponse(BaseModel):
    job_id: str
    query: str
    results: List[QueryHit]


# ----------------------------
# App
# ----------------------------

app = FastAPI(title="Qwen Recognition + Vectorstore API")


@app.on_event("startup")
def _preload_embed_model():
    # Force the SentenceTransformer onto GPU before serving requests so the
    # first /qwen/query call doesn't pay the load cost.
    print(f"[startup] Loading embedding model ({EMBED_DEVICE}): {EMBED_MODEL_ID} ...", flush=True)
    model = _get_embed_model()
    try:
        # Warm one forward pass so weights/kernels are materialized.
        model.encode(["warmup"], convert_to_numpy=True, normalize_embeddings=True)
    except Exception as e:
        print(f"[startup] embed warmup failed: {e}", flush=True)
    print("[startup] Embedding model ready.", flush=True)


@app.post("/qwen/jobs", response_class=PlainTextResponse)
async def run_qwen(req: QwenRunRequest):
    job_id = req.job_id or _new_job_id()
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)

    # Persist inputs (mirrors cutr_api behavior).
    (d / "input.png").write_bytes(base64.b64decode(req.image_base64))
    (d / "pred.json").write_text(json.dumps(req.pred, indent=2))

    img = Image.open(d / "input.png").convert("RGB")
    detections = req.pred.get("detections") or []
    if not detections:
        raise HTTPException(status_code=400, detail="pred.detections is empty")

    crops: List[Tuple[int, Image.Image]] = []
    for i, det in enumerate(detections):
        c = _crop_for_detection(img, det)
        if c is not None:
            crops.append((i, c))

    if not crops:
        raise HTTPException(status_code=400, detail="No valid crops produced from detections")

    # Run both prompts concurrently across crops.
    tags_task = asyncio.create_task(_run_prompt_on_crops(crops, PROMPT_TAGS))
    desc_task = asyncio.create_task(_run_prompt_on_crops(crops, PROMPT_DESCRIBE))
    tags_raw, desc_raw = await asyncio.gather(tags_task, desc_task)

    items: List[Dict[str, Any]] = []
    jsonl_lines: List[str] = []
    for i, _crop in crops:
        tag_list = _parse_tags(tags_raw.get(i, ""))
        first_tag = tag_list[0] if tag_list else None
        obj, desc = _parse_description(desc_raw.get(i, ""))
        items.append({
            "idx": i,
            "tag": first_tag,
            "tags": tag_list,
            "object": obj,
            "description": desc,
        })
        jsonl_lines.append(json.dumps({"idx": i, "tag": first_tag}))

    jsonl_text = "\n".join(jsonl_lines) + "\n"
    (d / "objects.jsonl").write_text(jsonl_text)
    (d / "objects_full.json").write_text(json.dumps(items, indent=2))

    # Kick off vectorstore build in background.
    _vs_set(job_id, status=VS_PENDING, error=None)
    _vs_executor.submit(_build_vectorstore, job_id, items)

    return PlainTextResponse(jsonl_text, media_type="application/x-ndjson")


@app.get("/qwen/jobs/{job_id}/vectorstore", response_model=VectorStoreStatus)
def vectorstore_status(job_id: str):
    st = _vs_get(job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    return VectorStoreStatus(
        status=str(st.get("status", VS_ERROR)),
        n=st.get("n"),
        error=st.get("error"),
    )


@app.post("/qwen/query", response_model=QueryResponse)
def query(req: QueryRequest):
    st = _vs_get(req.job_id)
    if not st:
        raise HTTPException(status_code=404, detail="Job not found")
    if st.get("status") != VS_DONE:
        raise HTTPException(status_code=409, detail=f"Vectorstore not ready (status={st.get('status')})")

    index, metadata = _load_vectorstore(req.job_id)
    model = _get_embed_model()
    emb = model.encode([req.query], convert_to_numpy=True, normalize_embeddings=True)
    k = max(1, min(int(req.top_k), len(metadata)))
    scores, indices = index.search(emb, k)

    hits: List[QueryHit] = []
    for score, i in zip(scores[0].tolist(), indices[0].tolist()):
        if i < 0 or i >= len(metadata):
            continue
        m = metadata[i]
        hits.append(QueryHit(
            idx=int(m["idx"]),
            tag=str(m.get("tag") or ""),
            object=str(m.get("object") or ""),
            description=str(m.get("description") or ""),
            tags=list(m.get("tags") or []),
            score=float(score),
        ))
    return QueryResponse(job_id=req.job_id, query=req.query, results=hits)


"""
uvicorn qwen_api:app --host 0.0.0.0 --port 8091 --workers 1
"""
