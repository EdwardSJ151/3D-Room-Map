"""Tools exposed to the ADK agent.

The main tool (`query_environment_objects`) queries the FAISS vectorstore
in-process, using the same embedding model as the notebook
(Qwen/Qwen3-Embedding-0.6B via SentenceTransformers).

Required files (from inference_timing.ipynb):
    objects.index       — FAISS index (IndexFlatIP, normalized embeddings)
    objects_meta.pkl    — list of dicts: id, object, description, image_path

Environment variables:
    VECTORSTORE_DIR      Directory with objects.index and objects_meta.pkl
    EMBEDDING_MODEL      SentenceTransformer model (default: Qwen/Qwen3-Embedding-0.6B)
    EMBEDDING_DEVICE     torch device (default: cpu)
    QWEN_USE_MOCK        "1" / "true" for a local mock dataset instead of FAISS.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_TOP_K = 10

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default: two levels up from this file (env_agent/ → scene_agent/ → 3D-Room-Map/)
_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_VS_DIR = _THIS_DIR.parent.parent  # 3D-Room-Map/

VECTORSTORE_DIR: Path = Path(
    os.environ.get("VECTORSTORE_DIR", str(_DEFAULT_VS_DIR))
).resolve()
EMBEDDING_MODEL: str = os.environ.get("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_DEVICE: str = os.environ.get("EMBEDDING_DEVICE", "cpu")
QWEN_USE_MOCK: bool = os.environ.get("QWEN_USE_MOCK", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

# ---------------------------------------------------------------------------
# Cached singletons (loaded at startup via preload_vectorstore)
# ---------------------------------------------------------------------------

_model = None
_index = None
_metadata: Optional[List[Dict[str, Any]]] = None
_preloaded = False


def _load_resources() -> tuple:
    """Load and cache the embedding model, FAISS index, and metadata."""
    global _model, _index, _metadata

    if _model is None:
        logger.info(
            "Loading embedding model %s on %s...",
            EMBEDDING_MODEL,
            EMBEDDING_DEVICE,
        )
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            EMBEDDING_MODEL,
            device=EMBEDDING_DEVICE,
        )
        logger.info("Embedding model loaded.")

    if _index is None or _metadata is None:
        import faiss

        index_path = VECTORSTORE_DIR / "objects.index"
        meta_path = VECTORSTORE_DIR / "objects_meta.pkl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"FAISS index file not found: {index_path}\n"
                "Generate it by running the indexing cells in inference_timing.ipynb."
            )
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {meta_path}\n"
                "Generate it by running the indexing cells in inference_timing.ipynb."
            )

        logger.info("Loading FAISS index from %s ...", index_path)
        _index = faiss.read_index(str(index_path))
        with open(meta_path, "rb") as f:
            _metadata = pickle.load(f)
        logger.info(
            "Vectorstore ready: %d vectors, %d metadata entries.",
            _index.ntotal,
            len(_metadata),
        )

    return _model, _index, _metadata


def preload_vectorstore() -> None:
    """Eager-load embedding model + FAISS at ADK startup (before any tool call)."""
    global _preloaded
    if _preloaded or QWEN_USE_MOCK:
        return
    logger.info(
        "Preloading vectorstore (dir=%s, model=%s, device=%s)...",
        VECTORSTORE_DIR,
        EMBEDDING_MODEL,
        EMBEDDING_DEVICE,
    )
    _load_resources()
    _preloaded = True


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------

_MOCK_OBJECTS: List[Dict[str, Any]] = [
    {
        "idx": 0,
        "object": "notebook",
        "description": "A silver laptop sitting open on top of a wooden desk, screen on.",
        "tags": ["laptop", "notebook", "computer", "silver", "desk"],
    },
    {
        "idx": 1,
        "object": "mug",
        "description": (
            "A white ceramic mug with coffee residue, placed to the right of "
            "the laptop on the desk."
        ),
        "tags": ["mug", "cup", "coffee", "white", "ceramic", "desk"],
    },
    {
        "idx": 2,
        "object": "pen",
        "description": (
            "A black ballpoint pen lying horizontally next to a small "
            "notepad on the desk surface."
        ),
        "tags": ["pen", "black", "writing", "desk", "notepad"],
    },
    {
        "idx": 3,
        "object": "monitor",
        "description": (
            "A wide curved monitor on a black stand, positioned behind the laptop."
        ),
        "tags": ["monitor", "screen", "display", "curved", "desk"],
    },
    {
        "idx": 4,
        "object": "office chair",
        "description": "A black mesh office chair with armrests, pulled up to the desk.",
        "tags": ["chair", "office", "black", "mesh", "furniture"],
    },
]


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "message": message, "results": []}
    payload.update(extra)
    return payload


def _emit_tool_response(payload: dict) -> dict:
    """Log and print the exact dict returned to the LLM as the tool result."""
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(f"\n--- query_environment_objects → model ---\n{text}\n")
    logger.info("query_environment_objects → model:\n%s", text)
    return payload


def _score_mock(query: str, item: Dict[str, Any]) -> float:
    q_tokens = {t.lower() for t in query.split() if len(t) > 2}
    if not q_tokens:
        return 0.0
    haystack = " ".join([
        str(item.get("object", "")),
        str(item.get("description", "")),
        " ".join(item.get("tags") or []),
    ]).lower()
    return sum(1 for t in q_tokens if t in haystack) / len(q_tokens)


def _mock_query(query: str, k: int) -> Dict[str, Any]:
    scored = sorted(
        [(_score_mock(query, obj), obj) for obj in _MOCK_OBJECTS],
        key=lambda x: x[0],
        reverse=True,
    )
    if all(s == 0 for s, _ in scored):
        chosen = [(0.0, obj) for obj in _MOCK_OBJECTS[:k]]
    else:
        chosen = scored[:k]
    results = [
        {
            "idx": obj["idx"],
            "object": obj["object"],
            "description": obj["description"],
            "tags": list(obj["tags"]),
            "score": float(score),
        }
        for score, obj in chosen
    ]
    return {"status": "success", "query": query, "count": len(results), "results": results, "mock": True}


# ---------------------------------------------------------------------------
# Main tool
# ---------------------------------------------------------------------------

def query_environment_objects(query: str, top_k: int = 5) -> dict:
    """
    Search the current environment for objects that semantically match `query`.

    Call whenever the user asks about the physical environment, the scene, the
    room, where an object is, what is on a surface, or asks to describe or
    identify something visible around them.

    Args:
        query (str): Natural-language description of what to find
            (e.g. "a pen on the desk", "dark wooden objects", "something to sit on").
        top_k (int): Number of results to return. Default 5, maximum 10.

    Returns:
        dict with:
            - status: "success" or "error"
            - results: list of matches with object, description, score, idx, image_path
            - message: explanation on error
    """
    if not query or not query.strip():
        return _emit_tool_response(
            _error("Empty query. Provide a description of what to search for.")
        )

    k = max(1, min(int(top_k), _MAX_TOP_K))

    if QWEN_USE_MOCK:
        return _emit_tool_response(_mock_query(query.strip(), k))

    try:
        model, index, metadata = _load_resources()
    except FileNotFoundError as exc:
        return _emit_tool_response(_error(str(exc)))
    except Exception as exc:
        return _emit_tool_response(_error(f"Failed to load vectorstore: {exc}"))

    try:
        emb = model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        # IndexFlatIP with normalized vectors: distances = cosine similarities
        distances, indices = index.search(emb, k)
    except Exception as exc:
        return _emit_tool_response(_error(f"Vectorstore search failed: {exc}"))

    results: List[Dict[str, Any]] = []
    for score, i in zip(distances[0], indices[0]):
        if i < 0 or i >= len(metadata):
            continue
        m = metadata[i]
        results.append({
            "idx": int(i),
            "object": m.get("object") or "",
            "description": m.get("description") or "",
            "image_path": m.get("image_path") or "",
            "score": float(score),
        })

    return _emit_tool_response({
        "status": "success",
        "query": query.strip(),
        "count": len(results),
        "results": results,
    })


__all__ = ["query_environment_objects", "preload_vectorstore"]
