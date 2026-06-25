"""Tools expostas ao agente ADK.

A tool principal (`query_environment_objects`) consulta o vectorstore FAISS
que e construido e servido por `qwen_api.py` (endpoint `POST /qwen/query`).

Configuracao via variaveis de ambiente:
    QWEN_API_BASE        URL base do FastAPI (default: http://localhost:8091)
    ENV_JOB_ID           Job id ja indexado no vectorstore (obrigatorio,
                         exceto quando QWEN_USE_MOCK=1)
    QWEN_QUERY_TIMEOUT_S Timeout HTTP em segundos (default: 15)
    QWEN_USE_MOCK        "1" / "true" para usar dataset mock local em vez
                         de chamar o FastAPI. Util para testar o agente
                         sem precisar rodar o qwen_api.py.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
from google.adk.tools import ToolContext

QWEN_API_BASE: str = os.environ.get("QWEN_API_BASE", "http://localhost:8091").rstrip("/")
ENV_JOB_ID: str = os.environ.get("ENV_JOB_ID", "").strip()
QWEN_QUERY_TIMEOUT_S: float = float(os.environ.get("QWEN_QUERY_TIMEOUT_S", "15"))
QWEN_USE_MOCK: bool = os.environ.get("QWEN_USE_MOCK", "0").strip().lower() in (
    "1", "true", "yes", "on",
)

_MAX_TOP_K = 20


_MOCK_OBJECTS: List[Dict[str, Any]] = [
    {
        "idx": 0,
        "object": "notebook",
        "description": (
            "A silver laptop sitting open on top of a wooden desk, screen on."
        ),
        "tags": ["laptop", "notebook", "computer", "silver", "desk"],
    },
    {
        "idx": 1,
        "object": "caneca",
        "description": (
            "A white ceramic mug with coffee residue, placed to the right of "
            "the laptop on the desk."
        ),
        "tags": ["mug", "cup", "coffee", "white", "ceramic", "desk"],
    },
    {
        "idx": 2,
        "object": "caneta",
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
            "A wide curved monitor on a black stand, positioned behind the "
            "laptop."
        ),
        "tags": ["monitor", "screen", "display", "curved", "desk"],
    },
    {
        "idx": 4,
        "object": "cadeira de escritorio",
        "description": (
            "A black mesh office chair with armrests, pulled up to the desk."
        ),
        "tags": ["chair", "office", "black", "mesh", "furniture"],
    },
    {
        "idx": 5,
        "object": "livro",
        "description": (
            "A thick hardcover book with a dark blue dust jacket, standing "
            "on a small shelf to the left of the desk."
        ),
        "tags": ["book", "blue", "hardcover", "shelf"],
    },
    {
        "idx": 6,
        "object": "luminaria",
        "description": (
            "A black articulated desk lamp clamped to the back edge of the "
            "desk."
        ),
        "tags": ["lamp", "light", "black", "desk"],
    },
    {
        "idx": 7,
        "object": "planta",
        "description": (
            "A small green potted plant in a terracotta pot, sitting on the "
            "windowsill behind the desk."
        ),
        "tags": ["plant", "green", "pot", "window"],
    },
]


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "message": message, "results": []}
    payload.update(extra)
    return payload


def _score_mock(query: str, item: Dict[str, Any]) -> float:
    """Pontuacao bem simples por sobreposicao de tokens para o modo mock."""
    q_tokens = {t.lower() for t in query.split() if len(t) > 2}
    if not q_tokens:
        return 0.0
    haystack_parts = [
        str(item.get("object", "")),
        str(item.get("description", "")),
        " ".join(item.get("tags") or []),
    ]
    haystack = " ".join(haystack_parts).lower()
    hits = sum(1 for t in q_tokens if t in haystack)
    return hits / max(len(q_tokens), 1)


def _mock_query(query: str, k: int) -> Dict[str, Any]:
    scored = [
        (_score_mock(query, obj), obj) for obj in _MOCK_OBJECTS
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    if all(score == 0 for score, _ in scored):
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
    return {
        "status": "success",
        "query": query,
        "job_id": "MOCK",
        "count": len(results),
        "results": results,
        "mock": True,
    }


def query_environment_objects(query: str, top_k: int = 5, tool_context: Optional[ToolContext] = None) -> dict:
    """
    Busca objetos no ambiente atual que correspondam semanticamente a `query`.

    Use SEMPRE que o usuario perguntar sobre o ambiente fisico, a cena, a sala,
    onde esta um objeto, o que esta sobre a mesa, ou pedir descricao /
    identificacao de algo visivel ao redor.

    Args:
        query (str): Descricao em linguagem natural do que procurar
            (ex.: "uma caneta sobre a mesa", "objetos de madeira escura",
            "algo para sentar").
        top_k (int): Quantos resultados retornar. Default 5, maximo 20.

    Returns:
        dict: Estrutura com:
            - status: "success" ou "error"
            - results: lista de objetos encontrados, cada um com
              `object`, `description`, `tags`, `score`, `idx`
            - message: explicacao em caso de erro
    """
    if not query or not query.strip():
        return _error("Query vazia. Forneca uma descricao do que procurar.")

    k = max(1, min(int(top_k), _MAX_TOP_K))

    if QWEN_USE_MOCK:
        return _mock_query(query.strip(), k)

    job_id = (
        (tool_context.state.get("job_id") if tool_context else None)
        or ENV_JOB_ID
    )
    if not job_id:
        return _error(
            "Nenhum job_id encontrado na sessao ou na variavel ENV_JOB_ID. "
            "O ambiente do usuario nao foi indexado. "
            "Defina QWEN_USE_MOCK=1 para testar com dados ficticios."
        )

    url = f"{QWEN_API_BASE}/qwen/query"
    payload = {"job_id": job_id, "query": query.strip(), "top_k": k}

    try:
        response = httpx.post(url, json=payload, timeout=QWEN_QUERY_TIMEOUT_S)
    except httpx.TimeoutException:
        return _error(f"Timeout ao consultar o vectorstore em {url}.")
    except httpx.RequestError as exc:
        return _error(f"Falha de rede ao consultar {url}: {exc}.")

    if response.status_code == 404:
        return _error(
            f"Job '{job_id}' nao encontrado no servidor de vectorstore."
        )
    if response.status_code == 409:
        return _error(
            f"Vectorstore para o job '{job_id}' ainda nao esta pronto."
        )
    if response.status_code >= 400:
        return _error(
            f"Erro HTTP {response.status_code} ao consultar o vectorstore: "
            f"{response.text[:200]}"
        )

    try:
        body = response.json()
    except ValueError:
        return _error("Resposta do vectorstore nao e JSON valido.")

    raw_hits = body.get("results") or []
    results: List[Dict[str, Any]] = []
    for hit in raw_hits:
        results.append({
            "idx": hit.get("idx"),
            "object": hit.get("object") or "",
            "description": hit.get("description") or "",
            "tags": list(hit.get("tags") or []),
            "score": float(hit.get("score", 0.0)),
        })

    return {
        "status": "success",
        "query": query.strip(),
        "job_id": job_id,
        "count": len(results),
        "results": results,
    }


__all__ = ["query_environment_objects"]
