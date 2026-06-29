"""Tools expostas ao agente ADK.

A tool principal (`query_environment_objects`) consulta o vectorstore FAISS
que e construido e servido por `qwen_api.py` (endpoint `POST /qwen/query`).

Configuracao via variaveis de ambiente:
    QWEN_API_BASE        URL base do FastAPI (default: http://localhost:8091)
    ENV_JOB_ID           Job id ja indexado no vectorstore (fallback quando
                         nao ha job_id na sessao ADK)
    QWEN_QUERY_TIMEOUT_S Timeout HTTP em segundos (default: 15)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from google.adk.tools import ToolContext

QWEN_API_BASE: str = os.environ.get("QWEN_API_BASE", "http://localhost:8091").rstrip("/")
ENV_JOB_ID: str = os.environ.get("ENV_JOB_ID", "").strip()
QWEN_QUERY_TIMEOUT_S: float = float(os.environ.get("QWEN_QUERY_TIMEOUT_S", "15"))
SEND_CROP_IMAGES: bool = os.environ.get("AGENT_SEND_CROP_IMAGES", "0") == "1"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

_MAX_TOP_K = 20


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "message": message, "results": []}
    payload.update(extra)
    return payload


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

    job_id = (
        (tool_context.state.get("job_id") if tool_context else None)
        or (tool_context.session.id if tool_context else None)
        or ENV_JOB_ID
    )
    if not job_id:
        return _error(
            "Nenhum job_id encontrado na sessao ou na variavel ENV_JOB_ID. "
            "O ambiente do usuario nao foi indexado."
        )

    url = f"{QWEN_API_BASE}/qwen/query"
    payload = {"job_id": job_id, "query": query.strip(), "top_k": k}

    try:
        _t0 = time.perf_counter()
        response = httpx.post(url, json=payload, timeout=QWEN_QUERY_TIMEOUT_S)
        retrieval_time_ms = (time.perf_counter() - _t0) * 1000.0
        if tool_context:
            tool_context.state["retrieval_time_ms"] = round(retrieval_time_ms, 2)
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

    # Verify room image exists early — the after_tool_callback will inject it properly.
    image_path = _REPO_ROOT / "cutr_jobs" / job_id / "input.png"
    if not image_path.exists():
        print(f"[tools] ERROR: room image not found at {image_path}", file=sys.stderr)
        return _error(f"Room image not found for job '{job_id}'. Ensure CuTR job completed successfully.")

    # Signal to the after_tool_callback which job's image to inject.
    if tool_context:
        tool_context.state["_pending_room_image_job_id"] = job_id

    # Optionally signal crop injection.
    if SEND_CROP_IMAGES and tool_context:
        tool_context.state["_pending_crop_idxs"] = [r["idx"] for r in results]

    return {
        "status": "success",
        "query": query.strip(),
        "job_id": job_id,
        "count": len(results),
        "results": results,
    }


__all__ = ["query_environment_objects"]
