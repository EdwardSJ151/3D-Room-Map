from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import requests

from eval.config import ADK_BASE, ADK_APP


def create_session(session_id: str, job_id: str) -> dict:
    url = f"{ADK_BASE}/apps/{ADK_APP}/users/{session_id}/sessions"
    resp = requests.post(
        url,
        json={"session_id": session_id, "state": {"job_id": job_id}},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def run_turn(session_id: str, user_message: str) -> List[Dict[str, Any]]:
    url = f"{ADK_BASE}/run"
    payload = {
        "app_name": ADK_APP,
        "user_id": session_id,
        "session_id": session_id,
        "new_message": {"role": "user", "parts": [{"text": user_message}]},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def parse_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract structured data from ADK event array."""
    assistant_answer = ""
    retrieved_records: List[Dict[str, Any]] = []
    best_idx: Optional[int] = None
    retrieval_time_ms: Optional[float] = None
    tool_was_called = False

    for event in events:
        content = event.get("content") or {}
        parts = content.get("parts") or []
        role = content.get("role", "")

        for part in parts:
            # Final text reply from the model
            if role == "model" and "text" in part and part["text"].strip():
                candidate = part["text"].strip()
                if not any(k in part for k in ("functionCall", "functionResponse")):
                    assistant_answer = candidate

            # Tool call detected
            if "functionCall" in part:
                tool_was_called = True

            # Tool response — grab results and state delta
            if "functionResponse" in part:
                fn_resp = part["functionResponse"]
                response_data = fn_resp.get("response") or {}
                if isinstance(response_data, dict):
                    results = response_data.get("results") or []
                    if results:
                        retrieved_records = results

        # state delta from event
        state_delta = event.get("actions", {}).get("stateDelta") or {}
        if "best_idx" in state_delta:
            best_idx = state_delta["best_idx"]
        if "retrieval_time_ms" in state_delta:
            retrieval_time_ms = state_delta["retrieval_time_ms"]

    return {
        "assistant_answer": assistant_answer,
        "retrieved_records": retrieved_records,
        "best_idx": best_idx,
        "retrieval_time_ms": retrieval_time_ms,
        "tool_was_called": tool_was_called,
    }
