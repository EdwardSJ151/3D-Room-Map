from __future__ import annotations

import time

import requests


def poll_until_done(
    url: str,
    label: str,
    done_value: str = "done",
    status_key: str = "status",
    error_value: str = "error",
    interval_s: float = 2.0,
    timeout_s: float = 600.0,
) -> dict:
    deadline = time.monotonic() + timeout_s
    while True:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        body = resp.json()
        status = body.get(status_key, "")
        print(f"[{label}] status={status}", flush=True)
        if status == done_value:
            return body
        if status == error_value:
            raise RuntimeError(f"{label} failed: {body.get('error') or body}")
        if time.monotonic() > deadline:
            raise TimeoutError(f"{label} timed out after {timeout_s}s")
        time.sleep(interval_s)
