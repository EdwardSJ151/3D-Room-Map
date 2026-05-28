# Paper AR Space — APIs

Two FastAPI services, each in its own conda env and terminal.

## 1. CuTR API (`cutr_api.py`)

```bash
bash setup_cubify.sh
conda activate Cubify
export CUBIFY_REPO=/content/ml-cubifyanything   # path to the cloned repo
export CUTR_MODEL_PATH=/content/cutr_rgb.pth    # optional; default is /content/cutr_rgb.pth
uvicorn cutr_api:app --host 0.0.0.0 --port 8090 --workers 1
```

Endpoints:
- `POST /cutr/jobs` — `{image_base64, meta_json, score_thresh?, max_edge?, device?, model_path?}` → `{job_id}`
- `GET  /cutr/jobs/{job_id}` — `{status, error?}`
- `GET  /cutr/jobs/{job_id}/download` — zip with `pred.json`, `input.png`, `meta.json`, `run.log`

## 2. Qwen + Vectorstore API (`qwen_api.py`)

Needs the vLLM Qwen 9B server **and** the FastAPI app, in two terminals (same env).

```bash
bash setup_qwen9b.sh
conda activate Qwen9B
```

Terminal A — vLLM (Qwen 9B, port 8010):
```bash
# any variant from cmd.txt, e.g.:
vllm serve QuantTrio/Qwen3.5-9B-AWQ \
    --served-model-name MY_MODEL \
    --max-num-seqs 32 --max-model-len 4096 \
    --gpu-memory-utilization 0.70 --tensor-parallel-size 1 \
    --enable-prefix-caching \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}' \
    --trust-remote-code --host 0.0.0.0 --port 8010
```

Terminal B — FastAPI (loads the embedding model on startup):
```bash
uvicorn qwen_api:app --host 0.0.0.0 --port 8091 --workers 1
```

Endpoints:
- `POST /qwen/jobs` — `{job_id, image_base64, pred}` → returns JSONL of `{idx, tag}`; builds the vectorstore in the background.
- `GET  /qwen/jobs/{job_id}/vectorstore` — `{status, n?, error?}`
- `POST /qwen/query` — `{job_id, query, top_k?}` → top-k matches.

## Typical flow

1. `POST /cutr/jobs` → get `job_id`, poll `GET /cutr/jobs/{job_id}` until `done`, download `pred.json` from `/download`.
2. `POST /qwen/jobs` with that `job_id`, the original image, and the `pred` dict → get JSONL back.
3. Poll `GET /qwen/jobs/{job_id}/vectorstore` until `done`.
4. `POST /qwen/query` with `{job_id, query}`.

## Env vars (optional)
- `CUBIFY_REPO` — path to `ml-cubifyanything` (default `/content/ml-cubifyanything`)
- `CUTR_JOBS_DIR` / `QWEN_JOBS_DIR` — output folders (default `cutr_jobs/`, `qwen_jobs/`)
- `VLLM_BASE`, `VLLM_MODEL` — vLLM endpoint (defaults `http://localhost:8010/v1`, `MY_MODEL`)
- `QWEN_EMBED_MODEL` — embedding model id (default `Qwen/Qwen3-Embedding-0.6B`)
- `QWEN_CONCURRENCY`, `QWEN_MAX_NEW_TOKENS`, `QWEN_TIMEOUT_S`
