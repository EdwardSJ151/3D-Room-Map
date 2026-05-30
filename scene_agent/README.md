# env_agent

Agente ADK (Gemini 2.5 Flash) que responde perguntas gerais diretamente e, quando perguntado sobre o **ambiente físico**, chama `query_environment_objects` — que busca no vectorstore FAISS em-processo usando `Qwen3-Embedding-0.6B`.

Requer `objects.index` e `objects_meta.pkl` gerados pelo `inference_timing.ipynb`.

## Executar

```bash
cd scene_agent
pip install -r requirements.txt

# edit .env: set GOOGLE_API_KEY (and VECTORSTORE_DIR if files are not in 3D-Room-Map/)

adk run env_agent     # terminal REPL
adk web               # browser UI — http://localhost:8000
```

On startup, the agent **preloads** the Qwen embedding model and FAISS index (first launch downloads ~1.2GB from Hugging Face). Wait until you see `Vectorstore ready` in the logs before chatting.

## Variáveis de ambiente (`.env`)

| Variável           | Default                      | Descrição                                      |
| ------------------ | ---------------------------- | ---------------------------------------------- |
| `GOOGLE_API_KEY`   | —                            | **Obrigatório.** API key do Google AI Studio.  |
| `GEMINI_MODEL`     | `gemini-2.5-flash`           | Modelo Gemini.                                 |
| `VECTORSTORE_DIR`  | raiz do projeto (`3D-Room-Map/`) | Pasta com `objects.index` e `objects_meta.pkl`. |
| `EMBEDDING_DEVICE` | `cpu`                        | `cpu` ou `cuda`.                               |
| `QWEN_USE_MOCK`    | `0`                          | `1` para responder com objetos fictícios sem carregar o FAISS. |
