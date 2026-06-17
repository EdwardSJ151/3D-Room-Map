# env_agent — Agente ADK com tool de vectorstore do ambiente

Agente conversacional construído com [Google ADK](https://google.github.io/adk-docs/)
que responde perguntas gerais com o LLM (Gemini 2.0 Flash) e, quando o usuário
pergunta sobre o **ambiente físico / objetos da cena**, chama uma tool que
consulta um vectorstore FAISS servido pelo serviço FastAPI `qwen_api.py`.

## Como funciona

```
Usuario  →  root_agent (gemini-2.0-flash)
                │
                ├─ pergunta geral ............. responde direto
                │
                └─ pergunta sobre ambiente ... query_environment_objects(query)
                                                    │
                                                    ▼
                                       POST /qwen/query (FastAPI)
                                                    │
                                                    ▼
                                       FAISS + Qwen3-Embedding
```

O LLM decide sozinho quando chamar a tool, com base no docstring de
`query_environment_objects` em [env_agent/tools.py](env_agent/tools.py) e na
`instruction` definida em [env_agent/agent.py](env_agent/agent.py).

## Pré-requisitos

- Python 3.10+
- Uma `GOOGLE_API_KEY` do [Google AI Studio](https://aistudio.google.com/apikey).
- O serviço **`qwen_api.py`** rodando (FastAPI + FAISS), e um `job_id` já
  processado com vectorstore em status `done`. Tipicamente:

  ```bash
  uvicorn qwen_api:app --host 0.0.0.0 --port 8091 --workers 1
  ```

  Você pode conferir o status do job com:

  ```bash
  curl http://localhost:8091/qwen/jobs/<JOB_ID>/vectorstore
  # esperado: {"status":"done", "n": <int>, "error": null}
  ```

## Instalação

```bash
cd C:\Users\NishinoTSK\Downloads\wsl\script
python -m venv .venv
.\.venv\Scripts\Activate.ps1     # Windows PowerShell
# source .venv/bin/activate       # WSL / Linux / macOS
pip install -r requirements.txt
```

## Configuração

Copie o exemplo e preencha os valores:

```bash
copy .env.example .env            # Windows
# cp .env.example .env             # WSL / Linux / macOS
```

Edite `.env` com:

| Variável               | Obrigatória | Descrição                                                                |
| ---------------------- | ----------- | ------------------------------------------------------------------------ |
| `GOOGLE_API_KEY`       | sim         | API key do Google AI Studio para o Gemini.                               |
| `GEMINI_MODEL`         | não         | Modelo Gemini (default `gemini-2.5-flash`).                              |
| `QWEN_USE_MOCK`        | não         | `1` para usar dataset mock e não chamar o `qwen_api.py`.                 |
| `ENV_JOB_ID`           | condicional | Job id já indexado no `qwen_api.py`. Obrigatório quando `QWEN_USE_MOCK=0`. |
| `QWEN_API_BASE`        | não         | URL do FastAPI (default `http://localhost:8091`).                        |
| `QWEN_QUERY_TIMEOUT_S` | não         | Timeout em segundos do HTTP call (default `15`).                         |

O ADK CLI carrega o `.env` automaticamente ao iniciar.

## Execução

Sempre rode a partir da pasta `script/` (pasta que **contém** `env_agent/`):

```bash
adk run env_agent     # REPL no terminal
# ou
adk web               # UI no browser (http://localhost:8000)
```

## Exemplos de prompt

**Pergunta geral (não usa a tool):**

> Explique o que é recursão em programação com um exemplo simples.

**Pergunta sobre o ambiente (deve disparar `query_environment_objects`):**

> O que tem em cima da minha mesa?
>
> Tem alguma coisa de madeira aqui no quarto?
>
> Descreva o objeto mais à direita da cena.

Se o `qwen_api.py` não estiver rodando, ou o `ENV_JOB_ID` apontar para um job
inexistente / não pronto, a tool devolve `status: error` com uma mensagem
explicativa, e o agente repassa isso ao usuário sem quebrar.

## Testar sem o `qwen_api.py` (modo mock)

Se você não consegue subir o `qwen_api.py` localmente (ex.: depende de GPU
para o `Qwen3-Embedding`), defina no `.env`:

```
QWEN_USE_MOCK=1
```

Nesse modo, `query_environment_objects` devolve resultados de uma cena
fictícia hardcoded em [env_agent/tools.py](env_agent/tools.py) (notebook,
caneca, caneta, monitor, cadeira, livro, luminária, planta) com um
ranking simples por sobreposição de tokens. Isso é o suficiente para
validar o fluxo do agente: quando ele decide chamar a tool, com que
`query`, e como ele formata a resposta final usando os hits.

`ENV_JOB_ID` deixa de ser obrigatório quando `QWEN_USE_MOCK=1`.

## Troubleshooting

### `429 RESOURCE_EXHAUSTED ... limit: 0` no `gemini-2.0-flash`

Significa que o free tier do `gemini-2.0-flash` está zerado no seu
projeto da Google AI Studio. Opções:

1. **Trocar o modelo** (mais rápido): edite `.env` para
   `GEMINI_MODEL=gemini-2.5-flash` (ou `gemini-2.5-flash-lite`) e tente
   de novo. Modelos diferentes têm cotas independentes.
2. **Habilitar billing**: em [aistudio.google.com](https://aistudio.google.com)
   → API key → projeto vinculado → ativar pagamento. As cotas pagas são
   muito maiores.
3. **Usar Vertex AI**: se você tem GCP, autentique via
   `gcloud auth application-default login` e configure
   `GOOGLE_GENAI_USE_VERTEXAI=TRUE` + `GOOGLE_CLOUD_PROJECT` +
   `GOOGLE_CLOUD_LOCATION` no `.env` — o ADK passa a usar a cota do
   projeto GCP em vez do free tier.

Documentação oficial: [ADK error 429](https://google.github.io/adk-docs/agents/models/google-gemini/#error-code-429-resource_exhausted).

### O agente não chama a tool quando eu pergunto sobre o ambiente

- Verifique que a pergunta menciona claramente "o ambiente", "a sala",
  "o que tem aqui", "objeto", etc. — para perguntas genéricas como
  "bom dia" o agente responde direto, sem tool, e isso é esperado.
- Confira os logs do `adk run`: deve aparecer uma `function_call` para
  `query_environment_objects` seguida de um `function_response`.

## Estrutura

```
script/
├── env_agent/
│   ├── __init__.py
│   ├── agent.py          # root_agent
│   └── tools.py          # query_environment_objects
├── .env.example
├── requirements.txt
└── README.md
```
