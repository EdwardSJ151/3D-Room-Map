"""Definicao do root_agent que o ADK descobre via `adk run` / `adk web`.

Estrutura esperada pelo ADK:
    env_agent/
        __init__.py   # re-exporta `agent`
        agent.py      # define `root_agent`

Para executar:
    cd C:\\Users\\NishinoTSK\\Downloads\\wsl\\script
    adk run env_agent          # CLI interativo
    adk web                    # UI no browser
"""

import os

from google.adk.agents import Agent

from .tools import query_environment_objects

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

INSTRUCTION = (
    "Voce e um assistente conversacional util e direto, falando portugues "
    "brasileiro por padrao (a menos que o usuario use outro idioma).\n"
    "\n"
    "Para perguntas gerais (conhecimento amplo, conversa, raciocinio, "
    "explicacoes), responda diretamente usando seu proprio conhecimento, "
    "sem chamar nenhuma tool.\n"
    "\n"
    "Use a tool `query_environment_objects` SEMPRE que o usuario perguntar:\n"
    "  - sobre o ambiente fisico, a sala, a cena ou o local onde ele esta;\n"
    "  - o que existe / o que tem ao redor, na mesa, no comodo;\n"
    "  - onde esta um objeto especifico, ou pedir para descrever algo visivel;\n"
    "  - identificacao de objetos a partir de pistas (cor, formato, material).\n"
    "\n"
    "Ao chamar a tool, formule uma `query` curta em linguagem natural com o "
    "que o usuario quer encontrar. Baseie sua resposta final apenas nos "
    "objetos retornados, mencionando descricoes e tags relevantes. Se a tool "
    "retornar `status: error`, explique de forma amigavel ao usuario que o "
    "ambiente nao esta indexado/disponivel no momento e diga o motivo."
)

root_agent = Agent(
    name="env_agent",
    model=GEMINI_MODEL,
    description=(
        "Assistente conversacional que tambem responde perguntas sobre o "
        "ambiente fisico do usuario, consultando um vectorstore de objetos "
        "detectados na cena."
    ),
    instruction=INSTRUCTION,
    tools=[query_environment_objects],
)

__all__ = ["root_agent"]
