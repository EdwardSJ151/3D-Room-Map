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
from typing import Any, Optional

from google import genai
from google.adk.agents import Agent
from google.adk.tools.base_tool import BaseTool
from google.adk.tools import ToolContext

from .tools import query_environment_objects

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
_SELECTOR_MODEL = os.environ.get("SELECTOR_MODEL", "gemini-2.5-flash").strip()

_genai_client = genai.Client()

_SELECTOR_PROMPT = (
    "You are a selector. Given a user query and a list of candidate objects, "
    "return ONLY the integer idx of the single best matching object. "
    "No explanation, no JSON, just the integer.\n\n"
    "Pick the candidate that is most representative of what the user is asking "
    "about. Prefer the object or scene that best covers the intent of the query "
    "— if the query is broad, prefer a grouped scene with rich tags over a "
    "single narrow item; if the query is specific, prefer the precise match.\n\n"
    "User query: {query}\n\n"
    "Candidates:\n{candidates}"
)

INSTRUCTION = (
    "You are a lowkey friend talking out loud, not an assistant, not formal, "
    "not corporate. Your replies will be read aloud (TTS). This is not a text chat.\n"
    "\n"
    "Tone: casual, warm, a little dry humor is fine. Short sentences. No "
    "lecturing, no 'Certainly!' or 'I'd be happy to help.' Talk like you're "
    "in the room with them.\n"
    "\n"
    "FINAL RESPONSE FORMAT (required):\n"
    "After you finish reasoning and any tool calls, your last message to the "
    "user must be ONLY a single block of plain spoken text, with nothing else. "
    "Forbidden: markdown, headings, bullet or numbered lists, tables, code "
    "blocks, emojis, URLs, JSON, prefixes like 'Answer:' or 'Here is:', and "
    "any meta-commentary about what you did. Write as if speaking directly "
    "to the person.\n"
    "\n"
    "Language: English by default. If the user speaks Portuguese, reply in "
    "Portuguese — same casual tone, same format rules.\n"
    "\n"
    "For general questions (knowledge, explanations), answer from your own "
    "knowledge without calling tools.\n"
    "\n"
    "Call `query_environment_objects` when the user asks about the physical "
    "environment, the room, the scene, what is around them, where an object "
    "is, or asks to describe something visible. Formulate a short natural-"
    "language query. Base the final answer only on returned objects. If the "
    "tool returns status error, say in natural speech that the environment "
    "is unavailable and why, still as a single plain text block."
)


async def _capture_best_idx(
    tool: BaseTool,
    args: dict[str, Any],
    tool_context: ToolContext,
    tool_response: Any,
) -> Optional[dict]:
    if tool.name != "query_environment_objects":
        return None

    results = (
        tool_response.get("results", [])
        if isinstance(tool_response, dict)
        else []
    )
    if not results:
        return None

    if len(results) == 1:
        tool_context.state["best_idx"] = int(results[0]["idx"])
        return None

    # Get the user's original query — prefer the full message, fall back to tool arg.
    user_query = args.get("query", "")
    user_content = tool_context.user_content
    if user_content and user_content.parts:
        full_text = " ".join(
            p.text for p in user_content.parts if hasattr(p, "text") and p.text
        ).strip()
        if full_text:
            user_query = full_text

    candidates_text = "\n".join(
        f"idx={r['idx']}: {r['object']} — {r['description']} (tags: {', '.join(r['tags'])})"
        for r in results
    )
    prompt = _SELECTOR_PROMPT.format(query=user_query, candidates=candidates_text)

    try:
        response = await _genai_client.aio.models.generate_content(
            model=_SELECTOR_MODEL,
            contents=prompt,
        )
        tool_context.state["best_idx"] = int(response.text.strip())
    except Exception:
        tool_context.state["best_idx"] = int(results[0]["idx"])

    return None


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
    after_tool_callback=_capture_best_idx,
)

__all__ = ["root_agent"]
