"""Root agent definition discovered by ADK via `adk run` / `adk web`.

Expected layout:
    env_agent/
        __init__.py   # re-exports `agent`
        agent.py      # defines `root_agent`

Run from the directory that contains `env_agent/`:
    adk run env_agent
    adk web
"""

import os

from google.adk.agents import Agent

from .tools import preload_vectorstore, query_environment_objects

# Load embedding model + FAISS before ADK accepts requests (not on first tool call).
preload_vectorstore()

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()

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
    "Language: English by default, unless the user uses another language.\n"
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

root_agent = Agent(
    name="env_agent",
    model=GEMINI_MODEL,
    description=(
        "Voice assistant (TTS) that answers general questions and questions "
        "about the physical environment via a scene object vectorstore."
    ),
    instruction=INSTRUCTION,
    tools=[query_environment_objects],
)

__all__ = ["root_agent"]
