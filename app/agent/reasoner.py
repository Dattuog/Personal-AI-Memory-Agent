import json
from typing import Any

from app.agent.schema import AgentDecision
from app.synthesis import get_llm_provider

DECISION_SYSTEM_PROMPT = """You are the reasoning layer of a personal memory agent. You are given a new piece of text the user just copied, along with a list of existing memories that are semantically related to it, if any.

Decide what action to take. Respond with ONLY a JSON object matching this exact schema, no other text, no markdown fences:

{
  "action": "ingest_silent" | "ingest_and_surface" | "ingest_and_flag_conflict" | "skip",
  "reasoning": "<one or two sentence explanation>",
  "related_memory_ids": ["<id>", ...],
  "conflict_summary": "<string or null>",
  "suggested_reminder": "<string or null>"
}

Guidelines:
- Use "skip" if the new content is noise, gibberish, or clearly not worth remembering.
- Use "ingest_and_flag_conflict" if the new content contradicts or supersedes an existing memory.
- Use "ingest_and_surface" if the new content is meaningfully related to one or more existing memories and the user would likely want to be reminded of that connection.
- Use "ingest_silent" for anything else worth storing but not otherwise noteworthy.
- If the new content mentions a deadline, due date, or something time-sensitive worth reminding the user about later, set "suggested_reminder" to a short reminder string. Otherwise set it to null.
- Only include memory IDs that are genuinely relevant in related_memory_ids.
- Never output the "self_review_flag" action here. That value is reserved for a different part of the system.
"""


def _build_user_message(new_content: str, candidates: list[dict[str, Any]]) -> str:
    if not candidates:
        candidate_block = "No related memories were found."
    else:
        lines = []
        for candidate in candidates:
            metadata = candidate.get("metadata") or {}
            lines.append(
                f"id: {candidate.get('id')}\n"
                f"text: {candidate.get('text', '')}\n"
                f"timestamp: {metadata.get('timestamp')}"
            )
        candidate_block = "\n\n".join(lines)
    return f"New clipboard content:\n{new_content}\n\nRelated existing memories:\n{candidate_block}"


def decide(new_content: str, candidates: list[dict[str, Any]]) -> AgentDecision:
    llm = get_llm_provider()
    user_message = _build_user_message(new_content, candidates)
    raw = llm.complete(system_prompt=DECISION_SYSTEM_PROMPT, user_message=user_message, temperature=0.1)
    try:
        parsed = json.loads(raw)
        return AgentDecision(**parsed)
    except Exception:
        return AgentDecision(
            action="ingest_silent",
            reasoning="Fallback: could not parse LLM decision.",
            related_memory_ids=[],
        )
