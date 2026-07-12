from datetime import datetime
import json
from typing import Any

from groq import Groq

from app.synthesis.base import LLMProvider

SYSTEM_PROMPT = (
    "You are a personal memory assistant. Answer the user's question using ONLY "
    "the provided memories below. Cite the memory date when relevant. "
    "If the memories don't contain enough information to answer, say so clearly "
    "instead of guessing."
)


class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self.api_key = api_key
        self.model = model
        self.client: Groq | None = None

    def synthesize(self, query: str, context_chunks: list[dict[str, Any]]) -> str:
        if not context_chunks:
            return "I don't have enough stored memory to answer that."
        if not self.api_key:
            joined = "\n".join(f"- {chunk['text']}" for chunk in context_chunks)
            return f"GROQ_API_KEY is not configured. Retrieved relevant memories:\n{joined}"

        context_text = self._format_context(context_chunks)
        user_message = f"Memories:\n\n{context_text}\n\nQuestion: {query}"
        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1000,
        )
        return response.choices[0].message.content or ""

    def complete(self, system_prompt: str, user_message: str, temperature: float = 0.2) -> str:
        if not self.api_key:
            return ""

        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=800,
        )
        return response.choices[0].message.content or ""

    def complete_with_tools(
        self,
        system_prompt: str,
        user_message: str,
        tools: list[dict[str, Any]],
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        if not self.api_key:
            return {"type": "text", "content": ""}

        response = self._client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=800,
        )
        message = response.choices[0].message

        if message.tool_calls:
            call = message.tool_calls[0]
            return {
                "type": "tool_call",
                "name": call.function.name,
                "arguments": json.loads(call.function.arguments or "{}"),
            }
        return {"type": "text", "content": message.content or ""}

    def _client(self) -> Groq:
        if self.client is None:
            self.client = Groq(api_key=self.api_key)
        return self.client

    def _format_context(self, context_chunks: list[dict[str, Any]]) -> str:
        context_blocks = []
        for chunk in context_chunks:
            metadata = chunk.get("metadata") or {}
            timestamp = metadata.get("timestamp")
            date_str = datetime.fromtimestamp(float(timestamp)).strftime("%Y-%m-%d") if timestamp else "unknown date"
            source = metadata.get("source", "unknown")
            context_blocks.append(f"[{date_str} | source: {source}]\n{chunk.get('text', '')}")
        return "\n\n---\n\n".join(context_blocks)

