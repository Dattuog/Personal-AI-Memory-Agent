import json
from typing import List, Dict, Any

from app.retrieval.retrieve import retrieve_and_rerank
from app.synthesis import get_llm_provider

CLASSIFY_SYSTEM_PROMPT = """Classify the user's request as either "simple" or "complex".

"simple" = a single factual lookup answerable from one retrieval pass.
"complex" = a multi-part request that benefits from being broken into sub-questions.

Respond with ONLY one word: simple or complex."""


def classify_query(query: str) -> str:
    llm = get_llm_provider()
    raw = llm.complete(system_prompt=CLASSIFY_SYSTEM_PROMPT, user_message=query, temperature=0.0)
    return "complex" if "complex" in raw.strip().lower() else "simple"


PLAN_SYSTEM_PROMPT = """You are the planning layer of a personal memory agent. Break the user's request into 2-5 concrete sub-questions that, together, cover what is needed to fully address it. Each sub-question should be answerable by searching the user's personal memory store.

Respond with ONLY a JSON array of strings, no other text:
["sub-question 1", "sub-question 2", ...]"""


def build_plan(query: str) -> List[str]:
    llm = get_llm_provider()
    raw = llm.complete(system_prompt=PLAN_SYSTEM_PROMPT, user_message=query, temperature=0.2)
    try:
        steps = json.loads(raw)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps[:5]
    except Exception:
        pass
    return [query]


def execute_plan(steps: List[str], top_k_per_step: int = 3) -> List[Dict[str, Any]]:
    step_results = []
    for step in steps:
        ranked = retrieve_and_rerank(step, top_k_per_step)
        step_results.append({
            "sub_question": step,
            "found_memories": ranked,
            "has_data": len(ranked) > 0,
        })
    return step_results


FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are a personal memory assistant. You were given a \
multi-part task, broken into sub-questions, each with whatever memories were found (or none). \
For each sub-question, judge whether the retrieved memories ACTUALLY answer it — a loosely \
related memory that doesn't address the specific sub-question does NOT count as answered.

Respond with ONLY a JSON object (no other text) matching this shape:
{
  "answer": "<full answer text, organized by sub-topic, with a 'Still missing' section>",
  "answered_sub_questions": ["<sub-question text>", ...],
  "missing_sub_questions": ["<sub-question text>", ...]
}

Do not invent information not present in the memories. Every sub-question must appear in \
exactly one of the two lists."""


def _format_step_results(step_results: List[Dict[str, Any]]) -> str:
    blocks = []
    for step in step_results:
        if step["has_data"]:
            mem_lines = "\n".join(f"- {m['text']}" for m in step["found_memories"])
        else:
            mem_lines = "(no relevant memories found)"
        blocks.append(f"Sub-question: {step['sub_question']}\n{mem_lines}")
    return "\n\n".join(blocks)


def synthesize_plan_answer(original_query: str, step_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    llm = get_llm_provider()
    context = _format_step_results(step_results)
    user_message = f"Original request: {original_query}\n\n{context}"
    raw = llm.complete(system_prompt=FINAL_SYNTHESIS_SYSTEM_PROMPT, user_message=user_message, temperature=0.2)
    try:
        parsed = json.loads(raw)
        return parsed
    except Exception:
        # fallback: treat everything as answered if parsing fails, rather than crashing
        return {
            "answer": raw,
            "answered_sub_questions": [s["sub_question"] for s in step_results],
            "missing_sub_questions": [],
        }


def run_multi_step_task(query: str) -> Dict[str, Any]:
    steps = build_plan(query)
    step_results = execute_plan(steps)
    synthesis = synthesize_plan_answer(query, step_results)
    all_sources = [m for s in step_results for m in s["found_memories"]]
    return {
        "answer": synthesis["answer"],
        "plan": steps,
        "missing_info": synthesis.get("missing_sub_questions", []),
        "sources": all_sources,
    }