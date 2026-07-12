# PAMA — Upgrade: Multi-Step Tasks

## Goal

Right now, every query does one retrieval pass and one synthesis call — fine for "when is my dentist appointment," but weak for something like "help me plan my September trip," which really needs the agent to: break the request into sub-questions, retrieve for each one separately, notice what's missing, and either flag the gaps or ask a clarifying question — instead of just answering from whatever the single retrieval pass happened to return.

This upgrade adds a **planner** that decomposes complex requests into steps, runs retrieval per step, and produces a final answer that explicitly separates "here's what I found" from "here's what's missing." It builds on the existing reasoning (`complete()`) and action-taking (`complete_with_tools()`, action registry) infrastructure.

---

## 1. Decide when to plan vs. answer directly

Not every query needs multi-step planning — "when is my dentist appointment" is a single retrieval. Use a cheap classification step before deciding which path to take.

**`app/agent/planner.py`:**
```python
import json
from typing import List, Dict, Any
from app.synthesis import get_llm_provider
from app.retrieval.retrieve import retrieve_and_rerank

CLASSIFY_SYSTEM_PROMPT = """Classify the user's request as either "simple" or "complex".

"simple" = a single factual lookup answerable from one retrieval pass (e.g., "when is my
dentist appointment", "what's my wifi password").

"complex" = a multi-part request that benefits from being broken into sub-questions (e.g.,
"help me plan my September trip", "what do I need to do before I move apartments",
"summarize everything about my job search").

Respond with ONLY one word: simple or complex."""


def classify_query(query: str) -> str:
    llm = get_llm_provider()
    raw = llm.complete(system_prompt=CLASSIFY_SYSTEM_PROMPT, user_message=query, temperature=0.0)
    normalized = raw.strip().lower()
    return "complex" if "complex" in normalized else "simple"
```

## 2. Build the step-decomposition prompt

```python
PLAN_SYSTEM_PROMPT = """You are the planning layer of a personal memory agent. Break the \
user's request into 2-5 concrete sub-questions that, together, cover what's needed to fully \
address it. Each sub-question should be answerable by searching the user's personal memory \
store.

Respond with ONLY a JSON array of strings, no other text:
["sub-question 1", "sub-question 2", ...]"""


def build_plan(query: str) -> List[str]:
    llm = get_llm_provider()
    raw = llm.complete(system_prompt=PLAN_SYSTEM_PROMPT, user_message=query, temperature=0.2)
    try:
        steps = json.loads(raw)
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            return steps[:5]  # hard cap, avoid runaway plans
    except Exception:
        pass
    return [query]  # fallback: treat the whole query as a single step
```

## 3. Execute the plan — retrieve per step, track gaps

```python
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
```

Reuses your existing `retrieve_and_rerank` from the core RAG pipeline unchanged — no need to duplicate retrieval logic here.

## 4. Synthesize a final answer that separates findings from gaps

```python
FINAL_SYNTHESIS_SYSTEM_PROMPT = """You are a personal memory assistant. You were given a \
multi-part task, broken into sub-questions, each with whatever memories were found (or none). \
Write a clear final answer that:
1. Answers what you can from the memories found, organized by sub-topic.
2. Explicitly lists which sub-questions had NO relevant memories, under a "Still missing"
   section, phrased as things the user hasn't told you yet — not as your own failure.
Keep it concise. Do not invent information not present in the memories."""


def _format_step_results(step_results: List[Dict[str, Any]]) -> str:
    blocks = []
    for step in step_results:
        if step["has_data"]:
            mem_lines = "\n".join(f"- {m['text']}" for m in step["found_memories"])
        else:
            mem_lines = "(no relevant memories found)"
        blocks.append(f"Sub-question: {step['sub_question']}\n{mem_lines}")
    return "\n\n".join(blocks)


def synthesize_plan_answer(original_query: str, step_results: List[Dict[str, Any]]) -> str:
    llm = get_llm_provider()
    context = _format_step_results(step_results)
    user_message = f"Original request: {original_query}\n\n{context}"
    return llm.complete(system_prompt=FINAL_SYNTHESIS_SYSTEM_PROMPT, user_message=user_message, temperature=0.3)
```

## 5. Tie it together in a single entry point

```python
def run_multi_step_task(query: str) -> Dict[str, Any]:
    steps = build_plan(query)
    step_results = execute_plan(steps)
    answer = synthesize_plan_answer(query, step_results)

    missing = [s["sub_question"] for s in step_results if not s["has_data"]]
    all_sources = [m for s in step_results for m in s["found_memories"]]

    return {
        "answer": answer,
        "plan": steps,
        "missing_info": missing,
        "sources": all_sources,
    }
```

---

## 6. Wire into `/query`

Update the `/query` handler (from the action-taking upgrade) to classify first, and route to the planner when complex.

**Update `app/main.py`'s query handler:**
```python
from app.agent.planner import classify_query, run_multi_step_task

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    complexity = classify_query(request.query)

    if complexity == "complex":
        result = run_multi_step_task(request.query)
        return QueryResponse(
            answer=result["answer"],
            action_taken=None,
            action_result=None,
            sources=result["sources"],
        )

    # existing simple-path logic (retrieval + synthesis, or tool-calling from the
    # action-taking upgrade) stays unchanged here
    return handle_query(request.query, request.top_k)
```

Consider adding `plan` and `missing_info` as optional fields on `QueryResponse` (in `app/models.py`) so the client can display the breakdown, not just the final text:
```python
class QueryResponse(BaseModel):
    answer: str
    action_taken: Optional[str] = None
    action_result: Optional[Dict[str, Any]] = None
    sources: List[QueryResult]
    plan: Optional[List[str]] = None
    missing_info: Optional[List[str]] = None
```

---

## 7. Optional: turn gaps into reminders automatically

Since you already have the action-taking system, a nice connective feature: when `missing_info` is non-empty, offer to create a reminder to fill the gap later, rather than just reporting it. Keep this as a follow-up, not required for the base multi-step feature — e.g., a separate `POST /query/fill-gap` endpoint the user can call with one of the `missing_info` items to create a reminder like "Add memory about hotel booking for September trip."

---

## 8. Tests

**`tests/test_planner.py`:**
```python
import json
from unittest.mock import patch, MagicMock

def test_classify_query_complex():
    with patch("app.agent.planner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "complex"
        from app.agent.planner import classify_query
        assert classify_query("help me plan my September trip") == "complex"

def test_build_plan_parses_json_list():
    canned = json.dumps(["What flights are booked?", "What hotel is booked?", "Any visa requirements noted?"])
    with patch("app.agent.planner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = canned
        from app.agent.planner import build_plan
        steps = build_plan("help me plan my September trip")
        assert len(steps) == 3

def test_build_plan_falls_back_on_bad_json():
    with patch("app.agent.planner.get_llm_provider") as mock_llm:
        mock_llm.return_value.complete.return_value = "not json"
        from app.agent.planner import build_plan
        steps = build_plan("help me plan my September trip")
        assert steps == ["help me plan my September trip"]

def test_execute_plan_flags_missing_data():
    with patch("app.agent.planner.retrieve_and_rerank") as mock_retrieve:
        mock_retrieve.side_effect = [[{"text": "Booked flights"}], []]
        from app.agent.planner import execute_plan
        results = execute_plan(["flights?", "hotel?"])
        assert results[0]["has_data"] is True
        assert results[1]["has_data"] is False
```

---

## 9. README update

```markdown
## Multi-Step Tasks

Complex requests (e.g., "help me plan my September trip") are no longer handled with a single
retrieval pass. PAMA first classifies whether a query is simple or complex, and for complex
requests, breaks it into 2-5 sub-questions, retrieves memories for each independently, and
produces a final answer that separates what it found from what's still missing — instead of
guessing or giving an incomplete answer without saying so. Simple factual queries continue to
use the existing single-pass retrieval and synthesis (or tool-calling) path unchanged.
```

---

## Acceptance Criteria

- [ ] `app/agent/planner.py` exists with `classify_query`, `build_plan`, `execute_plan`, `synthesize_plan_answer`, `run_multi_step_task`.
- [ ] `/query` classifies incoming requests and routes complex ones through the planner, simple ones through the existing path unchanged.
- [ ] Plans are capped at 5 steps to avoid runaway LLM calls on a single request.
- [ ] Malformed plan JSON falls back to treating the whole query as one step, rather than crashing.
- [ ] Final answers explicitly separate found information from a "Still missing" section when applicable.
- [ ] `QueryResponse` optionally exposes `plan` and `missing_info` fields.
- [ ] Tests cover classification, plan parsing (success + fallback), and gap detection with mocked LLM/retrieval calls.
- [ ] Manual test: ingest 2-3 memories about one aspect of a trip (e.g., flights only, no hotel/visa info), then query "help me plan my September trip" and confirm the response identifies what's known and explicitly flags what's missing (hotel, visa) rather than fabricating an answer.
