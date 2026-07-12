# PAMA - Personal AI Memory Agent

PAMA stores personal text memories, retrieves semantically similar memories, reranks them with recency, and synthesizes grounded answers with Groq. It also includes agentic capabilities for reasoning over clipboard content, taking actions, scheduled self-review, and multi-step planning.

## Architecture

```text
Manual/API Ingestion -> Chunking -> Embedding Provider -> VectorStore
Clipboard Daemon ----/                              |
                                                    v
Query -> Classification -> Retrieval/Planning -> Groq Synthesis or Tool Call
                                                    |
                                                    v
                                      Sources/Citations/Actions/Reminders

Scheduled Self-Review -> VectorStore Scan -> Groq Reasoning -> Reminders
```

## Setup

1. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
2. Start the full stack:

```bash
docker compose up --build
```

3. Use the API:

```bash
curl -X POST http://localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"text":"Renew my passport before the September trip.","tags":["travel"]}'

curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What travel admin should I remember?","top_k":5}'
```

Run the clipboard daemon outside Docker with:

```bash
python -m app.daemon
```

## Configuration

Switch backends in `.env` and restart. No application code changes are required.

```text
VECTOR_STORE=chroma       # chroma | pinecone
EMBEDDING_PROVIDER=local  # local | openai
SELF_REVIEW_INTERVAL_HOURS=24
SELF_REVIEW_COOLDOWN_HOURS=48
```

Use `CHROMA_HOST=chroma` with the compose setup to connect to the Chroma service container. Leave `CHROMA_HOST` blank for embedded local Chroma.

Scheduled self-review currently requires Chroma for full-store scans. `PineconeStore.update_metadata()` is implemented, but `PineconeStore.list_all()` raises `NotImplementedError` until a project-specific Pinecone listing strategy is added.

## Agentic Capabilities

PAMA goes beyond passive storage-and-retrieval in four ways:

- **Reasoning daemon** - the clipboard watcher no longer relies on a fixed similarity threshold. New clipboard content, along with related existing memories, is handed to Groq, which decides whether to store it silently, surface a meaningful connection, flag a conflict, or skip it as noise.
- **Action-taking** - through Groq tool-calling, `/query` can create reminders or send desktop notifications when a request is actionable, such as "remind me to renew my passport". The reasoning daemon can also spawn reminders when it detects a deadline in newly captured content.
- **Scheduled self-review** - PAMA periodically reviews its memory store for time-sensitive items and creates reminders proactively. Reviewed memories are cooled down for 48 hours by default. Trigger on demand with `POST /self-review/run`.
- **Multi-step tasks** - complex requests are broken into sub-questions, each retrieved independently, with the final answer separating what was found from what is still missing.

Reminders are listed at `GET /reminders`. Reasoning and self-review decisions are logged to `agent_log.json`, and notable surfaced/conflict decisions are exposed through `GET /surfaced`.

## API Notes

- `POST /query` returns `answer`, `sources`, and optionally `action_taken`, `action_result`, `plan`, and `missing_info`.
- `GET /reminders` returns persisted reminders from `reminders.json`.
- `POST /self-review/run` runs the autonomous review immediately for testing or manual checks.

## Design Decisions

The default retrieval score uses a 70/30 blend: 70 percent cosine similarity and 30 percent recency decay. That keeps semantic match as the primary signal while letting newer personal memories win when two candidates are similarly relevant.

Synthesis uses Groq's OpenAI-compatible chat completions API with the default model `llama-3.3-70b-versatile` for fast, free-tier-friendly inference.

The default 7-day half-life is a practical starting point for personal notes, tasks, and clipboard discoveries that often become stale within a week. The default 0.75 similarity threshold filters weak matches before synthesis so the LLM receives a tighter, more trustworthy context window.

The included tests cover chunking, recency math, API wiring, action execution, reasoning fallback, self-review, and multi-step planning. A production rollout should add the manual eval set described in `PAMA_BUILD_INSTRUCTIONS.md` and record before/after ranking metrics here once real memories and decoys are available.
