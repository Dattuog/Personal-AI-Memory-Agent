# PAMA - Personal AI Memory Agent

PAMA stores personal text memories, retrieves semantically similar memories, reranks them with recency, and synthesizes grounded answers with Groq.

## Architecture

```text
Manual/API Ingestion -> Chunking -> Embedding Provider -> VectorStore
Clipboard Daemon ----/                              |
                                                    v
Query -> Embedding Provider -> Retrieval -> Recency Rerank -> Groq Synthesis
                                                    |
                                                    v
                                             Sources/Citations
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
```

Use `CHROMA_HOST=chroma` with the compose setup to connect to the Chroma service container. Leave `CHROMA_HOST` blank for embedded local Chroma.

## Design Decisions

The default retrieval score uses a 70/30 blend: 70 percent cosine similarity and 30 percent recency decay. That keeps semantic match as the primary signal while letting newer personal memories win when two candidates are similarly relevant.

Synthesis uses Groq's OpenAI-compatible chat completions API with the default model `llama-3.3-70b-versatile` for fast, free-tier-friendly inference.

The default 7-day half-life is a practical starting point for personal notes, tasks, and clipboard discoveries that often become stale within a week. The default 0.75 similarity threshold filters weak matches before synthesis so the LLM receives a tighter, more trustworthy context window.

The included tests cover chunking, recency math, and API wiring. A production rollout should add the manual eval set described in `PAMA_BUILD_INSTRUCTIONS.md` and record before/after ranking metrics here once real memories and decoys are available.
