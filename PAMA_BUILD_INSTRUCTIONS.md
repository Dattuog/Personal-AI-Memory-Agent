# PAMA — Personal AI Memory Agent
## Complete Build Instructions

**Stack:** Python, FastAPI, ChromaDB, Anthropic Claude API, Docker

This document is a complete build spec. Follow the phases in order — each phase builds on the previous one and should be independently testable before moving on.

---

## 0. Project Structure

Create this layout first:

```
pama/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entrypoint
│   ├── config.py                # Settings via pydantic-settings
│   ├── models.py                 # Pydantic schemas
│   ├── daemon.py                  # Clipboard watcher daemon
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   └── ingest.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── rerank.py
│   │   └── retrieve.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── base.py               # VectorStore ABC
│   │   ├── chroma_store.py
│   │   └── pinecone_store.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py               # EmbeddingProvider ABC
│   │   ├── local_embedding.py
│   │   └── openai_embedding.py
│   ├── synthesis/
│   │   ├── __init__.py
│   │   └── claude_client.py
│   └── security/
│       ├── __init__.py
│       └── sensitive_filter.py
├── tests/
│   ├── test_chunker.py
│   ├── test_rerank.py
│   └── test_api.py
├── Dockerfile
├── Dockerfile.daemon
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 1. Environment & Config

**`requirements.txt`:**
```
fastapi
uvicorn[standard]
pydantic
pydantic-settings
chromadb
anthropic
sentence-transformers
openai
pinecone-client
pyperclip
python-dotenv
pytest
httpx
```

**`.env.example`:**
```
ANTHROPIC_API_KEY=your_key_here
VECTOR_STORE=chroma          # chroma | pinecone
EMBEDDING_PROVIDER=local     # local | openai
CHROMA_PATH=./chroma_data
CHROMA_HOST=                 # set if using Chroma as a server, else blank for embedded
PINECONE_API_KEY=
PINECONE_INDEX=memories
OPENAI_API_KEY=
RERANK_ALPHA=0.7
DECAY_HALF_LIFE_DAYS=7
SIMILARITY_THRESHOLD=0.75
```

**`app/config.py`:**
```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    anthropic_api_key: str
    vector_store: str = "chroma"
    embedding_provider: str = "local"
    chroma_path: str = "./chroma_data"
    chroma_host: str = ""
    pinecone_api_key: str = ""
    pinecone_index: str = "memories"
    openai_api_key: str = ""
    rerank_alpha: float = 0.7
    decay_half_life_days: float = 7.0
    similarity_threshold: float = 0.75

    class Config:
        env_file = ".env"

settings = Settings()
```

---

## 2. Data Model

**`app/models.py`:**
```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class MemoryEntry(BaseModel):
    id: str
    text: str
    source: str                      # "manual" | "clipboard" | "import"
    timestamp: float
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class IngestRequest(BaseModel):
    text: str
    source: str = "manual"
    tags: List[str] = []

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

class QueryResult(BaseModel):
    text: str
    score: float
    cosine_similarity: float
    decay_score: float
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    answer: str
    sources: List[QueryResult]
```

---

## 3. Storage Layer (Swappable VectorStore)

**`app/storage/base.py`:**
```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStore(ABC):
    @abstractmethod
    def add(self, id: str, embedding: List[float], text: str, metadata: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def query(self, embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Return list of dicts: {id, text, metadata, distance}"""
        ...

    @abstractmethod
    def delete(self, id: str) -> None:
        ...
```

**`app/storage/chroma_store.py`:** implement `ChromaStore(VectorStore)` using `chromadb.PersistentClient` (embedded mode) or `chromadb.HttpClient` (if `CHROMA_HOST` is set, for the server-container setup). Use cosine distance metric on the collection. Store `timestamp` in metadata — required for reranking later.

**`app/storage/pinecone_store.py`:** implement `PineconeStore(VectorStore)` using the `pinecone` SDK. Same interface, upsert with metadata including `text` and `timestamp`.

**`app/storage/__init__.py`:** add a factory function:
```python
def get_vector_store() -> VectorStore:
    # reads settings.vector_store and instantiates the right class
```

---

## 4. Embedding Layer (Swappable EmbeddingProvider)

**`app/embeddings/base.py`:**
```python
from abc import ABC, abstractmethod
from typing import List

class EmbeddingProvider(ABC):
    @abstractmethod
    def embed(self, text: str) -> List[float]:
        ...
```

**`app/embeddings/local_embedding.py`:** `LocalEmbedding(EmbeddingProvider)` using `sentence-transformers` (`all-MiniLM-L6-v2`).

**`app/embeddings/openai_embedding.py`:** `OpenAIEmbedding(EmbeddingProvider)` using `text-embedding-3-small`.

**`app/embeddings/__init__.py`:** factory function `get_embedding_provider()` mirroring the vector store factory.

**Important constraint:** No code outside `app/storage/` or `app/embeddings/` should ever import a concrete class (`ChromaStore`, `PineconeStore`, `LocalEmbedding`, `OpenAIEmbedding`) directly. Everything else depends only on `VectorStore` / `EmbeddingProvider` interfaces, obtained via the factory functions. This is required for the "swap backend via config, zero code changes" architecture goal.

---

## 5. Ingestion Pipeline

**`app/ingestion/chunker.py`:**
- Implement `chunk_text(text: str, max_tokens: int = 300, overlap: int = 50) -> List[str]`.
- Use a simple sentence-boundary-aware splitter (split on sentences, greedily pack into chunks under `max_tokens`, with overlap between consecutive chunks).

**`app/ingestion/ingest.py`:**
- `ingest_entry(text: str, source: str, tags: List[str]) -> List[str]`:
  1. Chunk the text.
  2. Embed each chunk via `get_embedding_provider()`.
  3. Store each chunk in the vector store via `get_vector_store()`, with metadata `{source, tags, timestamp: time.time(), parent_id}`.
  4. Return list of generated chunk IDs.

---

## 6. Retrieval + Recency-Weighted Reranking

**`app/retrieval/rerank.py`:**
```python
import math
import time

def decay_score(entry_timestamp: float, now: float, half_life_days: float) -> float:
    age_days = (now - entry_timestamp) / 86400
    return 0.5 ** (age_days / half_life_days)

def blended_score(cosine_sim: float, decay: float, alpha: float) -> float:
    return alpha * cosine_sim + (1 - alpha) * decay
```

**`app/retrieval/retrieve.py`:**
- `retrieve_and_rerank(query: str, top_k: int) -> List[dict]`:
  1. Embed the query via `get_embedding_provider()`.
  2. Query the vector store for a wider candidate pool (e.g., `k=20`).
  3. Convert Chroma/Pinecone distance to cosine similarity (`similarity = 1 - distance` for Chroma cosine space; Pinecone returns similarity directly if using cosine metric — handle both).
  4. **Pre-filter**: drop candidates with `cosine_similarity < settings.similarity_threshold`.
  5. For each remaining candidate, compute `decay_score` from its metadata timestamp, then `blended_score` using `settings.rerank_alpha` and `settings.decay_half_life_days`.
  6. Sort descending by blended score, return top `top_k`.

---

## 7. Claude Synthesis

**`app/synthesis/claude_client.py`:**
- `synthesize_answer(query: str, retrieved_chunks: List[dict]) -> str`:
  1. Build a prompt: system instruction ("You are a personal memory assistant. Answer using only the provided memories. If the memories don't contain the answer, say so.") + inject retrieved chunk texts with their metadata (source, date).
  2. Call `anthropic.Anthropic().messages.create(...)` with a Claude model.
  3. Return the text response.

---

## 8. FastAPI App

**`app/main.py`:** wire up these endpoints:

| Method | Path | Description |
|---|---|---|
| POST | `/ingest` | Accepts `IngestRequest`, runs `ingest_entry`, returns chunk IDs |
| POST | `/query` | Accepts `QueryRequest`, runs retrieve_and_rerank + synthesize_answer, returns `QueryResponse` |
| GET | `/surfaced` | Returns recently surfaced memories from the clipboard daemon (see Phase 9) |
| GET | `/health` | Returns `{"status": "ok"}`, and optionally pings the vector store |

Use Pydantic models from `app/models.py` for all request/response bodies. Add CORS middleware if you plan to build a frontend later.

---

## 9. Passive Agentic Loop (Clipboard Daemon)

**`app/security/sensitive_filter.py`:**
```python
import re

SENSITIVE_PATTERNS = [
    r'sk-[a-zA-Z0-9]{20,}',
    r'\b\d{13,19}\b',
    r'-----BEGIN.*PRIVATE KEY-----',
]

def looks_sensitive(content: str) -> bool:
    return any(re.search(p, content) for p in SENSITIVE_PATTERNS)
```

**`app/daemon.py`:**
- Async loop polling `pyperclip.paste()` every ~1 second.
- Dedupe via SHA-256 hash comparison against the last seen clipboard value.
- Skip content under ~20 characters, and skip anything matching `looks_sensitive`.
- For new valid content:
  1. Embed it.
  2. Check near-duplicate against existing store (similarity > 0.95) — skip if already stored.
  3. Ingest it (`source="clipboard"`).
  4. Query for related existing memories (similarity > `settings.similarity_threshold`, excluding the just-added entry).
  5. If related memories found, write a record to a small local store (SQLite table or JSON file `surfaced.json`) that the `/surfaced` endpoint reads from.
- Run this as its own process (`python -m app.daemon`), separate from the `uvicorn` process, sharing the same vector store backend.

---

## 10. Dockerization

**`Dockerfile`** (API):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`Dockerfile.daemon`** (clipboard daemon):
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "-m", "app.daemon"]
```

**`docker-compose.yml`:**
```yaml
version: "3.9"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - chroma_data:/app/chroma_data
    depends_on:
      - chroma

  daemon:
    build:
      context: .
      dockerfile: Dockerfile.daemon
    env_file: .env
    volumes:
      - chroma_data:/app/chroma_data
    depends_on:
      - chroma

  chroma:
    image: chromadb/chroma
    ports:
      - "8001:8000"
    volumes:
      - chroma_server_data:/chroma/chroma

volumes:
  chroma_data:
  chroma_server_data:
```

Note: running Chroma as its own server container (rather than embedded mode inside the API container) decouples storage lifecycle from the app, which makes the later Pinecone swap architecturally consistent. Set `CHROMA_HOST=chroma` in `.env` when using this setup, and update `ChromaStore` to use `chromadb.HttpClient(host=...)` when that variable is set.

---

## 11. Testing

- `tests/test_chunker.py`: verify chunk sizes stay under `max_tokens`, overlap is present, no data loss across chunks.
- `tests/test_rerank.py`: verify `decay_score` returns 1.0 at age=0, 0.5 at age=half_life, and monotonically decreases; verify `blended_score` behaves correctly at alpha=0 and alpha=1 edge cases.
- `tests/test_api.py`: use `httpx`/`TestClient` to hit `/ingest` then `/query` end-to-end against a temporary Chroma path, and assert the answer references ingested content.
- Build a small manual eval set (15–20 time-sensitive queries with known-correct recent memories vs. decoy older memories) and confirm reranking improves ranking of the correct memory vs. plain cosine similarity alone. Record this as a before/after metric for the README.

---

## 12. README

Write a `README.md` covering:
- Architecture diagram (ASCII or a simple image) showing: Ingestion → Chunking → Embedding → VectorStore, and Query → Embedding → Retrieval → Reranking → Claude Synthesis, plus the Clipboard Daemon as a parallel process feeding into the same store.
- Setup instructions (`.env` config, `docker compose up`).
- Design decisions section: why 70/30 cosine/decay blend, why 7-day half-life, why 0.75 similarity threshold — justify these numbers if possible using the eval set results from Phase 11.
- A note on the swap architecture: how to switch `VECTOR_STORE` and `EMBEDDING_PROVIDER` via `.env` alone, with no code changes required.

---

## Build Order (recommended)

1. Phase 0–2: scaffolding, config, data model.
2. Phase 3–4: storage + embedding interfaces (build both concrete implementations for each even if you only use one at first — this is what makes the "swappable" claim real).
3. Phase 5–7: ingestion → retrieval/reranking → Claude synthesis, tested independently via a script before wiring into FastAPI.
4. Phase 8: FastAPI endpoints wiring everything together.
5. Phase 9: clipboard daemon (build last — most independent piece, most fun to demo).
6. Phase 10: Dockerize everything, verify the Chroma → Pinecone swap actually works with only `.env` changes (no code edits).
7. Phase 11–12: tests, eval set, README.

## Acceptance Criteria (what "done" looks like)

- [ ] Can ingest 500–2,000 text entries without errors.
- [ ] `/query` returns a Claude-synthesized answer grounded in retrieved memories, with citations/sources in the response.
- [ ] Reranking demonstrably improves ranking on a labeled time-sensitive eval set (record the before/after numbers).
- [ ] Clipboard daemon auto-ingests new clipboard content, skips secrets/duplicates, and surfaces related memories via `/surfaced`.
- [ ] Swapping `VECTOR_STORE=chroma` → `VECTOR_STORE=pinecone` (or `EMBEDDING_PROVIDER=local` → `openai`) requires only an `.env` change and container restart — no code edits.
- [ ] Full stack runs via `docker compose up`.
