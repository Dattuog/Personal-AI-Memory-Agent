# PAMA — Add a Streamlit Dashboard

## Goal

Add a simple visual dashboard so you can see what the agent is doing without curl/PowerShell — browse memories, ask questions and see the plan/sources/actions, view reminders, and watch the reasoning daemon's decision log (surfaced connections, flagged conflicts, self-review activity) update live.

This is a **separate, standalone Streamlit app** that talks to the existing FastAPI backend over HTTP — it does not replace or modify the API, it's a new file that reads from it.

---

## 1. Add one missing backend endpoint first

The dashboard needs to list all memories, which isn't currently exposed over HTTP (only used internally by self-review via `store.list_all()`).

**Add to `app/main.py`:**
```python
from app.storage import get_vector_store

@app.get("/memories")
def list_memories(limit: int = 200):
    store = get_vector_store()
    memories = store.list_all(limit=limit)
    return {"count": len(memories), "memories": memories}
```

---

## 2. Install Streamlit

Add to `requirements.txt`:
```text
streamlit
requests
pandas
```

Install:
```powershell
pip install streamlit requests pandas
```

---

## 3. Build the dashboard app

**`dashboard/streamlit_app.py`** (new folder, new file):

```python
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="PAMA Dashboard", layout="wide")


def api_get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to reach API ({path}): {e}")
        return None


def api_post(path: str, json_body: dict | None = None):
    try:
        resp = requests.post(f"{API_BASE}{path}", json=json_body, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to reach API ({path}): {e}")
        return None


def read_local_log(path_str: str) -> list:
    path = Path(path_str)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text())
    except Exception:
        return []


# ---------- Sidebar navigation ----------
st.sidebar.title("PAMA Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Ask the Agent", "Memories", "Reminders", "Agent Activity", "Self-Review"],
)

health = api_get("/health")
if health and health.get("status") == "ok":
    st.sidebar.success("API: connected")
else:
    st.sidebar.error("API: unreachable — is uvicorn running?")


# ---------- Overview ----------
if page == "Overview":
    st.title("Overview")

    col1, col2, col3 = st.columns(3)

    memories = api_get("/memories")
    mem_count = memories["count"] if memories else 0
    col1.metric("Total Memories", mem_count)

    reminders = api_get("/reminders")
    rem_count = len(reminders["reminders"]) if reminders else 0
    col2.metric("Active Reminders", rem_count)

    agent_log = read_local_log("agent_log.json")
    col3.metric("Agent Decisions Logged", len(agent_log))

    st.subheader("Recent Agent Decisions")
    if agent_log:
        recent = agent_log[-10:][::-1]
        df = pd.DataFrame(recent)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].apply(lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No agent decisions logged yet. Run the clipboard daemon (`python -m app.daemon`) and copy something.")


# ---------- Ask the Agent ----------
elif page == "Ask the Agent":
    st.title("Ask the Agent")
    query = st.text_input("Your question", placeholder="e.g. What travel admin should I remember?")
    top_k = st.slider("Top K sources", min_value=1, max_value=10, value=5)

    if st.button("Ask", type="primary") and query:
        with st.spinner("Thinking..."):
            result = api_post("/query", {"query": query, "top_k": top_k})

        if result:
            st.markdown("### Answer")
            st.write(result.get("answer", ""))

            if result.get("action_taken"):
                st.success(f"Action taken: **{result['action_taken']}**")
                st.json(result.get("action_result", {}))

            if result.get("plan"):
                st.markdown("### Plan (multi-step breakdown)")
                for i, step in enumerate(result["plan"], 1):
                    st.write(f"{i}. {step}")

            if result.get("missing_info"):
                st.markdown("### Still Missing")
                for item in result["missing_info"]:
                    st.write(f"- {item}")

            if result.get("sources"):
                st.markdown("### Sources")
                df = pd.DataFrame(result["sources"])
                st.dataframe(df, use_container_width=True)


# ---------- Memories ----------
elif page == "Memories":
    st.title("Memories")
    search = st.text_input("Filter by text contains", "")

    memories = api_get("/memories")
    if memories and memories["memories"]:
        rows = memories["memories"]
        if search:
            rows = [m for m in rows if search.lower() in m["text"].lower()]

        for m in rows:
            ts = m["metadata"].get("timestamp")
            date_str = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "unknown"
            source = m["metadata"].get("source", "manual")
            with st.expander(f"[{date_str}] ({source}) {m['text'][:80]}"):
                st.write(m["text"])
                st.caption(f"ID: {m['id']}")
                st.json(m["metadata"])
    else:
        st.info("No memories yet. Ingest something via POST /ingest.")


# ---------- Reminders ----------
elif page == "Reminders":
    st.title("Reminders")
    reminders = api_get("/reminders")
    if reminders and reminders["reminders"]:
        df = pd.DataFrame(reminders["reminders"])
        if "created_at" in df.columns:
            df["created_at"] = df["created_at"].apply(lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M"))
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No reminders yet.")


# ---------- Agent Activity ----------
elif page == "Agent Activity":
    st.title("Agent Activity Log")
    st.caption("Full reasoning history from the clipboard daemon and self-review.")

    agent_log = read_local_log("agent_log.json")
    if agent_log:
        df = pd.DataFrame(agent_log[::-1])
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].apply(lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"))

        action_filter = st.multiselect(
            "Filter by action",
            options=df["action"].unique().tolist() if "action" in df.columns else [],
            default=None,
        )
        if action_filter:
            df = df[df["action"].isin(action_filter)]

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No decisions logged yet.")

    st.divider()
    st.subheader("Surfaced (notable connections & conflicts)")
    surfaced = api_get("/surfaced")
    if surfaced:
        st.json(surfaced)


# ---------- Self-Review ----------
elif page == "Self-Review":
    st.title("Scheduled Self-Review")
    st.write(
        "This runs automatically on a schedule (see `SELF_REVIEW_INTERVAL_HOURS` in `.env`), "
        "but you can trigger it manually here for testing."
    )
    if st.button("Run self-review now", type="primary"):
        with st.spinner("Reviewing memory store..."):
            result = api_post("/self-review/run")
        if result:
            st.success(f"Flagged {result.get('flagged_count', 0)} item(s).")
            st.json(result.get("flagged", []))
```

---

## 4. Run the dashboard

Make sure the FastAPI server (`uvicorn app.main:app --reload --port 8000`) is already running in another terminal, then:

```powershell
streamlit run dashboard\streamlit_app.py
```

This opens a browser tab, typically at `http://localhost:8501`.

---

## 5. What each page shows

| Page | Purpose |
|---|---|
| **Overview** | Quick counts (memories, reminders, logged decisions) + the 10 most recent agent decisions |
| **Ask the Agent** | Same as `/query` but visual — shows the answer, plan (if complex), missing info, action taken, and sources in a table instead of raw JSON |
| **Memories** | Browse/search everything stored, expandable to see full text + metadata |
| **Reminders** | Table view of everything in `reminders.json` |
| **Agent Activity** | Full decision log from the clipboard daemon and self-review, filterable by action type (`ingest_silent`, `ingest_and_surface`, `ingest_and_flag_conflict`, `self_review_flag`) |
| **Self-Review** | Manual trigger button, useful for demoing the autonomous review without waiting 24 hours |

---

## 6. Notes on running everything together

You'll now have up to 3 things running simultaneously, each in its own terminal:
1. `uvicorn app.main:app --reload --port 8000` — the API
2. `python -m app.daemon` — the clipboard reasoning daemon (optional, only if testing that)
3. `streamlit run dashboard\streamlit_app.py` — the dashboard

The dashboard reads `agent_log.json` and `reminders.json` directly off disk (since they're local files) and calls the API over HTTP for everything else — so make sure the dashboard is run from the project root (same directory as those JSON files), not from inside `dashboard/`.

---

## 7. Optional polish (skip for now, nice later)

- Auto-refresh: wrap the Overview/Agent Activity pages with `st.rerun()` on a timer (e.g., via `streamlit-autorefresh` package) so you can watch the daemon's decisions appear live without manually refreshing.
- A "clear all data" button on a Settings page that wipes `chroma_data/`, `reminders.json`, `agent_log.json` for quick resets between test runs, instead of doing it manually via PowerShell each time.

## Acceptance Criteria

- [ ] `GET /memories` endpoint added and returns stored memories.
- [ ] `dashboard/streamlit_app.py` created with all 6 pages.
- [ ] `streamlit run dashboard\streamlit_app.py` launches without errors while the API is running.
- [ ] Ask the Agent page correctly displays plan/missing_info/action_taken/sources for both simple and complex queries.
- [ ] Agent Activity page reflects daemon decisions after copying test clipboard content.
- [ ] Self-Review page's manual trigger button works and displays flagged results.
