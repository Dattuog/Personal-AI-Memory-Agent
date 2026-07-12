import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="PAMA Dashboard", layout="wide")


def api_get(path: str, params: dict[str, Any] | None = None) -> Any | None:
    try:
        response = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Failed to reach API ({path}): {exc}")
        return None


def api_post(path: str, json_body: dict[str, Any] | None = None) -> Any | None:
    try:
        response = requests.post(f"{API_BASE}{path}", json=json_body, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"Failed to reach API ({path}): {exc}")
        return None


def read_local_json(path_str: str) -> list[dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def format_timestamp(value: Any) -> str:
    if not value:
        return "unknown"
    try:
        return datetime.fromtimestamp(float(value)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(value)


def dataframe_with_timestamps(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].apply(format_timestamp)
    if "created_at" in df.columns:
        df["created_at"] = df["created_at"].apply(format_timestamp)
    return df


st.sidebar.title("PAMA Dashboard")
page = st.sidebar.radio(
    "Go to",
    ["Overview", "Ask the Agent", "Memories", "Reminders", "Agent Activity", "Self-Review"],
)

health = api_get("/health")
if health and health.get("status") == "ok":
    st.sidebar.success("API: connected")
else:
    st.sidebar.error("API: unreachable - is uvicorn running?")

if page == "Overview":
    st.title("Overview")

    col1, col2, col3 = st.columns(3)

    memories = api_get("/memories")
    col1.metric("Total Memories", memories["count"] if memories else 0)

    reminders = api_get("/reminders")
    col2.metric("Active Reminders", len(reminders["reminders"]) if reminders else 0)

    agent_log = read_local_json("agent_log.json")
    col3.metric("Agent Decisions Logged", len(agent_log))

    st.subheader("Recent Agent Decisions")
    if agent_log:
        st.dataframe(dataframe_with_timestamps(agent_log[-10:][::-1]), use_container_width=True)
    else:
        st.info("No agent decisions logged yet. Run the clipboard daemon (`python -m app.daemon`) and copy something.")

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
                st.markdown("### Plan")
                for index, step in enumerate(result["plan"], start=1):
                    st.write(f"{index}. {step}")

            if result.get("missing_info"):
                st.markdown("### Still Missing")
                for item in result["missing_info"]:
                    st.write(f"- {item}")

            if result.get("sources"):
                st.markdown("### Sources")
                st.dataframe(pd.DataFrame(result["sources"]), use_container_width=True)

elif page == "Memories":
    st.title("Memories")
    search = st.text_input("Filter by text contains", "")
    limit = st.slider("Limit", min_value=25, max_value=1000, value=200, step=25)

    memories = api_get("/memories", {"limit": limit})
    if memories and memories.get("memories"):
        rows = memories["memories"]
        if search:
            rows = [memory for memory in rows if search.lower() in memory.get("text", "").lower()]

        for memory in rows:
            metadata = memory.get("metadata") or {}
            date_str = format_timestamp(metadata.get("timestamp"))
            source = metadata.get("source", "manual")
            text = memory.get("text", "")
            with st.expander(f"[{date_str}] ({source}) {text[:80]}"):
                st.write(text)
                st.caption(f"ID: {memory.get('id', 'unknown')}")
                st.json(metadata)
    else:
        st.info("No memories yet. Ingest something via POST /ingest.")

elif page == "Reminders":
    st.title("Reminders")
    reminders = api_get("/reminders")
    if reminders and reminders.get("reminders"):
        st.dataframe(dataframe_with_timestamps(reminders["reminders"]), use_container_width=True)
    else:
        st.info("No reminders yet.")

elif page == "Agent Activity":
    st.title("Agent Activity Log")
    st.caption("Full reasoning history from the clipboard daemon and self-review.")

    agent_log = read_local_json("agent_log.json")
    if agent_log:
        df = dataframe_with_timestamps(agent_log[::-1])
        if "action" in df.columns:
            action_filter = st.multiselect("Filter by action", options=sorted(df["action"].dropna().unique()))
            if action_filter:
                df = df[df["action"].isin(action_filter)]
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No decisions logged yet.")

    st.divider()
    st.subheader("Surfaced Connections and Conflicts")
    surfaced = api_get("/surfaced")
    if surfaced:
        st.json(surfaced)
    else:
        st.info("No surfaced items yet.")

elif page == "Self-Review":
    st.title("Scheduled Self-Review")
    st.write(
        "This runs automatically on a schedule using SELF_REVIEW_INTERVAL_HOURS, "
        "but you can trigger it manually here for testing."
    )
    if st.button("Run self-review now", type="primary"):
        with st.spinner("Reviewing memory store..."):
            result = api_post("/self-review/run")
        if result:
            st.success(f"Flagged {result.get('flagged_count', 0)} item(s).")
            st.json(result.get("flagged", []))
