"""
Streamlit demo UI — single app covering three jobs:
1. 📄 Ingest    — upload files, paste URLs, paste messages (scoped to a session)
2. 💬 Ask       — chat-style Q&A with source citations (QA or Test-Case mode)
3. 📊 Evaluate  — run RAGAS evaluation and inspect metrics

Communicates with the FastAPI backend (port 8001) via HTTP requests.
"""

import json

import pandas as pd
import streamlit as st

# --- Configuration ---
API_BASE_URL = st.sidebar.text_input(
    "API Server",
    value="http://localhost:8001",
    help="Address of the FastAPI backend.",
    key="api_base_url",
)

st.set_page_config(
    page_title="RAG Test Case Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helper Functions ---
def api_request(method: str, endpoint: str, **kwargs):
    """Make an API request with unified error handling."""
    import requests
    try:
        resp = requests.request(method, f"{API_BASE_URL}{endpoint}", timeout=300, **kwargs)
        if resp.status_code >= 400:
            return None, resp.json().get("detail", resp.text)
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API server"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)


def create_session(name, description=""):
    return api_request("POST", "/session", json={"name": name, "description": description})


def list_sessions():
    data, err = api_request("GET", "/session")
    return data or [], err


def delete_session(session_id):
    return api_request("DELETE", f"/session/{session_id}")


def get_session_stats(session_id):
    return api_request("GET", f"/session/{session_id}/stats")


def ingest_file(file, session_id):
    import requests
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"session_id": session_id} if session_id else {}
    resp = requests.post(f"{API_BASE_URL}/ingest/file", files=files, data=data, timeout=300)
    if resp.status_code == 200:
        return resp.json(), None
    return None, resp.json().get("detail", resp.text)


def ingest_url(url, session_id):
    return api_request("POST", "/ingest/url", json={"url": url, "session_id": session_id})


def ingest_message(text, session_id):
    return api_request(
        "POST", "/ingest/message",
        json={"text": text, "source_label": "pasted", "session_id": session_id},
    )


def query(question, session_id, mode=None):
    return api_request(
        "POST", "/query",
        json={"question": question, "session_id": session_id, "mode": mode},
    )


def ingest_status(session_id=None):
    return api_request("GET", "/ingest/status", params={"session_id": session_id})


def check_api_health():
    data, err = api_request("GET", "/health")
    return data, err


# --- Session State ---
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_latest" not in st.session_state:
    st.session_state.eval_latest = None


# --- Sidebar ---
with st.sidebar:
    st.title("🧪 RAG API")
    st.caption("Ingest · Ask · Evaluate")
    st.divider()

    health, health_err = check_api_health()
    if health:
        llm_ok = health.get("providers", {}).get("llm_available", False)
        provider = health.get("providers", {}).get("llm_provider", "?")
        total_chunks = health.get("vector_store", {}).get("total_chunks", 0)
        st.metric("Indexed Chunks", total_chunks)
        if llm_ok:
            st.success(f"✅ LLM Connected ({provider})")
        else:
            st.error("❌ LLM Unavailable")
    else:
        st.error(f"❌ API Offline: {health_err}")
        st.caption("Start with: `uvicorn src.ragpoc.api.main:app --reload --port 8001`")

    st.divider()

    # Session management
    st.subheader("📂 Sessions")

    if st.button("🔄 Refresh Sessions", use_container_width=True):
        sessions, err = list_sessions()
        if err:
            st.error(f"Failed to load sessions: {err}")
        else:
            st.session_state.sessions = sessions

    if not st.session_state.sessions:
        sessions, err = list_sessions()
        if not err:
            st.session_state.sessions = sessions

    with st.expander("➕ Create Session", expanded=not st.session_state.sessions):
        name = st.text_input("Session Name", placeholder="e.g., Telemetry Verification")
        desc = st.text_area("Description (optional)", height=80)
        if st.button("Create Session", use_container_width=True, type="primary"):
            if name.strip():
                session, err = create_session(name.strip(), desc.strip())
                if err:
                    st.error(f"Failed: {err}")
                else:
                    st.success(f"Created: {session['name']}")
                    st.session_state.current_session_id = session["session_id"]
                    st.session_state.messages = []
                    st.rerun()
            else:
                st.warning("Enter a session name")

    if st.session_state.sessions:
        for session in st.session_state.sessions:
            is_active = session["session_id"] == st.session_state.current_session_id
            col1, col2 = st.columns([4, 1])
            with col1:
                label = f"{'🟢 ' if is_active else '⚪ '}{session['name']}"
                if st.button(label, key=f"sel_{session['session_id']}", use_container_width=True):
                    st.session_state.current_session_id = session["session_id"]
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{session['session_id']}",
                             help="Delete session"):
                    _, err = delete_session(session["session_id"])
                    if not err:
                        if st.session_state.current_session_id == session["session_id"]:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()

    if st.session_state.current_session_id:
        stats, _ = get_session_stats(st.session_state.current_session_id)
        if stats:
            st.metric("Session Chunks", stats.get("total_chunks", 0))

    st.divider()
    st.caption("Groq / Ollama · LlamaIndex · ChromaDB · Streamlit")


# --- Main Tabs ---
tab_ingest, tab_ask, tab_evaluate = st.tabs(["📄 Ingest", "💬 Ask", "📊 Evaluate"])


# ═══════════════════════════════════════════════
# TAB 1: INGEST
# ═══════════════════════════════════════════════
with tab_ingest:
    st.header("📄 Ingest")
    st.caption("Upload files, paste URLs, or paste a text/message (incl. ADO work items) into a session.")

    if not st.session_state.current_session_id:
        st.warning("⚠️ Select or create a session in the sidebar first.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload File")
        uploaded_file = st.file_uploader("PDF or DOCX", type=["pdf", "docx"], key="fu_ingest")
        if uploaded_file and st.button("📤 Upload & Index", key="btn_file"):
            if not st.session_state.current_session_id:
                st.error("Select a session first")
            else:
                with st.spinner("Processing file..."):
                    data, err = ingest_file(uploaded_file, st.session_state.current_session_id)
                    if err:
                        st.error(f"Failed: {err}")
                    else:
                        st.success(data["message"])
                        st.json(data)
                        st.rerun()

        st.divider()
        st.subheader("Scrape URL")
        url_input = st.text_input("Enter a URL", placeholder="https://example.com/spec")
        if st.button("🌐 Scrape & Index", key="btn_url"):
            if not st.session_state.current_session_id:
                st.warning("Select a session first")
            elif not url_input:
                st.warning("Enter a URL")
            else:
                with st.spinner("Fetching and extracting..."):
                    data, err = ingest_url(url_input, st.session_state.current_session_id)
                    if err:
                        st.error(f"Failed: {err}")
                    else:
                        st.success(data["message"])
                        st.json(data)

    with col2:
        st.subheader("Paste Text / Work Item")
        message_input = st.text_area(
            "Paste a user story, requirement, chat thread, or any text content.",
            height=320,
            placeholder="e.g. Paste a Work Item / User Story with title, description, acceptance criteria...",
            key="face_message",
        )
        if st.button("📋 Index Pasted Text", key="btn_message"):
            if not st.session_state.current_session_id:
                st.warning("Select a session first")
            elif not message_input.strip():
                st.warning("Paste some text first")
            else:
                with st.spinner("Indexing..."):
                    data, err = ingest_message(message_input, st.session_state.current_session_id)
                    if err:
                        st.error(f"Failed: {err}")
                    else:
                        st.success(data["message"])
                        st.json(data)

    st.divider()
    if st.button("🔄 Refresh Collection Stats", key="btn_stats"):
        if st.session_state.current_session_id:
            data, err = ingest_status(st.session_state.current_session_id)
        else:
            data, err = ingest_status()
        if err:
            st.error(err)
        else:
            st.json(data)


# ═══════════════════════════════════════════════
# TAB 2: ASK
# ═══════════════════════════════════════════════
with tab_ask:
    st.header("💬 Ask")
    st.caption("Ask about the indexed content. Answers are grounded in the session only.")

    if not st.session_state.current_session_id:
        st.warning("⚠️ Select or create a session in the sidebar first.")

    col_mode = st.columns(2)[0]
    output_mode = col_mode.radio(
        "Output Mode",
        ["Auto", "QA (Answer)", "Test Case"],
        horizontal=True,
        help="Test Case generates structured QA test cases (best for ADO user stories).",
    )
    mode = {"QA (Answer)": "qa", "Test Case": "test_case"}.get(output_mode)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("is_test_case"):
                try:
                    test_cases = json.loads(msg["content"])
                    for tc in test_cases:
                        with st.expander(f"📋 {tc.get('test_id', 'TC_001')}: {tc.get('title', 'Untitled')}", expanded=True):
                            st.markdown(f"**Requirement Reference:** {tc.get('requirement_reference', 'N/A')}")
                            st.markdown(f"**Description:** {tc.get('description', 'N/A')}")
                            st.markdown("**Preconditions:**")
                            for pc in tc.get("preconditions", []):
                                st.markdown(f"- {pc}")
                            st.markdown("**Test Steps:**")
                            for i, step in enumerate(tc.get("test_steps", []), 1):
                                st.markdown(f"{i}. {step}")
                            st.markdown(f"**Expected Result:** {tc.get('expected_result', 'N/A')}")
                            status = tc.get("status", "Not Run")
                            st.markdown(f"**Status:** ⚪ {status}")
                except json.JSONDecodeError:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])

            if msg.get("metadata"):
                with st.expander("📊 Details"):
                    st.json(msg["metadata"])

    if prompt := st.chat_input(
        "Ask a question... (paste a user story + 'test case for...' to generate test cases)",
        disabled=not st.session_state.current_session_id,
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving, grading, generating..."):
                # If user typed "test case", force test_case mode
                if any(kw in prompt.lower() for kw in
                       ["generate test case", "test case for", "create test cases"]):
                    use_mode = "test_case"
                else:
                    use_mode = mode

                data, err = query(prompt, st.session_state.current_session_id, use_mode)
                if err:
                    ans = f"{err}"
                    st.error(ans)
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                else:
                    answer = data["answer"]
                    metadata = {
                        "confidence": data["confidence"],
                        "declined": data["declined"],
                        "retrieved_chunks": data["retrieved_chunks"],
                        "relevant_chunks": data["relevant_chunks"],
                        "sources": data["sources"],
                    }

                    is_test_case = False
                    if use_mode == "test_case":
                        try:
                            json.loads(answer)
                            is_test_case = True
                        except json.JSONDecodeError:
                            pass

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "metadata": metadata,
                        "is_test_case": is_test_case,
                    })

                    if is_test_case:
                        st.rerun()
                    else:
                        st.markdown(answer)
                        if data["declined"]:
                            st.caption("ℹ️ The pipeline declined (no relevant context).")
                        if data["sources"]:
                            st.caption("**Sources:**")
                            for src in data["sources"]:
                                st.caption(f"  • {src['title']} ({src['source_ref']})")

    if st.button("🗑️ Clear Chat", disabled=not st.session_state.messages):
        st.session_state.messages = []
        st.rerun()


# ═══════════════════════════════════════════════
# TAB 3: EVALUATE
# ═══════════════════════════════════════════════
with tab_evaluate:
    st.header("📊 Evaluation Dashboard")
    st.caption("Run RAGAS evaluation against the golden Q&A set and inspect metrics.")

    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_eval = st.button(
            "▶️ Run RAGAS Evaluation",
            type="primary",
            help="Runs all golden Q&A pairs through the live pipeline and computes metrics.",
        )
    with col_info:
        st.caption("⚠️ May take several minutes depending on golden set size and LLM speed.")

    if run_eval:
        with st.spinner("Running evaluation — this may take a few minutes..."):
            data, err = api_request("POST", "/evaluate/run")
            if err:
                st.error(err)
            else:
                st.success("✅ Evaluation complete!")
                st.session_state.eval_latest = data

    st.divider()
    st.subheader("Latest Results")

    latest_data = st.session_state.get("eval_latest")
    if latest_data is None:
        latest_data, _ = api_request("GET", "/evaluate/results/latest")

    if latest_data:
        ts = latest_data.get("run_timestamp", "Unknown")
        total_q = latest_data.get("total_questions", 0)
        st.caption(f"🕐 Run at: `{ts}` — {total_q} questions evaluated")

        metrics = latest_data.get("metrics", {})
        if metrics:
            metric_rows = [
                ("Faithfulness", metrics.get("faithfulness", 0)),
                ("Answer Relevancy", metrics.get("answer_relevancy", 0)),
                ("Context Precision", metrics.get("context_precision", 0)),
                ("Context Recall", metrics.get("context_recall", 0)),
            ]
            cols = st.columns(4)
            for col, (name, value) in zip(cols, metric_rows):
                col.metric(name, f"{value:.1%}")

            st.write("")
            for name, value in metric_rows:
                st.write(f"**{name}** — {value:.1%}")
                st.progress(float(value))

            if metrics.get("note") == "basic_metrics_fallback":
                st.warning("⚠️ RAGAS library scoring unavailable — basic word-overlap metrics shown instead.")

        details = latest_data.get("details", [])
        if details:
            st.divider()
            st.subheader("Per-Question Breakdown")
            declined_count = sum(1 for d in details if d.get("declined"))
            avg_conf = sum(d.get("confidence", 0) for d in details) / len(details)
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Questions", len(details))
            c2.metric("Declined", declined_count)
            c3.metric("Avg Confidence", f"{avg_conf:.2f}")

            rows = [
                {
                    "Question": d.get("question", "")[:80],
                    "Category": d.get("category", ""),
                    "Confidence": f"{d.get('confidence', 0):.2f}",
                    "Declined": "❌" if d.get("declined") else "✅",
                    "Retrieved": d.get("retrieved_chunks", 0),
                    "Relevant": d.get("relevant_chunks", 0),
                    "Answer (preview)": d.get("answer", "")[:120],
                }
                for d in details
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("No evaluation results yet. Click **▶️ Run RAGAS Evaluation** above to start.")

    st.divider()
    st.subheader("Historical Runs")
    data, _ = api_request("GET", "/evaluate/results")
    if data:
        runs = data.get("results", [])
        if runs:
            for run in runs:
                n = run.get("total_questions", "?")
                m = run.get("metrics", {})
                summary = "  |  ".join(
                    f"{k.replace('_', ' ').title()}: {v:.1%}"
                    for k, v in m.items()
                    if isinstance(v, float)
                )
                with st.expander(f"📄 `{run['filename']}` — {n} questions"):
                    st.caption(f"🕐 {run.get('run_timestamp', '')}")
                    if summary:
                        st.caption(summary)
                    else:
                        st.caption("No metrics available.")
        else:
            st.caption("No historical runs yet.")
    else:
        st.caption("Could not load history — API server not reachable.")