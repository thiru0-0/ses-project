"""
Streamlit demo UI — single app covering three jobs:
1. 📄 Ingest — upload files, paste URLs, paste messages
2. 💬 Chat / Generate — Q&A or structured test case generation
3. 📊 Evaluate — display Ragas evaluation metrics

Communicates with the FastAPI backend via HTTP requests.
"""

import uuid
import requests
import streamlit as st

# --- Configuration ---
API_BASE_URL = "http://localhost:8001"

st.set_page_config(
    page_title="RAG POC (Phase 2)",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state for session scoping
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# --- Sidebar ---
with st.sidebar:
    st.title("🔍 RAG POC")
    st.caption("Phase 2: Test Case Generator")
    
    st.divider()
    
    st.subheader("⚙️ Settings")
    
    # Mode Toggle
    app_mode = st.radio(
        "Generation Mode",
        options=["auto", "qa", "test_case"],
        format_func=lambda x: {
            "auto": "🤖 Auto-Detect (Default)",
            "qa": "💬 General Q&A",
            "test_case": "🎫 Test Case Generator"
        }[x],
        help="Select 'Test Case Generator' to force structured 8-field output."
    )
    
    # Session scoping
    use_session_scope = st.checkbox(
        "Enable Session Scoping", 
        value=False,
        help="If enabled, queries will only search documents ingested during this browser session."
    )
    
    st.divider()

    # Health check
    try:
        health = requests.get(f"{API_BASE_URL}/health", timeout=5).json()
        llm_ok = health.get("providers", {}).get("llm_available", False)
        total_chunks = health.get("vector_store", {}).get("total_chunks", 0)

        st.metric("Indexed Chunks", total_chunks)
        if llm_ok:
            st.success("✅ LLM Connected")
        else:
            st.error("❌ LLM Unavailable")
    except requests.exceptions.ConnectionError:
        st.error("❌ API Server Offline")
        st.caption(f"Start with: `uvicorn src.ragpoc.api.main:app --port 8001`")


# --- Tabs ---
tab_ingest, tab_chat, tab_evaluate = st.tabs(["📄 Ingest", "💬 Chat / Generate", "📊 Evaluate"])


# ═══════════════════════════════════════════════
# TAB 1: INGEST
# ═══════════════════════════════════════════════
with tab_ingest:
    st.header("📄 Unified Ingest")
    st.caption("Upload a file, paste a URL, or paste plain text. The system will auto-detect the input type.")

    # We use a form to collect inputs, though we only expect one to be filled
    with st.form("ingest_form", clear_on_submit=True):
        col_file, col_text = st.columns(2)
        
        with col_file:
            uploaded_file = st.file_uploader(
                "Upload a PDF or DOCX file",
                type=["pdf", "docx"],
            )
            
        with col_text:
            text_input = st.text_area(
                "Or paste a URL, message, or user story",
                height=150,
                placeholder="https://example.com/article  OR  As a user I want to..."
            )
            
        submit_btn = st.form_submit_button("📤 Ingest Content")
        
    if submit_btn:
        if not uploaded_file and not text_input.strip():
            st.warning("Please provide a file or text.")
        else:
            with st.spinner("Processing content..."):
                try:
                    # Prepare multipart form data for /ingest/auto
                    files = {}
                    data = {}
                    
                    if uploaded_file:
                        files["file"] = (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type,
                        )
                    
                    if text_input.strip():
                        data["text"] = text_input.strip()
                        
                    if use_session_scope:
                        data["session_id"] = st.session_state.session_id
                        
                    resp = requests.post(
                        f"{API_BASE_URL}/ingest/auto",
                        files=files if files else None,
                        data=data if data else None,
                        timeout=120,
                    )
                    
                    if resp.status_code == 200:
                        res_data = resp.json()
                        st.success(res_data["message"])
                        with st.expander("Ingest Details"):
                            st.json(res_data)
                    else:
                        st.error(f"Error: {resp.json().get('detail', resp.text)}")
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API server")
                    
    # --- Collection Stats ---
    st.divider()
    if st.button("🔄 Refresh Stats", key="btn_stats"):
        try:
            resp = requests.get(f"{API_BASE_URL}/ingest/status", timeout=10)
            if resp.status_code == 200:
                st.json(resp.json())
        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API server")


# ═══════════════════════════════════════════════
# TAB 2: CHAT / GENERATE
# ═══════════════════════════════════════════════
with tab_chat:
    st.header("💬 Chat / Generate")
    st.caption("Ask questions or generate test cases based on ingested documents.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("metadata"):
                with st.expander("📊 Source Details & Metadata"):
                    st.json(msg["metadata"])

    # Chat input
    if prompt := st.chat_input("Ask a question or paste a user story to generate a test case..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Query the pipeline
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating response..."):
                try:
                    payload = {"question": prompt}
                    if app_mode != "auto":
                        payload["mode"] = app_mode
                    if use_session_scope:
                        payload["session_id"] = st.session_state.session_id
                        
                    resp = requests.post(
                        f"{API_BASE_URL}/query",
                        json=payload,
                        timeout=120,
                    )
                    
                    if resp.status_code == 200:
                        data = resp.json()
                        answer = data["answer"]
                        
                        # Apply human-review styling if flagged
                        if data.get("needs_review"):
                            st.warning("⚠️ **Human Review Recommended**: Generated with low confidence.")
                            
                        st.markdown(answer)

                        # Show metadata and sources
                        metadata = {
                            "mode": data["mode"],
                            "confidence": data["confidence"],
                            "needs_review": data["needs_review"],
                            "declined": data["declined"],
                            "retrieved_chunks": data["retrieved_chunks"],
                            "relevant_chunks": data["relevant_chunks"],
                            "sources": data["sources"],
                        }

                        if data["sources"]:
                            st.caption("**Sources:**")
                            for src in data["sources"]:
                                st.caption(f"  • {src['title']} ({src['source_ref']})")

                        with st.expander("📊 Source Details & Metadata"):
                            st.json(metadata)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "metadata": metadata,
                            }
                        )
                    else:
                        error_msg = f"Error: {resp.json().get('detail', resp.text)}"
                        st.error(error_msg)
                        st.session_state.messages.append(
                            {"role": "assistant", "content": error_msg}
                        )
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to API server")


# ═══════════════════════════════════════════════
# TAB 3: EVALUATE
# ═══════════════════════════════════════════════
with tab_evaluate:
    st.header("📊 Evaluation Dashboard")
    st.caption("Run RAGAS evaluation against the golden Q&A set and ADO test case set.")

    # --- Run Evaluation ---
    st.subheader("Run Evaluation")
    col_run, col_info = st.columns([1, 2])
    with col_run:
        run_eval = st.button(
            "▶️ Run RAGAS Evaluation",
            key="btn_run_eval",
            type="primary",
            help="Runs all golden pairs through the live pipeline and computes metrics.",
        )
    with col_info:
        st.caption(
            "⚠️ This may take several minutes depending on the number of golden Q&A "
            "pairs and LLM response time."
        )

    if run_eval:
        with st.spinner("Running evaluation — this may take a few minutes..."):
            try:
                resp = requests.post(
                    f"{API_BASE_URL}/evaluate/run",
                    timeout=600,
                )
                if resp.status_code == 200:
                    st.success("✅ Evaluation complete!")
                    st.session_state["eval_latest"] = resp.json()
                elif resp.status_code == 404:
                    st.error(f"⚠️ {resp.json().get('detail', 'Golden Q&A set not found.')}")
                else:
                    st.error(f"Evaluation failed: {resp.json().get('detail', resp.text)}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server.")
            except requests.exceptions.Timeout:
                st.error("Evaluation timed out after 10 minutes.")

    st.divider()

    # --- Latest Results ---
    st.subheader("Latest Results")

    latest_data = st.session_state.get("eval_latest")
    if latest_data is None:
        try:
            resp = requests.get(f"{API_BASE_URL}/evaluate/results/latest", timeout=10)
            if resp.status_code == 200:
                latest_data = resp.json()
        except requests.exceptions.ConnectionError:
            pass

    if latest_data:
        ts = latest_data.get("run_timestamp", "Unknown")
        total_q = latest_data.get("total_questions", 0)
        st.caption(f"🕐 Run at: `{ts}` — {total_q} questions evaluated")

        metrics = latest_data.get("metrics", {})
        if metrics:
            metric_display = [
                ("Faithfulness", metrics.get("faithfulness", 0)),
                ("Answer Relevancy", metrics.get("answer_relevancy", 0)),
                ("Context Precision", metrics.get("context_precision", 0)),
                ("Context Recall", metrics.get("context_recall", 0)),
                ("Test Coverage", metrics.get("test_coverage", 0)),
            ]
            
            # Filter out metrics that don't exist (like test coverage if it wasn't an ADO run)
            metric_display = [(n, v) for n, v in metric_display if v is not None]

            cols = st.columns(len(metric_display))
            for col, (name, value) in zip(cols, metric_display):
                col.metric(name, f"{value:.1%}")

            st.write("")
            for name, value in metric_display:
                st.write(f"**{name}** — {value:.1%}")
                st.progress(float(value))

            if metrics.get("note") == "basic_metrics_fallback" or metrics.get("note") == "ado_metrics":
                st.warning(
                    f"⚠️ Note: {metrics.get('note')} used."
                )

            import pandas as pd
            df = pd.DataFrame(
                {
                    "Metric": [m[0] for m in metric_display],
                    "Score": [m[1] for m in metric_display],
                }
            )
            st.bar_chart(df.set_index("Metric"))

        details = latest_data.get("details", [])
        if details:
            st.divider()
            st.subheader("Per-Question Breakdown")
            declined_count = sum(1 for d in details if d.get("declined"))
            avg_conf = sum(d.get("confidence", 0) for d in details) / len(details) if len(details) > 0 else 0
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Questions", len(details))
            c2.metric("Declined", declined_count)
            c3.metric("Avg Confidence", f"{avg_conf:.2f}")

            import pandas as pd
            df_details = pd.DataFrame(
                [
                    {
                        "Question": d.get("question", "")[:80],
                        "Category": d.get("category", ""),
                        "Mode": d.get("mode", "qa"),
                        "Confidence": f"{d.get('confidence', 0):.2f}",
                        "Declined": "❌" if d.get("declined") else "✅",
                        "Retrieved": d.get("retrieved_chunks", 0),
                        "Relevant": d.get("relevant_chunks", 0),
                        "Answer (preview)": d.get("answer", "")[:120],
                    }
                    for d in details
                ]
            )
            st.dataframe(df_details, use_container_width=True)
    else:
        st.info(
            "No evaluation results yet. "
            "Click **▶️ Run RAGAS Evaluation** above to start."
        )

    # --- Historical Runs ---
    st.divider()
    st.subheader("Historical Runs")
    try:
        resp = requests.get(f"{API_BASE_URL}/evaluate/results", timeout=10)
        if resp.status_code == 200:
            history = resp.json()
            runs = history.get("results", [])
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
    except requests.exceptions.ConnectionError:
        st.caption("Could not load history — API server not reachable.")
