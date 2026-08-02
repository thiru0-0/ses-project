"""
Streamlit demo UI — Chat-style interface with session management.

Single input box for all interactions (ingest + query) like ChatGPT/Claude.
Sidebar for session management.
"""

import json
import requests
import streamlit as st


# --- Configuration ---
API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="RAG Test Case Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --- Helper Functions ---
def api_request(method: str, endpoint: str, **kwargs):
    """Make an API request with error handling."""
    try:
        resp = requests.request(method, f"{API_BASE_URL}{endpoint}", timeout=120, **kwargs)
        if resp.status_code >= 400:
            return None, resp.json().get("detail", resp.text)
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to API server"
    except requests.exceptions.Timeout:
        return None, "Request timed out"
    except Exception as e:
        return None, str(e)


def create_session(name: str, description: str = ""):
    data, err = api_request("POST", "/session", json={"name": name, "description": description})
    return data, err


def list_sessions():
    data, err = api_request("GET", "/session")
    return data or [], err


def delete_session(session_id: str):
    data, err = api_request("DELETE", f"/session/{session_id}")
    return data, err


def get_session_stats(session_id: str):
    data, err = api_request("GET", f"/session/{session_id}/stats")
    return data, err


def ingest_file(file, session_id: str):
    files = {"file": (file.name, file.getvalue(), file.type)}
    data = {"session_id": session_id} if session_id else {}
    resp = requests.post(f"{API_BASE_URL}/ingest/file", files=files, data=data, timeout=120)
    if resp.status_code == 200:
        return resp.json(), None
    return None, resp.json().get("detail", resp.text)


def ingest_url(url: str, session_id: str):
    data, err = api_request("POST", "/ingest/url", json={"url": url, "session_id": session_id})
    return data, err


def ingest_message(text: str, session_id: str, source_label: str = "pasted"):
    data, err = api_request("POST", "/ingest/message", json={"text": text, "session_id": session_id, "source_label": source_label})
    return data, err


def query(question: str, session_id: str, mode: str | None = None):
    data, err = api_request("POST", "/query", json={"question": question, "session_id": session_id, "mode": mode})
    return data, err


def check_api_health():
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Health check failed: {resp.status_code}"
    except Exception as e:
        return None, str(e)


# --- Session State Initialization ---
if "sessions" not in st.session_state:
    st.session_state.sessions = []
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_ingest" not in st.session_state:
    st.session_state.pending_ingest = None


# --- Sidebar ---
with st.sidebar:
    st.title("🧪 RAG Test Case Generator")
    st.caption("Multi-session RAG with test case generation")
    st.divider()

    # Health check
    health, health_err = check_api_health()
    if health:
        llm_ok = health.get("providers", {}).get("llm_available", False)
        total_chunks = health.get("vector_store", {}).get("total_chunks", 0)
        st.metric("Total Indexed Chunks", total_chunks)
        if llm_ok:
            st.success("✅ LLM Connected")
        else:
            st.error("❌ LLM Unavailable")
    else:
        st.error(f"❌ API Server Offline: {health_err}")
        st.caption("Start with: `uvicorn src.ragpoc.api.main:app --port 8000`")

    st.divider()

    # Session Management
    st.subheader("📂 Sessions")

    # Refresh sessions
    if st.button("🔄 Refresh Sessions", use_container_width=True):
        sessions, err = list_sessions()
        if err:
            st.error(f"Failed to load sessions: {err}")
        else:
            st.session_state.sessions = sessions

    # Load sessions on first run
    if not st.session_state.sessions:
        sessions, err = list_sessions()
        if not err:
            st.session_state.sessions = sessions

    # Create new session
    with st.expander("➕ Create New Session", expanded=not st.session_state.sessions):
        new_session_name = st.text_input("Session Name", placeholder="e.g., Login Feature Testing")
        new_session_desc = st.text_area("Description (optional)", placeholder="What are you testing?")
        if st.button("Create Session", use_container_width=True, type="primary"):
            if new_session_name.strip():
                session, err = create_session(new_session_name.strip(), new_session_desc.strip())
                if err:
                    st.error(f"Failed to create session: {err}")
                else:
                    st.success(f"Created session: {session['name']}")
                    st.session_state.current_session_id = session["session_id"]
                    st.session_state.messages = []
                    st.rerun()
            else:
                st.warning("Please enter a session name")

    # Session list
    if st.session_state.sessions:
        st.caption("Click a session to switch")
        for session in st.session_state.sessions:
            is_active = session["session_id"] == st.session_state.current_session_id
            col1, col2 = st.columns([4, 1])
            with col1:
                label = f"{'🟢 ' if is_active else '⚪ '}{session['name']}"
                if session.get("description"):
                    label += f" — {session['description'][:30]}..."
                if st.button(label, key=f"session_{session['session_id']}", use_container_width=True):
                    st.session_state.current_session_id = session["session_id"]
                    st.session_state.messages = []
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{session['session_id']}", help="Delete session"):
                    _, err = delete_session(session["session_id"])
                    if err:
                        st.error(f"Failed to delete: {err}")
                    else:
                        if st.session_state.current_session_id == session["session_id"]:
                            st.session_state.current_session_id = None
                            st.session_state.messages = []
                        st.rerun()

    st.divider()

    # Current session stats
    if st.session_state.current_session_id:
        stats, _ = get_session_stats(st.session_state.current_session_id)
        if stats:
            st.metric("Session Chunks", stats.get("total_chunks", 0))

    st.divider()
    st.caption("Built with Ollama · LlamaIndex · ChromaDB · Streamlit")


# --- Main Chat Interface ---
st.header("💬 Chat")

# Show current session
if st.session_state.current_session_id:
    session_name = next(
        (s["name"] for s in st.session_state.sessions if s["session_id"] == st.session_state.current_session_id),
        "Unknown"
    )
    st.caption(f"Session: **{session_name}** (`{st.session_state.current_session_id[:8]}...`)")
else:
    st.warning("⚠️ No session selected. Create or select a session from the sidebar.")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("is_test_case"):
            # Render test case nicely
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
                        st.markdown(f"**Actual Result:** {tc.get('actual_result', '_(to be filled during execution)_')}")
                        status = tc.get("status", "Not Run")
                        status_color = {"Pass": "🟢", "Fail": "🔴", "Not Run": "⚪"}.get(status, "⚪")
                        st.markdown(f"**Status:** {status_color} {status}")
            except json.JSONDecodeError:
                st.markdown(msg["content"])
        else:
            st.markdown(msg["content"])
        
        if msg.get("metadata"):
            with st.expander("📊 Details"):
                st.json(msg["metadata"])

# Chat input - single unified input
if prompt := st.chat_input("Ask a question, paste a URL, upload a file, or say 'generate test case for...'", disabled=not st.session_state.current_session_id):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process the input
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            session_id = st.session_state.current_session_id
            
            # Check if it's a URL
            if prompt.startswith(("http://", "https://")):
                # Ingest URL
                data, err = ingest_url(prompt, session_id)
                if err:
                    answer = f"❌ Failed to ingest URL: {err}"
                else:
                    answer = f"✅ Ingested URL: **{data['title']}** ({data['chunk_count']} chunks)"
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)
            
            # Check if it's a file upload request (we'll handle file upload separately)
            # For now, treat as query
            else:
                # Determine mode - check if user wants test case generation
                mode = None
                if any(kw in prompt.lower() for kw in ["generate test case", "create test case", "test case for"]):
                    mode = "test_case"
                
                data, err = query(prompt, session_id, mode)
                if err:
                    answer = f"❌ Error: {err}"
                    st.error(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    answer = data["answer"]
                    metadata = {
                        "confidence": data["confidence"],
                        "declined": data["declined"],
                        "retrieved_chunks": data["retrieved_chunks"],
                        "relevant_chunks": data["relevant_chunks"],
                        "sources": data["sources"],
                    }
                    
                    # Check if response is test case JSON
                    is_test_case = False
                    if mode == "test_case":
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
                        # Re-render to show formatted test cases
                        st.rerun()
                    else:
                        st.markdown(answer)
                        if data["sources"]:
                            st.caption("**Sources:**")
                            for src in data["sources"]:
                                st.caption(f"  • {src['title']} ({src['source_ref']})")


# --- File Upload Area (below chat) ---
if st.session_state.current_session_id:
    st.divider()
    st.subheader("📎 Quick Ingest")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload PDF/DOCX",
            type=["pdf", "docx"],
            key="file_uploader_main",
            label_visibility="collapsed",
        )
        if uploaded_file and st.button("📤 Ingest File", use_container_width=True):
            with st.spinner("Processing file..."):
                data, err = ingest_file(uploaded_file, st.session_state.current_session_id)
                if err:
                    st.error(f"Failed: {err}")
                else:
                    st.success(f"Ingested: {data['title']} ({data['chunk_count']} chunks)")
                    st.rerun()
    
    with col2:
        url_input = st.text_input(
            "Or paste a URL",
            placeholder="https://example.com/requirements",
            key="url_input_main",
            label_visibility="collapsed",
        )
        if url_input and st.button("🌐 Ingest URL", use_container_width=True):
            with st.spinner("Fetching and indexing..."):
                data, err = ingest_url(url_input, st.session_state.current_session_id)
                if err:
                    st.error(f"Failed: {err}")
                else:
                    st.success(f"Ingested: {data['title']} ({data['chunk_count']} chunks)")
                    st.rerun()