"""
Session management API routes.

POST /session           — create a new session
GET  /session           — list all sessions
GET  /session/{id}      — get session details
DELETE /session/{id}    — delete a session
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.ragpoc.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


class SessionCreateRequest(BaseModel):
    name: str
    description: str = ""


class SessionResponse(BaseModel):
    session_id: str
    name: str
    description: str
    created_at: str
    chunk_count: int


# In-memory session registry (can be persisted to a file or DB later)
_sessions: dict[str, dict] = {}


def _get_vector_store(session_id: Optional[str] = None) -> VectorStore:
    """Get a VectorStore instance for the given session."""
    from src.ragpoc.api.main import vector_store as global_vector_store
    
    if session_id:
        # Create a new VectorStore with session_id for filtering
        # The global vector_store is used for indexing, but we need a fresh one for queries
        # Actually, let's use the global one but set the session
        global_vector_store.set_session(session_id)
        return global_vector_store
    
    # For listing sessions, use global without filter
    global_vector_store.set_session(None)
    return global_vector_store


@router.post("", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    session = {
        "session_id": session_id,
        "name": request.name,
        "description": request.description,
        "created_at": now,
        "updated_at": now,
    }
    
    _sessions[session_id] = session
    logger.info("Created session: %s (%s)", session_id, request.name)
    
    # Get chunk count for this session
    store = _get_vector_store(session_id)
    stats = store.get_stats(session_id)
    
    return SessionResponse(
        session_id=session_id,
        name=request.name,
        description=request.description,
        created_at=now,
        chunk_count=stats.get("total_chunks", 0),
    )


@router.get("", response_model=list[SessionResponse])
async def list_sessions():
    """List all sessions with their chunk counts."""
    store = _get_vector_store(None)
    sessions_list = []
    
    for session_id, session in _sessions.items():
        stats = store.get_stats(session_id)
        sessions_list.append(
            SessionResponse(
                session_id=session_id,
                name=session["name"],
                description=session["description"],
                created_at=session["created_at"],
                chunk_count=stats.get("total_chunks", 0),
            )
        )
    
    return sessions_list


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session details."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = _sessions[session_id]
    store = _get_vector_store(session_id)
    stats = store.get_stats(session_id)
    
    return SessionResponse(
        session_id=session_id,
        name=session["name"],
        description=session["description"],
        created_at=session["created_at"],
        chunk_count=stats.get("total_chunks", 0),
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its associated data."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Delete data from vector store
    store = _get_vector_store(session_id)
    store.clear(session_id)
    
    # Remove from registry
    del _sessions[session_id]
    logger.info("Deleted session: %s", session_id)
    
    return {"message": f"Session {session_id} deleted successfully"}


@router.get("/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a specific session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    store = _get_vector_store(session_id)
    return store.get_stats(session_id)