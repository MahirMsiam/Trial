"""
Session management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from logging_config import logger
from api.dependencies import get_rag_pipeline, verify_session
from api.models import SessionCreateRequest, SessionResponse, ErrorResponse

router = APIRouter(prefix="/api/session", tags=["session"])


@router.post("", response_model=SessionResponse, status_code=201)
async def create_session(request: SessionCreateRequest, pipeline=Depends(get_rag_pipeline)):
    """
    Create a new conversation session.
    Returns session ID and metadata for tracking conversation history.
    """
    try:
        logger.info(f"Creating new session with metadata: {request.metadata}")
        
        conv_manager = pipeline.conversation_manager
        
        # Create new session
        session_id = conv_manager.create_session(metadata=request.metadata)
        
        # Get session details
        session = conv_manager.get_session(session_id)
        
        response = SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            last_active=session.last_active.isoformat(),
            message_count=len(session.history)
        )
        
        logger.info(f"Created session: {session_id}")
        return response
    except Exception as e:
        logger.error(f"Failed to create session: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str = Depends(verify_session), pipeline=Depends(get_rag_pipeline)):
    """
    Get information about a specific session.
    Returns session metadata and activity information.
    """
    try:
        logger.info(f"Getting session info: {session_id}")
        
        conv_manager = pipeline.conversation_manager
        
        # Session already verified by dependency
        session = conv_manager.get_session(session_id)
        
        response = SessionResponse(
            session_id=session.session_id,
            created_at=session.created_at.isoformat(),
            last_active=session.last_active.isoformat(),
            message_count=len(session.history)
        )
        
        logger.info(f"Retrieved session info for: {session_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session: {str(e)}")


@router.get("/{session_id}/history")
async def get_session_history(
    session_id: str = Depends(verify_session),
    max_turns: int = None,
    pipeline=Depends(get_rag_pipeline)
) -> List[Dict[str, Any]]:
    """
    Get conversation history for a session.
    Returns list of messages with roles, content, and timestamps.
    """
    try:
        logger.info(f"Getting session history: {session_id}, max_turns={max_turns}")
        
        conv_manager = pipeline.conversation_manager

        # Verify dependency already ensured session exists; just fetch history
        history = conv_manager.get_context_for_prompt(session_id=session_id, max_turns=max_turns)
        
        logger.info(f"Retrieved {len(history)} messages for session {session_id}")
        return history
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session history for {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve session history: {str(e)}")


@router.delete("/{session_id}", status_code=204)
async def delete_session(session_id: str, pipeline=Depends(get_rag_pipeline)):
    """
    Clear/delete a conversation session.
    Removes all messages and session data.
    """
    try:
        logger.info(f"Deleting session: {session_id}")
        
        conv_manager = pipeline.conversation_manager
        
        # Clear session
        conv_manager.clear_session(session_id=session_id)
        
        logger.info(f"Deleted session: {session_id}")
        return None
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
