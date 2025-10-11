"""
FastAPI dependency injection functions for resource management.
"""

from fastapi import HTTPException, Depends
from typing import Optional
from logging_config import logger
from config import DATABASE_PATH

# Module-level variables for singleton instances
_rag_pipeline = None
_initialization_error = None


def get_rag_pipeline():
    """
    Dependency function that returns a singleton instance of RAGPipeline.
    Initializes on first call and returns the same instance for all subsequent calls.
    This avoids reloading models, FAISS index, and LLM client on every request.
    """
    global _rag_pipeline, _initialization_error
    
    if _initialization_error:
        logger.error(f"RAG pipeline initialization failed previously: {_initialization_error}")
        raise HTTPException(status_code=503, detail=f"RAG system not initialized: {_initialization_error}")
    
    if _rag_pipeline is None:
        try:
            logger.info("Initializing RAG pipeline singleton...")
            from rag_pipeline import RAGPipeline
            _rag_pipeline = RAGPipeline()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            _initialization_error = str(e)
            logger.error(f"Failed to initialize RAG pipeline: {e}", exc_info=True)
            raise HTTPException(status_code=503, detail=f"RAG system not initialized: {str(e)}")
    
    return _rag_pipeline


def get_judgment_search():
    """
    Dependency function that returns a new JudgmentSearch instance.
    Uses yield to ensure database connection is closed after request completes.
    """
    try:
        from search_database import JudgmentSearch
        searcher = JudgmentSearch(db_path=DATABASE_PATH)
        logger.debug("Created JudgmentSearch instance")
        yield searcher
    except Exception as e:
        logger.error(f"Failed to create JudgmentSearch instance: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Database unavailable")
    finally:
        try:
            searcher.close()
            logger.debug("Closed JudgmentSearch database connection")
        except:
            pass


def verify_session(session_id: str, pipeline=Depends(get_rag_pipeline)):
    """
    Dependency function to validate session_id.
    Raises 404 if session not found or expired.
    """
    try:
        conv_manager = pipeline.conversation_manager
        session = conv_manager.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        logger.debug(f"Verified session: {session_id}")
        return session_id
    except ValueError as e:
        logger.warning(f"Session validation failed for {session_id}: {e}")
        raise HTTPException(status_code=404, detail="Session not found or expired")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying session {session_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Session verification failed")
