"""
Chat and RAG-related API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
import json
from anyio import to_thread
from logging_config import logger
from api.dependencies import get_rag_pipeline
from api.models import (
    ChatRequest,
    ChatResponse, ChunkResponse, ErrorResponse
)

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest, pipeline=Depends(get_rag_pipeline)):
    """
    Standard RAG chat query with context retrieval and LLM response.
    Returns complete response with sources and citations.
    """
    try:
        logger.info(f"Chat request: query='{request.query[:100]}...', session_id={request.session_id}")
        
        # Process query through RAG pipeline
        result = pipeline.process_query(
            query=request.query,
            session_id=request.session_id,
            filters=request.filters
        )
        
        # Convert sources to ChunkResponse format
        # Sources are case-level with nested chunks, extract chunk-level data
        raw_sources = result.get('sources', [])
        chunk_list = []
        for case in raw_sources:
            # Use max_chunks_per_case from request (default: 1)
            for ch in case.get('chunks', [])[:request.max_chunks_per_case]:
                chunk_list.append({
                    'chunk_text': ch['chunk_text'],
                    'similarity': ch.get('similarity'),
                    'case_id': ch['case_id'],
                    'case_number': ch.get('case_number'),
                    'case_type': ch.get('case_type'),
                    'judgment_date': ch.get('judgment_date'),
                    'petitioner': ch.get('petitioner'),
                    'respondent': ch.get('respondent'),
                    'full_case_id': ch.get('full_case_id'),
                    'source': ch.get('source', 'hybrid')
                })
        
        sources = [ChunkResponse(**c) for c in chunk_list]
        
        response = ChatResponse(
            response=result['response'],
            sources=sources,
            query_type=result['query_type'],
            session_id=result['session_id']
        )
        
        logger.info(f"Chat response generated for session {response.session_id} with {len(sources)} sources")
        return response
    except Exception as e:
        logger.error(f"Chat request failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat request failed: {str(e)}")


@router.post("/stream")
async def chat_stream(request: ChatRequest, pipeline=Depends(get_rag_pipeline)):
    """
    Streaming RAG chat query with Server-Sent Events.
    Yields tokens as they are generated for real-time response.
    """
    async def event_generator():
        try:
            logger.info(f"Streaming chat request: query='{request.query[:100]}...', session_id={request.session_id}")
            
            # Create a synchronous generator function
            def sync_stream():
                for chunk in pipeline.process_query_stream(
                    query=request.query,
                    session_id=request.session_id,
                    filters=request.filters
                ):
                    yield chunk
            
            # Run synchronous streaming in a thread pool to avoid blocking event loop
            chunks = await to_thread.run_sync(lambda: list(sync_stream()))
            
            # Process and yield chunks
            for chunk in chunks:
                chunk_type = chunk.get('type')
                chunk_content = chunk.get('content')
                
                if chunk_type == 'sources':
                    # Send retrieved contexts
                    yield f"data: {json.dumps({'type': 'sources', 'sources': chunk_content})}\n\n"
                
                elif chunk_type == 'token':
                    # Send each generated token
                    yield f"data: {json.dumps({'type': 'token', 'token': chunk_content})}\n\n"
                
                elif chunk_type == 'complete':
                    # chunk_content is dict with response/query_type/session_id
                    yield f"data: {json.dumps({'type': 'complete', **chunk_content})}\n\n"
                
                elif chunk_type == 'error':
                    # Send error message
                    yield f"data: {json.dumps({'type': 'error', 'error': chunk_content})}\n\n"
            
            logger.info("Streaming chat response completed")
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")
