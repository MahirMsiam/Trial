"""
Search-related API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import List
from logging_config import logger
from api.dependencies import get_rag_pipeline, get_judgment_search, get_retriever
from api.models import (
    KeywordSearchRequest, SemanticSearchRequest, HybridSearchRequest, CrimeSearchRequest,
    SearchResultResponse, ChunkResponse, CrimeSearchResponse, JudgmentResponse, ErrorResponse, CaseResult
)

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/keyword", response_model=SearchResultResponse)
async def keyword_search(request: KeywordSearchRequest, searcher=Depends(get_judgment_search)):
    """
    Keyword-based SQL search using filters and optional free-text query.
    Searches through case metadata and returns matching judgments.
    Works in degraded mode even when RAG pipeline is unavailable.
    """
    try:
        logger.info(f"Keyword search request: query={request.query}, filters={request.filters}")
        
        # If free-text query provided, use pure SQL LIKE query for degraded mode compatibility
        if request.query:
            try:
                # Try to use retriever for better full-text search if available
                pipeline = get_rag_pipeline()
                retriever = pipeline.retriever
                filter_dict = request.filters.model_dump(exclude_none=True) if request.filters else {}
                kw_results = retriever.retrieve_keyword(query=request.query, filters=filter_dict)
                
                # Convert keyword results to judgment summaries
                # Group by case_id to avoid duplicates
                seen_cases = {}
                for r in kw_results:
                    case_id = r["case_id"]
                    if case_id not in seen_cases:
                        seen_cases[case_id] = JudgmentResponse(
                            id=case_id,
                            file_name=None,
                            case_number=r.get("case_number"),
                            case_type=r.get("case_type"),
                            full_case_id=r.get("full_case_id"),
                            judgment_date=r.get("judgment_date"),
                            petitioner_name=r.get("petitioner"),
                            respondent_name=r.get("respondent")
                        )
                judgments = list(seen_cases.values())
            except HTTPException as e:
                # Pipeline unavailable - fallback to using JudgmentSearch with filter-only
                logger.warning(f"RAG pipeline unavailable for keyword search, using JudgmentSearch fallback: {e.detail}")
                
                # Use JudgmentSearch for filtering, then filter results by query text in Python
                filter_dict = request.filters.model_dump(exclude_none=True) if request.filters else {}
                results = searcher.search(**filter_dict)
                
                # If query provided, filter results by matching text in full_text field
                if request.query and results:
                    query_lower = request.query.lower()
                    filtered_results = []
                    for result in results:
                        # Check if query appears in full_text (case-insensitive)
                        full_text = result.get('full_text', '').lower()
                        if query_lower in full_text:
                            filtered_results.append(result)
                    results = filtered_results
                
                judgments = [JudgmentResponse(**result) for result in results]
        else:
            # Use structured filter-only search (no pipeline required)
            filter_dict = request.filters.model_dump(exclude_none=True) if request.filters else {}
            results = searcher.search(**filter_dict)
            judgments = [JudgmentResponse(**result) for result in results]
        
        logger.info(f"Keyword search returned {len(judgments)} results")
        return SearchResultResponse(
            results=judgments,
            count=len(judgments),
            query=request.query
        )
    except Exception as e:
        logger.error(f"Keyword search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/semantic", response_model=List[ChunkResponse])
async def semantic_search(request: SemanticSearchRequest, retriever=Depends(get_retriever)):
    """
    FAISS semantic search using sentence embeddings.
    Returns relevant text chunks with similarity scores.
    """
    try:
        logger.info(f"Semantic search request: query='{request.query}', top_k={request.top_k}")
        
        # Check if FAISS index is available
        if getattr(retriever, 'index', None) is None or not getattr(retriever, 'chunks_map', {}):
            logger.warning("FAISS index not available for semantic search")
            raise HTTPException(
                status_code=503, 
                detail="Semantic search unavailable. Run create_index.py"
            )
        
        # Perform semantic search
        chunks = retriever.retrieve_semantic(query=request.query, top_k=request.top_k)
        
        # Convert to response format
        chunk_responses = [ChunkResponse(**chunk) for chunk in chunks]
        
        logger.info(f"Semantic search returned {len(chunk_responses)} chunks")
        return chunk_responses
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Semantic search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.post("/hybrid", response_model=List[CaseResult])
async def hybrid_search(request: HybridSearchRequest, retriever=Depends(get_retriever)):
    """
    Hybrid semantic + keyword search combining FAISS and SQL.
    Returns case-level results with nested chunks from both retrieval methods.
    """
    try:
        logger.info(f"Hybrid search request: query='{request.query}', top_k={request.top_k}")
        
        # Perform hybrid search
        results = retriever.hybrid_retrieve(
            query=request.query, 
            top_k=request.top_k, 
            filters=request.filters
        )
        
        # Map to CaseResult to enforce API contract
        case_results = [CaseResult(**r) for r in results]
        
        logger.info(f"Hybrid search returned {len(case_results)} case-level results")
        return case_results
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Hybrid search failed: {str(e)}")


@router.post("/crime", response_model=CrimeSearchResponse)
async def crime_search(request: CrimeSearchRequest, pipeline=Depends(get_rag_pipeline)):
    """
    Search for cases by crime category.
    Uses crime classification to find relevant cases.
    """
    try:
        logger.info(f"Crime search request: query='{request.query}', limit={request.limit}")
        
        # Perform crime search
        result = pipeline.search_crime_cases(query=request.query, limit=request.limit)
        
        # Convert cases to ChunkResponse format
        cases = [ChunkResponse(**c) for c in result.get('cases', [])]
        
        response = CrimeSearchResponse(
            response=result['response'],
            crime_type=result['crime_type'],
            count=len(cases),
            cases=cases,
            summary=result['summary']
        )
        
        logger.info(f"Crime search returned {response.count} cases for crime type: {response.crime_type}")
        return response
    except Exception as e:
        logger.error(f"Crime search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Crime search failed: {str(e)}")
