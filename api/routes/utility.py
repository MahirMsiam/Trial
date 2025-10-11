"""
Utility API endpoints for health checks and statistics.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
import os
import sqlite3
from logging_config import logger
from api.dependencies import get_rag_pipeline, get_judgment_search
from api.models import HealthResponse, StatsResponse, ErrorResponse
from config import DATABASE_PATH, FAISS_INDEX_PATH, CHUNKS_MAP_PATH, LLM_PROVIDER

router = APIRouter(prefix="/api", tags=["utility"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check endpoint.
    Verifies database, FAISS index, and RAG system availability.
    """
    try:
        logger.info("Health check requested")
        
        # Check database exists and is connectable
        database_connected = False
        if os.path.exists(DATABASE_PATH):
            try:
                conn = sqlite3.connect(DATABASE_PATH)
                conn.close()
                database_connected = True
            except:
                pass
        
        # Check FAISS index exists
        faiss_index_loaded = (
            os.path.exists(FAISS_INDEX_PATH) and 
            os.path.exists(CHUNKS_MAP_PATH)
        )
        
        # Determine overall status without initializing RAG pipeline
        if database_connected:
            if faiss_index_loaded:
                status = "healthy"
                status_code = 200
            else:
                status = "degraded"
                status_code = 200
                logger.warning("System is degraded: FAISS index not available")
        else:
            status = "unhealthy"
            status_code = 503
            logger.error("System is unhealthy: Database unavailable")
        
        response = HealthResponse(
            status=status,
            database_connected=database_connected,
            faiss_index_loaded=faiss_index_loaded,
            llm_provider=LLM_PROVIDER
        )
        
        logger.info(f"Health check completed: status={status}")
        
        # Return with appropriate status code
        return JSONResponse(
            status_code=status_code,
            content=response.model_dump()
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        # Return partial health info even on error with 503
        response = HealthResponse(
            status="unhealthy",
            database_connected=False,
            faiss_index_loaded=False,
            llm_provider=LLM_PROVIDER
        )
        return JSONResponse(
            status_code=503,
            content=response.model_dump()
        )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(searcher=Depends(get_judgment_search)):
    """
    Get database statistics.
    Returns counts and breakdowns of judgments, case types, advocates, and laws.
    """
    try:
        logger.info("Statistics requested")
        
        # Get stats from database
        stats = searcher.get_stats()
        
        # Normalize case_type_breakdown to list of dicts
        case_type_breakdown_raw = stats.get('case_type_breakdown', [])
        case_type_breakdown = [
            {"case_type": row[0], "count": row[1]} 
            for row in case_type_breakdown_raw
        ]
        
        response = StatsResponse(
            total_judgments=stats['total_judgments'],
            case_types=stats['case_types'],
            total_advocates=stats['total_advocates'],
            total_laws_cited=stats.get('total_laws_cited', stats.get('total_laws', 0)),
            case_type_breakdown=case_type_breakdown
        )
        
        logger.info(f"Statistics retrieved: {stats['total_judgments']} total judgments")
        return response
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve statistics: {str(e)}")


@router.get("/version")
async def get_version():
    """
    Get API version information.
    Returns version number and API details.
    """
    return {
        "version": "1.0.0",
        "api_name": "Bangladesh Supreme Court RAG API",
        "description": "REST API for legal judgment search and RAG-based question answering"
    }
