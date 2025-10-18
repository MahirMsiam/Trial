"""
FastAPI main application entry point.
Bangladesh Supreme Court RAG API - REST API for legal judgment search and RAG-based question answering.
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
import traceback
from logging_config import logger
from config import CORS_ORIGINS
from api.routes.search import router as search_router
from api.routes.chat import router as chat_router
from api.routes.cases import router as cases_router
from api.routes.session import router as session_router
from api.routes.utility import router as utility_router

# Create FastAPI application
app = FastAPI(
    title="Bangladesh Supreme Court RAG API",
    description="REST API for searching legal judgments and RAG-based question answering over 8000+ Bangladesh Supreme Court cases",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Configure CORS
# WARNING: Using allow_origins=["*"] is permissive and suitable for development only.
# In production, restrict this to specific frontend domains for security.
# Example: Set CORS_ORIGINS in .env to "https://yourdomain.com,https://app.yourdomain.com"
allow_origins = [o.strip() for o in CORS_ORIGINS]
allow_credentials = not (len(allow_origins) == 1 and allow_origins[0] == "*")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router)
app.include_router(chat_router)
app.include_router(cases_router)
app.include_router(session_router)
app.include_router(utility_router)

# Enable Prometheus metrics if configured
from config import ENABLE_METRICS

if ENABLE_METRICS:
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
        Instrumentator().instrument(app).expose(app)
        logger.info("✅ Metrics enabled at /metrics")
    except ImportError:
        logger.warning("⚠️  prometheus-fastapi-instrumentator not installed. Install with: pip install prometheus-fastapi-instrumentator")
    except Exception as e:
        logger.warning(f"⚠️  Failed to enable metrics: {e}")


@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler.
    Validates configuration and initializes RAG pipeline.
    """
    logger.info("=" * 80)
    logger.info("Starting Bangladesh Supreme Court RAG API...")
    logger.info("=" * 80)
    
    try:
        # Validate configuration
        from config import validate_config
        issues = validate_config()
        if issues:
            for issue in issues:
                logger.warning(f"Config issue: {issue}")
        else:
            logger.info("Configuration validated successfully")
        
        # Initialize RAG pipeline by calling the dependency once
        # This triggers model loading at startup rather than on first request
        from api.dependencies import get_rag_pipeline
        try:
            pipeline = get_rag_pipeline()
            logger.info("RAG pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}")
            logger.warning("API will start in degraded mode - RAG features may be unavailable")
        
        logger.info("=" * 80)
        logger.info("API startup complete")
        logger.info("Documentation available at: /api/docs")
        logger.info("Health check available at: /api/health")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)
        logger.warning("API starting with configuration warnings")


@app.on_event("shutdown")
async def shutdown_event():
    """
    Application shutdown event handler.
    Cleanup resources before shutdown.
    """
    logger.info("=" * 80)
    logger.info("Shutting down Bangladesh Supreme Court RAG API...")
    logger.info("=" * 80)
    
    # Cleanup resources if needed
    # (Current implementation doesn't require explicit cleanup)
    
    logger.info("API shutdown complete")


@app.get("/")
async def root():
    """
    Root endpoint with API information and documentation links.
    """
    return {
        "message": "Bangladesh Supreme Court RAG API",
        "version": "1.0.0",
        "docs": "/api/docs",
        "redoc": "/api/redoc",
        "health": "/api/health",
        "version_info": "/api/version"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Global exception handler for HTTPException.
    Ensures consistent error response format.
    """
    logger.warning(f"HTTP {exc.status_code} error: {exc.detail} - Path: {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler for unhandled exceptions.
    Logs full traceback and returns generic error message to client.
    """
    logger.error(f"Unhandled exception in {request.method} {request.url.path}", exc_info=True)
    logger.error(f"Exception details: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please check server logs for details.",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )


if __name__ == "__main__":
    """
    Main entry point for running the API directly.
    For development with auto-reload: uvicorn api.main:app --reload
    For production: uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
    """
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
