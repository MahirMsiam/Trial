"""
Pydantic models for API request and response validation.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# --- Request Models ---

class KeywordFilters(BaseModel):
    """Filter parameters for keyword search."""
    case_type: Optional[str] = None
    case_number: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    advocate: Optional[str] = None
    section: Optional[str] = None
    rule_outcome: Optional[str] = None
    year: Optional[str] = None


class KeywordSearchRequest(BaseModel):
    """Request model for keyword-based SQL search."""
    query: Optional[str] = None
    filters: Optional[KeywordFilters] = None


class SemanticSearchRequest(BaseModel):
    """Request model for FAISS semantic search."""
    query: str = Field(..., description="Query text for semantic search")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = None


class HybridSearchRequest(BaseModel):
    """Request model for hybrid semantic + keyword search."""
    query: str = Field(..., description="Query text for hybrid search")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Request model for RAG chat query."""
    query: str = Field(..., description="User question or query")
    session_id: Optional[str] = Field(default=None, description="Conversation session ID")
    filters: Optional[Dict[str, Any]] = None
    stream: bool = Field(default=False, description="Enable streaming response")
    max_chunks_per_case: int = Field(default=1, ge=1, le=5, description="Maximum chunks to return per case in sources (default: 1)")


class CrimeSearchRequest(BaseModel):
    """Request model for crime category search."""
    query: str = Field(..., description="Crime type or description to search for")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of cases to return")


class CompareCasesRequest(BaseModel):
    """Request model for comparing multiple cases."""
    case_ids: List[int] = Field(..., min_items=2, description="List of case IDs to compare (minimum 2)")


class SessionCreateRequest(BaseModel):
    """Request model for creating a new conversation session."""
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional session metadata")


# --- Response Models ---

class JudgmentResponse(BaseModel):
    """Response model for a single judgment/case."""
    model_config = ConfigDict(extra='ignore')
    
    id: int
    file_name: Optional[str] = None
    case_number: Optional[str] = None
    case_year: Optional[str] = None
    case_type: Optional[str] = None
    full_case_id: Optional[str] = None
    petitioner_name: Optional[str] = None
    respondent_name: Optional[str] = None
    judgment_date: Optional[str] = None
    judgment_outcome: Optional[str] = None
    judgment_summary: Optional[str] = None
    court_name: Optional[str] = None
    rule_type: Optional[str] = None
    rule_outcome: Optional[str] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    advocates: Optional[Dict[str, List[str]]] = None
    laws: Optional[List[str]] = None
    judges: Optional[List[str]] = None


class SearchResultResponse(BaseModel):
    """Response model for search results."""
    results: List[JudgmentResponse]
    count: int
    query: Optional[str] = None


class ChunkResponse(BaseModel):
    """Response model for a text chunk with metadata."""
    model_config = ConfigDict(extra='ignore')
    
    chunk_text: str
    similarity: Optional[float] = None
    case_id: int
    case_number: Optional[str] = None
    case_type: Optional[str] = None
    judgment_date: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    full_case_id: Optional[str] = None
    source: Optional[str] = 'semantic'


class CaseResult(BaseModel):
    """Response model for hybrid search case-level results."""
    model_config = ConfigDict(extra='ignore')
    
    case_id: int
    case_number: Optional[str] = None
    case_type: Optional[str] = None
    judgment_date: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    full_case_id: Optional[str] = None
    hybrid_score: float
    chunk_count: int
    chunks: List[ChunkResponse]


class ChatResponse(BaseModel):
    """Response model for RAG chat query."""
    response: str
    sources: List[ChunkResponse]
    query_type: str
    session_id: str


class CrimeSearchResponse(BaseModel):
    """Response model for crime category search."""
    response: str
    crime_type: str
    count: int
    cases: List[ChunkResponse]
    summary: str


class CaseComparisonResponse(BaseModel):
    """Response model for case comparison."""
    comparison: str
    cases: List[JudgmentResponse]


class CaseSummaryResponse(BaseModel):
    """Response model for case summary."""
    summary: str
    case_data: JudgmentResponse


class SessionResponse(BaseModel):
    """Response model for session information."""
    session_id: str
    created_at: str
    last_active: str
    message_count: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    database_connected: bool
    faiss_index_loaded: bool
    llm_provider: str


class StatsResponse(BaseModel):
    """Response model for database statistics."""
    total_judgments: int
    case_types: int
    total_advocates: int
    total_laws_cited: int
    case_type_breakdown: List[Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str
    detail: Optional[str] = None
    status_code: int
