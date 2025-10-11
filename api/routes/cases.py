"""
Case-specific API endpoints (summary, comparison).
"""

from fastapi import APIRouter, Depends, HTTPException
from logging_config import logger
from api.dependencies import get_rag_pipeline
from api.models import (
    CompareCasesRequest,
    CaseComparisonResponse, CaseSummaryResponse, 
    JudgmentResponse, ErrorResponse
)

router = APIRouter(prefix="/api", tags=["cases"])


@router.post("/case/{case_id}/summary", response_model=CaseSummaryResponse)
async def summarize_case(case_id: int, pipeline=Depends(get_rag_pipeline)):
    """
    Generate a summary of a specific case using LLM.
    Includes case metadata and AI-generated summary.
    """
    try:
        logger.info(f"Case summary request: case_id={case_id}")
        
        # Generate case summary
        result = pipeline.summarize_case(case_id=case_id)
        
        # Check if case was found (RAGPipeline returns None in case_data when not found)
        if not result or not result.get("case_data"):
            logger.warning(f"Case not found: case_id={case_id}")
            raise HTTPException(status_code=404, detail=f"Case with ID {case_id} not found")
        
        # Convert to response format
        response = CaseSummaryResponse(
            summary=result["summary"],
            case_data=JudgmentResponse(**result["case_data"])
        )
        
        logger.info(f"Case summary generated for case_id={case_id}")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Case summary failed for case_id={case_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Case summary failed: {str(e)}")


@router.post("/cases/compare", response_model=CaseComparisonResponse)
async def compare_cases(request: CompareCasesRequest, pipeline=Depends(get_rag_pipeline)):
    """
    Compare multiple cases using LLM analysis.
    Identifies similarities, differences, and legal precedents.
    """
    try:
        logger.info(f"Case comparison request: case_ids={request.case_ids}")
        
        if len(request.case_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 case IDs required for comparison")
        
        # Compare cases
        result = pipeline.compare_cases(case_ids=request.case_ids)
        
        # Check if we have at least 2 valid cases (RAGPipeline filters out invalid cases)
        if len(result.get("cases", [])) < 2:
            logger.warning(f"Insufficient valid cases found for comparison: {len(result.get('cases', []))} of {len(request.case_ids)}")
            raise HTTPException(status_code=400, detail="Need at least 2 valid cases to compare")
        
        # Convert cases to response format
        cases = [JudgmentResponse(**case) for case in result["cases"]]
        
        response = CaseComparisonResponse(
            comparison=result["comparison"],
            cases=cases
        )
        
        logger.info(f"Case comparison completed for {len(cases)} cases")
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Case comparison failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Case comparison failed: {str(e)}")
