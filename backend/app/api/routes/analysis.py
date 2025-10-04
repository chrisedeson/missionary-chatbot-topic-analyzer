"""
Analysis API routes for topic discovery and classification.
Updated implementation with hybrid analysis service integration.
"""

from fastapi import APIRouter, HTTPException, Depends, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import logging

from app.core.auth import require_developer_auth
from app.services.analysis import analysis_service
from app.services.sse import create_sse_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analysis", tags=["analysis"])

class AnalysisRequest(BaseModel):
    mode: str = "all"  # "sample" or "all"
    sample_size: Optional[int] = None

class AnalysisResponse(BaseModel):
    run_id: str
    status: str
    message: str

@router.post("/start", response_model=AnalysisResponse)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    user=Depends(require_developer_auth)
):
    """
    Start a new topic analysis run.
    Requires developer authentication.
    """
    try:
        logger.info(f"Starting analysis: mode={request.mode}, sample_size={request.sample_size}")
        
        # Validate request
        if request.mode not in ["sample", "all"]:
            raise HTTPException(status_code=400, detail="Mode must be 'sample' or 'all'")
        
        if request.mode == "sample" and not request.sample_size:
            raise HTTPException(status_code=400, detail="Sample size required for sample mode")
        
        # Start analysis
        run_id = await analysis_service.start_analysis(
            mode=request.mode,
            sample_size=request.sample_size
        )
        
        return AnalysisResponse(
            run_id=run_id,
            status="started",
            message=f"Analysis started with mode: {request.mode}"
        )
        
    except Exception as e:
        logger.error(f"Error starting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/runs")
async def get_analysis_runs(
    limit: int = 20,
    user=Depends(require_developer_auth)
):
    """
    Get history of analysis runs.
    Requires developer authentication.
    """
    try:
        runs = await analysis_service.get_analysis_history(limit=limit)
        return {"runs": runs}
        
    except Exception as e:
        logger.error(f"Error getting analysis runs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/runs/{run_id}/status")
async def get_run_status(run_id: str):
    """
    Get the status of a specific analysis run.
    Public endpoint for status checking.
    """
    try:
        status = analysis_service.get_run_status(run_id)
        
        if not status:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        
        return {
            "run_id": run_id,
            "status": status["status"],
            "progress": status.get("progress", {}),
            "started_at": status["started_at"],
            "completed_at": status.get("completed_at"),
            "error": status.get("error")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting run status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/runs/{run_id}/topics")
async def get_run_topics(run_id: str):
    """
    Get topics discovered in a specific analysis run.
    Public endpoint for viewing results.
    """
    try:
        # Check if run exists and is completed
        run_status = analysis_service.get_run_status(run_id)
        
        if not run_status:
            raise HTTPException(status_code=404, detail="Analysis run not found")
        
        if run_status["status"] != "completed":
            raise HTTPException(status_code=400, detail="Analysis run not completed")
        
        # Get topics for this run
        topics = await analysis_service.get_topics(run_id)
        
        return {"topics": topics}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting run topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/runs/{run_id}/export")
async def export_run_results(
    run_id: str,
    user=Depends(require_developer_auth)
):
    """
    Export complete results for an analysis run.
    Requires developer authentication.
    """
    try:
        results = await analysis_service.export_results(run_id)
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error exporting results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{run_id}/progress")
async def stream_analysis_progress(run_id: str, request: Request):
    """
    Stream real-time progress updates for an analysis run using Server-Sent Events.
    Public endpoint for progress monitoring.
    """
    try:
        return create_sse_response(run_id, request)
        
    except Exception as e:
        logger.error(f"Error creating SSE stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy endpoints for backward compatibility
@router.post("/run")
async def legacy_run_analysis(
    mode: str = "all",
    sample_size: Optional[int] = None,
    user=Depends(require_developer_auth)
):
    """
    Legacy endpoint - redirects to /start
    """
    request = AnalysisRequest(mode=mode, sample_size=sample_size)
    return await start_analysis(request, BackgroundTasks(), user)

@router.get("/status/{run_id}")
async def legacy_get_status(run_id: str):
    """
    Legacy endpoint - redirects to /runs/{run_id}/status
    """
    return await get_run_status(run_id)

@router.get("/history")
async def legacy_get_history(
    limit: int = 20,
    user=Depends(require_developer_auth)
):
    """
    Legacy endpoint - redirects to /runs
    """
    return await get_analysis_runs(limit, user)

@router.delete("/clear")
async def clear_analysis_data(
    confirm: bool = False,
    user=Depends(require_developer_auth)
):
    """
    Clear all analysis data (topics and runs).
    Requires developer authentication and confirmation.
    """
    if not confirm:
        raise HTTPException(
            status_code=400, 
            detail="Must set confirm=true to clear analysis data"
        )
    
    try:
        # This would clear all analysis data from database
        logger.warning("Analysis data clear requested - not implemented yet")
        
        return {
            "message": "Analysis data clear not implemented yet",
            "status": "pending"
        }
        
    except Exception as e:
        logger.error(f"Error clearing analysis data: {e}")
        raise HTTPException(status_code=500, detail=str(e))