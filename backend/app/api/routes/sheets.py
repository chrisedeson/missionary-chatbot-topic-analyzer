"""
Google Sheets API routes for data synchronization.
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Dict, Any
import logging
from datetime import datetime

from app.core.auth import require_developer_auth
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["sheets"])

class ConnectionTestResponse(BaseModel):
    status: str
    message: str
    last_test: str
    sheets_accessible: bool

@router.get("/test-connection", response_model=ConnectionTestResponse)
async def test_sheets_connection(user=Depends(require_developer_auth)):
    """
    Test connection to Google Sheets API.
    Requires developer authentication.
    """
    try:
        # For now, return a mock response since we don't have real Google Sheets integration yet
        # TODO: Implement actual Google Sheets API test
        
        logger.info("Testing Google Sheets connection")
        
        # Mock successful connection
        return ConnectionTestResponse(
            status="success",
            message="Google Sheets connection test successful",
            last_test=datetime.now().isoformat(),
            sheets_accessible=True
        )
        
    except Exception as e:
        logger.error(f"Error testing sheets connection: {e}")
        return ConnectionTestResponse(
            status="error", 
            message=f"Connection failed: {str(e)}",
            last_test=datetime.now().isoformat(),
            sheets_accessible=False
        )

@router.get("/status")
async def get_sheets_status():
    """
    Get Google Sheets integration status.
    Public endpoint.
    """
    try:
        return {
            "integration_status": "configured",
            "sheets_id": settings.QUESTIONS_SHEET_ID,
            "last_sync": None,  # TODO: Implement actual sync tracking
            "total_questions": 0,  # TODO: Get from actual sheets
            "sync_enabled": True
        }
        
    except Exception as e:
        logger.error(f"Error getting sheets status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/sync")
async def sync_from_sheets(user=Depends(require_developer_auth)):
    """
    Trigger synchronization from Google Sheets.
    Requires developer authentication.
    """
    try:
        # TODO: Implement actual Google Sheets sync
        logger.info("Starting Google Sheets sync")
        
        return {
            "status": "started",
            "message": "Sheets synchronization started",
            "sync_id": "mock_sync_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
    except Exception as e:
        logger.error(f"Error starting sheets sync: {e}")
        raise HTTPException(status_code=500, detail=str(e))