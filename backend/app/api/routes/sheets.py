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
from app.services.google_sheets import google_sheets_service

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
        # Test connection to the Questions sheet
        test_result = google_sheets_service.test_connection(settings.QUESTIONS_SHEET_ID)
        
        logger.info("Testing Google Sheets connection")
        
        return ConnectionTestResponse(
            status=test_result["status"],
            message=test_result["message"],
            last_test=datetime.now().isoformat(),
            sheets_accessible=test_result["accessible"]
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
        # Get sheet structure to validate columns
        structure = google_sheets_service.get_sheet_structure(settings.QUESTIONS_SHEET_ID)
        
        return {
            "integration_status": "configured",
            "sheets_id": settings.QUESTIONS_SHEET_ID,
            "last_sync": None,  # TODO: Implement actual sync tracking
            "total_questions": structure.get("total_rows", 0) - 1,  # Subtract header row
            "sync_enabled": True,
            "sheet_structure": structure
        }
        
    except Exception as e:
        logger.error(f"Error getting sheets status: {e}")
        return {
            "integration_status": "error",
            "sheets_id": settings.QUESTIONS_SHEET_ID,
            "error": str(e),
            "sync_enabled": False
        }

@router.get("/structure")
async def get_sheet_structure(user=Depends(require_developer_auth)):
    """
    Get Google Sheets structure for column validation.
    Requires developer authentication.
    """
    try:
        structure = google_sheets_service.get_sheet_structure(settings.QUESTIONS_SHEET_ID)
        
        # Validate columns
        expected_columns = ['Time Stamp', 'Country', 'User Language', 'State', 'Question']
        is_valid, column_mapping, missing_columns = google_sheets_service.validate_columns(
            structure['columns'], 
            expected_columns
        )
        
        suggestions = google_sheets_service.suggest_column_fixes(
            structure['columns'], 
            expected_columns
        ) if not is_valid else {}
        
        return {
            "status": "success",
            "structure": structure,
            "validation": {
                "is_valid": is_valid,
                "column_mapping": column_mapping,
                "missing_columns": missing_columns,
                "suggestions": suggestions,
                "expected_columns": expected_columns
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting sheet structure: {e}")
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