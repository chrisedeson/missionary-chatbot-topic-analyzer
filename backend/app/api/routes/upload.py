"""
Upload API Routes

Handles file uploads and data processing pipeline.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse
import uuid
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any
import json
from datetime import datetime

from app.core.config import settings
from app.core.auth import get_current_developer
from app.utils.sse_manager import sse_manager
from app.services.data_processing import data_processing_service

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory storage for upload status (use Redis in production)
upload_status: Dict[str, Dict] = {}

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    developer=Depends(get_current_developer)
):
    """Handle file upload"""
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Generate unique ID for this upload
        upload_id = str(uuid.uuid4())
        
        # Read file content
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="File is empty")
        
        # Store file info
        file_info = {
            "upload_id": upload_id,
            "filename": file.filename,
            "size": len(content),
            "status": "uploaded",
            "content": content,  # In production, store in proper file storage
            "uploaded_at": datetime.now().isoformat()
        }
        
        upload_status[upload_id] = file_info
        
        logger.info(f"File uploaded successfully: {file.filename} ({len(content)} bytes)")
        
        return JSONResponse({
            "upload_id": upload_id,
            "filename": file.filename,
            "size": len(content),
            "status": "uploaded"
        })
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/process/{upload_id}")
async def process_file(
    upload_id: str, 
    background_tasks: BackgroundTasks,
    developer=Depends(get_current_developer)
):
    """Process uploaded file and extract questions"""
    try:
        # Get uploaded file
        if upload_id not in upload_status:
            raise HTTPException(status_code=404, detail="Upload not found")
        
        file_info = upload_status[upload_id]
        if file_info["status"] != "uploaded":
            raise HTTPException(status_code=400, detail="File not ready for processing")
        
        # Generate processing ID
        processing_id = str(uuid.uuid4())
        
        # Create progress callback for SSE updates
        async def progress_callback(proc_id: str, stage: str, progress: int, message: str):
            """Send progress updates via SSE"""
            update_data = {
                "processing_id": proc_id,
                "stage": stage,
                "progress": progress,
                "message": message,
                "timestamp": datetime.now().isoformat()
            }
            await sse_manager.send_to_client(proc_id, json.dumps(update_data))
        
        # Start processing in background
        async def run_processing():
            try:
                await data_processing_service.process_questions_file(
                    file_content=file_info["content"],
                    filename=file_info["filename"],
                    processing_id=processing_id,
                    progress_callback=progress_callback
                )
            except Exception as e:
                logger.error(f"Background processing failed: {e}")
                await progress_callback(processing_id, "error", 0, f"Processing failed: {str(e)}")
        
        # Start background task
        background_tasks.add_task(run_processing)
        
        logger.info(f"Started processing for upload {upload_id} with processing ID {processing_id}")
        
        return JSONResponse({
            "processing_id": processing_id,
            "upload_id": upload_id,
            "status": "processing",
            "message": "Data processing started"
        })
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{processing_id}")
async def get_processing_status(
    processing_id: str,
    developer=Depends(get_current_developer)
):
    """Get processing status and results"""
    status = data_processing_service.get_processing_status(processing_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Processing job not found")
    
    return status

@router.get("/progress/{processing_id}")
async def get_processing_progress(processing_id: str):
    """Get real-time processing progress via Server-Sent Events"""
    
    # Note: We don't require auth here since processing_id is a UUID
    # and this is only for progress updates, not sensitive data
    
    async def event_stream():
        """Generate SSE events for processing progress"""
        client_queue = None
        try:
            # Register client for updates
            client_queue = sse_manager.add_client(processing_id)
            
            # Send initial connection event
            initial_message = json.dumps({
                'type': 'connected', 
                'processing_id': processing_id,
                'timestamp': datetime.now().isoformat()
            })
            yield f"data: {initial_message}\n\n"
            
            # Stream updates with timeout
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                    yield f"data: {message}\n\n"
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    heartbeat = json.dumps({
                        'type': 'heartbeat',
                        'timestamp': datetime.now().isoformat()
                    })
                    yield f"data: {heartbeat}\n\n"
                except asyncio.CancelledError:
                    logger.info(f"SSE stream cancelled for processing {processing_id}")
                    break
                    
        except Exception as e:
            logger.error(f"SSE stream error for processing {processing_id}: {e}")
            error_message = json.dumps({
                'type': 'error', 
                'message': f"Stream error: {str(e)}",
                'timestamp': datetime.now().isoformat()
            })
            yield f"data: {error_message}\n\n"
        finally:
            if client_queue:
                sse_manager.remove_client(processing_id, client_queue)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

@router.get("/history")
async def get_processing_history(developer=Depends(get_current_developer)):
    """Get history of all processing jobs"""
    return {
        "history": data_processing_service.get_all_processing_history()
    }

@router.get("/uploads")
async def get_uploaded_files(developer=Depends(get_current_developer)):
    """Get list of uploaded files"""
    return {
        "uploads": list(upload_status.values())
    }
