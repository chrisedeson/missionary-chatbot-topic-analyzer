"""
Server-Sent Events (SSE) handlers for real-time progress updates.

This module provides SSE streaming for analysis progress tracking, allowing
the frontend to receive real-time updates during long-running analysis operations.
"""

import asyncio
import json
import logging
from typing import Dict, AsyncGenerator, Optional
from datetime import datetime
from fastapi import Request
from fastapi.responses import StreamingResponse
from prisma import Prisma

from app.core.database import get_db

logger = logging.getLogger(__name__)


class SSEManager:
    """
    Manager for Server-Sent Events connections.
    
    Tracks active SSE connections and broadcasts progress updates to all
    connected clients for a given analysis run.
    """
    
    def __init__(self):
        self.connections: Dict[str, Dict] = {}
    
    async def add_connection(self, run_id: str, connection_id: str):
        """Add a new SSE connection for a run."""
        if run_id not in self.connections:
            self.connections[run_id] = {}
        
        self.connections[run_id][connection_id] = {
            'connected': True,
            'last_event': None,
            'connected_at': datetime.utcnow()
        }
        
        logger.info(f"Added SSE connection {connection_id} for run {run_id}")
    
    def remove_connection(self, run_id: str, connection_id: str):
        """Remove an SSE connection."""
        if run_id in self.connections and connection_id in self.connections[run_id]:
            del self.connections[run_id][connection_id]
            
            # Clean up empty run connections
            if not self.connections[run_id]:
                del self.connections[run_id]
            
            logger.info(f"Removed SSE connection {connection_id} for run {run_id}")
    
    async def broadcast_progress(self, run_id: str, progress_data: Dict):
        """Broadcast progress update to all connections for a run."""
        if run_id in self.connections:
            for connection_id in list(self.connections[run_id].keys()):
                try:
                    connection = self.connections[run_id][connection_id]
                    connection['last_event'] = progress_data
                    connection['last_update'] = datetime.utcnow()
                except Exception as e:
                    logger.error(f"Error updating connection {connection_id}: {e}")
                    self.remove_connection(run_id, connection_id)
    
    def get_connection_count(self, run_id: str) -> int:
        """Get number of active connections for a run."""
        return len(self.connections.get(run_id, {}))


# Global SSE manager
sse_manager = SSEManager()


async def get_run_status(db: Prisma, run_id: str) -> Optional[Dict]:
    """Get analysis run status from database."""
    try:
        run = await db.analysisrun.find_unique(
            where={"id": run_id}
        )
        
        if not run:
            return None
        
        return {
            "status": run.status,
            "progress": run.progress,
            "message": run.message,
            "error": run.error,
            "results_summary": {
                "total_questions": run.totalQuestions,
                "similar_questions": run.similarQuestions,
                "new_topics_discovered": run.newTopicsDiscovered
            } if run.status == "completed" else None
        }
    except Exception as e:
        logger.error(f"Error fetching run status: {e}")
        return None


async def progress_stream(run_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events stream for analysis progress.
    
    Streams real-time progress updates from the database to connected clients.
    """
    import uuid
    connection_id = str(uuid.uuid4())
    db = await get_db()
    
    try:
        # Add this connection to the manager
        await sse_manager.add_connection(run_id, connection_id)
        
        logger.info(f"Starting SSE stream for run {run_id}, connection {connection_id}")
        
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id, 'connection_id': connection_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
        
        # Check if run exists
        run_status = await get_run_status(db, run_id)
        if not run_status:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis run not found'})}\n\n"
            return
        
        # Send current status
        yield f"data: {json.dumps({'type': 'status', 'status': run_status['status'], 'progress': run_status['progress']})}\n\n"
        
        # If run is already completed, send final state and exit
        if run_status['status'] in ['completed', 'failed']:
            if run_status['status'] == 'completed':
                yield f"data: {json.dumps({'type': 'complete', 'summary': run_status.get('results_summary', {}), 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': run_status.get('error', 'Analysis failed'), 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            return
        
        # Stream progress updates
        last_progress = run_status['progress']
        last_status = run_status['status']
        poll_interval = 1  # seconds
        
        while not await request.is_disconnected():
            # Fetch current status from database
            current_status = await get_run_status(db, run_id)
            
            if not current_status:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis run no longer exists', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
                break
            
            # Send progress update if it changed
            if (current_status['progress'] != last_progress or 
                current_status['status'] != last_status or
                current_status['message'] != run_status.get('message')):
                
                # Derive stage from status and message
                stage = current_status['status']
                if current_status['message']:
                    # Try to extract stage from message patterns
                    msg = current_status['message'].lower()
                    if 'embedding' in msg or 'generating embeddings' in msg:
                        stage = 'generating_embeddings'
                    elif 'classif' in msg or 'similarity' in msg:
                        stage = 'classifying_questions'
                    elif 'cluster' in msg or 'discover' in msg:
                        stage = 'discovering_topics'
                    elif 'saving' in msg or 'save' in msg:
                        stage = 'saving_results'
                    elif current_status['status'] == 'running':
                        stage = 'running'
                
                progress_event = {
                    'type': 'progress',
                    'status': current_status['status'],
                    'stage': stage,
                    'progress': current_status['progress'],
                    'message': current_status['message'],
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                yield f"data: {json.dumps(progress_event)}\n\n"
                
                last_progress = current_status['progress']
                last_status = current_status['status']
                
                # Broadcast to other connections
                await sse_manager.broadcast_progress(run_id, progress_event)
            
            # Check if run completed
            if current_status['status'] == 'completed':
                completion_event = {
                    'type': 'complete',
                    'summary': current_status.get('results_summary', {}),
                    'timestamp': datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(completion_event)}\n\n"
                
                # Broadcast completion
                await sse_manager.broadcast_progress(run_id, completion_event)
                logger.info(f"Analysis run {run_id} completed successfully")
                break
                
            elif current_status['status'] == 'failed':
                error_event = {
                    'type': 'error',
                    'message': current_status.get('error', 'Analysis failed'),
                    'timestamp': datetime.utcnow().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                
                # Broadcast error
                await sse_manager.broadcast_progress(run_id, error_event)
                logger.error(f"Analysis run {run_id} failed: {current_status.get('error')}")
                break
            
            # Send keepalive comment every 30 seconds to prevent timeout
            if poll_interval % 30 == 0:
                yield f": keepalive\n\n"
            
            # Wait before next check
            await asyncio.sleep(poll_interval)
        
    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled for run {run_id}, connection {connection_id}")
    except Exception as e:
        logger.error(f"Error in SSE stream for run {run_id}: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
    finally:
        # Clean up connection
        sse_manager.remove_connection(run_id, connection_id)
        logger.info(f"SSE stream ended for run {run_id}, connection {connection_id} (total connections: {sse_manager.get_connection_count(run_id)})")


def create_sse_response(run_id: str, request: Request) -> StreamingResponse:
    """
    Create a StreamingResponse for Server-Sent Events.
    
    Sets up proper headers for SSE streaming and CORS support.
    """
    return StreamingResponse(
        progress_stream(run_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control, Content-Type",
            "Access-Control-Allow-Methods": "GET, OPTIONS"
        }
    )


async def update_run_progress(
    db: Prisma,
    run_id: str,
    progress: int,
    message: str,
    status: Optional[str] = None
):
    """
    Helper function to update analysis run progress.
    
    Can be called from analysis services to update progress in real-time.
    """
    try:
        update_data = {
            "progress": progress,
            "message": message
        }
        
        if status:
            update_data["status"] = status
        
        await db.analysisrun.update(
            where={"id": run_id},
            data=update_data
        )
        
        # Broadcast to SSE connections
        await sse_manager.broadcast_progress(run_id, {
            "type": "progress",
            "progress": progress,
            "message": message,
            "status": status,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.debug(f"Updated progress for run {run_id}: {progress}% - {message}")
        
    except Exception as e:
        logger.error(f"Error updating run progress: {e}")