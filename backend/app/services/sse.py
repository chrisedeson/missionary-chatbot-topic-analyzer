"""
Server-Sent Events (SSE) handlers for real-time progress updates.
"""

import asyncio
import json
import logging
from typing import Dict, AsyncGenerator
from fastapi import Request
from fastapi.responses import StreamingResponse

from app.services.analysis import analysis_service

logger = logging.getLogger(__name__)

class SSEManager:
    """
    Manager for Server-Sent Events connections.
    """
    
    def __init__(self):
        self.connections: Dict[str, Dict] = {}
    
    async def add_connection(self, run_id: str, connection_id: str):
        """Add a new SSE connection for a run."""
        if run_id not in self.connections:
            self.connections[run_id] = {}
        
        self.connections[run_id][connection_id] = {
            'connected': True,
            'last_event': None
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
                except Exception as e:
                    logger.error(f"Error updating connection {connection_id}: {e}")
                    self.remove_connection(run_id, connection_id)

# Global SSE manager
sse_manager = SSEManager()

async def progress_stream(run_id: str, request: Request) -> AsyncGenerator[str, None]:
    """
    Generate Server-Sent Events stream for analysis progress.
    """
    import uuid
    connection_id = str(uuid.uuid4())
    
    try:
        # Add this connection to the manager
        await sse_manager.add_connection(run_id, connection_id)
        
        logger.info(f"Starting SSE stream for run {run_id}, connection {connection_id}")
        
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'run_id': run_id, 'timestamp': str(asyncio.get_event_loop().time())})}\n\n"
        
        # Check if run exists
        run_status = analysis_service.get_run_status(run_id)
        if not run_status:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis run not found'})}\n\n"
            return
        
        # Send current status
        yield f"data: {json.dumps({'type': 'status', 'status': run_status['status']})}\n\n"
        
        # If run is already completed, send final state and exit
        if run_status['status'] in ['completed', 'failed']:
            if run_status['status'] == 'completed':
                yield f"data: {json.dumps({'type': 'complete', 'summary': run_status.get('results_summary', {})})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': run_status.get('error', 'Analysis failed')})}\n\n"
            return
        
        # Stream progress updates
        last_progress = None
        
        while not await request.is_disconnected():
            current_progress = analysis_service.get_run_progress(run_id)
            current_status = analysis_service.get_run_status(run_id)
            
            if not current_status:
                yield f"data: {json.dumps({'type': 'error', 'message': 'Analysis run no longer exists'})}\n\n"
                break
            
            # Send progress update if it changed
            if current_progress and current_progress != last_progress:
                progress_event = {
                    'type': 'progress',
                    'stage': current_progress['stage'],
                    'progress': current_progress['progress'],
                    'message': current_progress['message'],
                    'timestamp': current_progress.get('timestamp', str(asyncio.get_event_loop().time()))
                }
                
                yield f"data: {json.dumps(progress_event)}\n\n"
                last_progress = current_progress
                
                # Broadcast to other connections
                await sse_manager.broadcast_progress(run_id, progress_event)
            
            # Check if run completed
            if current_status['status'] == 'completed':
                completion_event = {
                    'type': 'complete',
                    'summary': current_status.get('results_summary', {}),
                    'timestamp': str(asyncio.get_event_loop().time())
                }
                yield f"data: {json.dumps(completion_event)}\n\n"
                
                # Broadcast completion
                await sse_manager.broadcast_progress(run_id, completion_event)
                break
            elif current_status['status'] == 'failed':
                error_event = {
                    'type': 'error',
                    'message': current_status.get('error', 'Analysis failed'),
                    'timestamp': str(asyncio.get_event_loop().time())
                }
                yield f"data: {json.dumps(error_event)}\n\n"
                
                # Broadcast error
                await sse_manager.broadcast_progress(run_id, error_event)
                break
            
            # Wait before next check
            await asyncio.sleep(1)
        
    except asyncio.CancelledError:
        logger.info(f"SSE stream cancelled for run {run_id}, connection {connection_id}")
    except Exception as e:
        logger.error(f"Error in SSE stream for run {run_id}: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': f'Stream error: {str(e)}'})}\n\n"
    finally:
        # Clean up connection
        sse_manager.remove_connection(run_id, connection_id)
        logger.info(f"SSE stream ended for run {run_id}, connection {connection_id}")

def create_sse_response(run_id: str, request: Request) -> StreamingResponse:
    """
    Create a StreamingResponse for Server-Sent Events.
    """
    return StreamingResponse(
        progress_stream(run_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )