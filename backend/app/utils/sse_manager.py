"""
Server-Sent Events (SSE) Manager

Handles real-time progress updates for data processing and analysis.
This is a lightweight SSE manager for non-database-backed streaming.
For analysis runs, use the database-backed SSE in app/services/sse.py
"""

import asyncio
import logging
from typing import Dict, Set, AsyncGenerator, Optional
from collections import defaultdict
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class SSEManager:
    """Manages Server-Sent Events for real-time updates"""
    
    def __init__(self):
        self.clients: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self.active_streams: Set[str] = set()
        self._lock = asyncio.Lock()
    
    async def add_client(self, processing_id: str) -> asyncio.Queue:
        """Add a client to receive updates for a processing job"""
        async with self._lock:
            client_queue = asyncio.Queue(maxsize=100)  # Prevent memory issues
            self.clients[processing_id].add(client_queue)
            self.active_streams.add(processing_id)
            logger.info(f"Added client for processing {processing_id} (total clients: {len(self.clients[processing_id])})")
            return client_queue
    
    async def remove_client(self, processing_id: str, client_queue: Optional[asyncio.Queue] = None):
        """Remove a client from receiving updates"""
        async with self._lock:
            if client_queue and processing_id in self.clients:
                self.clients[processing_id].discard(client_queue)
            
            # Clean up if no clients remain
            if not self.clients.get(processing_id):
                self.active_streams.discard(processing_id)
                if processing_id in self.clients:
                    del self.clients[processing_id]
                logger.info(f"Removed all clients for processing {processing_id}")
    
    async def send_to_clients(self, processing_id: str, message: Dict):
        """Send a message to all clients listening to a processing job"""
        if processing_id not in self.clients:
            logger.debug(f"No clients for processing {processing_id}")
            return
        
        # Add timestamp to message
        if 'timestamp' not in message:
            message['timestamp'] = datetime.utcnow().isoformat()
        
        message_str = json.dumps(message)
        
        # Send to all clients for this processing job
        dead_queues = []
        for client_queue in self.clients[processing_id].copy():
            try:
                # Non-blocking put with timeout
                await asyncio.wait_for(
                    client_queue.put(message_str), 
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                logger.warning(f"Client queue full for processing {processing_id}, skipping message")
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                dead_queues.append(client_queue)
        
        # Remove dead queues
        if dead_queues:
            async with self._lock:
                for queue in dead_queues:
                    self.clients[processing_id].discard(queue)
    
    async def send_progress(
        self, 
        processing_id: str, 
        progress: int, 
        message: str,
        stage: Optional[str] = None
    ):
        """Send a progress update to clients"""
        await self.send_to_clients(processing_id, {
            'type': 'progress',
            'progress': progress,
            'message': message,
            'stage': stage
        })
    
    async def send_error(self, processing_id: str, error: str):
        """Send an error message to clients"""
        await self.send_to_clients(processing_id, {
            'type': 'error',
            'message': error
        })
    
    async def send_complete(
        self, 
        processing_id: str, 
        summary: Optional[Dict] = None
    ):
        """Send a completion message to clients"""
        await self.send_to_clients(processing_id, {
            'type': 'complete',
            'summary': summary or {}
        })
    
    async def stream_for_client(
        self, 
        processing_id: str
    ) -> AsyncGenerator[str, None]:
        """Stream messages for a specific client"""
        client_queue = await self.add_client(processing_id)
        
        try:
            # Send initial connection confirmation
            yield f"data: {json.dumps({'type': 'connected', 'processing_id': processing_id, 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            
            while True:
                try:
                    # Wait for message with timeout for heartbeat
                    message = await asyncio.wait_for(
                        client_queue.get(), 
                        timeout=30.0
                    )
                    
                    # Format as SSE
                    yield f"data: {message}\n\n"
                    
                    # Check if this is a completion or error message
                    try:
                        msg_data = json.loads(message)
                        if msg_data.get('type') in ['complete', 'error']:
                            logger.info(f"Stream ending for {processing_id}: {msg_data.get('type')}")
                            break
                    except json.JSONDecodeError:
                        pass
                    
                except asyncio.TimeoutError:
                    # Send heartbeat comment to keep connection alive
                    yield ': heartbeat\n\n'
                    
                except asyncio.CancelledError:
                    logger.info(f"Stream cancelled for processing {processing_id}")
                    break
                    
                except Exception as e:
                    logger.error(f"Error in stream for processing {processing_id}: {e}")
                    yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                    break
                    
        finally:
            await self.remove_client(processing_id, client_queue)
    
    def get_active_streams(self) -> Set[str]:
        """Get list of active processing streams"""
        return self.active_streams.copy()
    
    def get_client_count(self, processing_id: str) -> int:
        """Get number of connected clients for a processing job"""
        return len(self.clients.get(processing_id, set()))
    
    async def cleanup_processing(self, processing_id: str):
        """Clean up all clients for a processing job"""
        async with self._lock:
            if processing_id in self.clients:
                # Send final cleanup message to all clients
                await self.send_complete(processing_id, {
                    'message': 'Processing stream closed'
                })
                
                # Wait a bit for messages to be delivered
                await asyncio.sleep(0.5)
                
                del self.clients[processing_id]
                self.active_streams.discard(processing_id)
                logger.info(f"Cleaned up processing {processing_id}")


# Global SSE manager instance for upload/processing streams
sse_manager = SSEManager()