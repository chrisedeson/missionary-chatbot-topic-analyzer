"""
Server-Sent Events (SSE) Manager

Handles real-time progress updates for data processing and analysis.
"""

import asyncio
import logging
from typing import Dict, Set, AsyncGenerator
from collections import defaultdict

logger = logging.getLogger(__name__)

class SSEManager:
    """Manages Server-Sent Events for real-time updates"""
    
    def __init__(self):
        self.clients: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self.active_streams: Set[str] = set()
    
    def add_client(self, processing_id: str) -> asyncio.Queue:
        """Add a client to receive updates for a processing job"""
        client_queue = asyncio.Queue()
        self.clients[processing_id].add(client_queue)
        self.active_streams.add(processing_id)
        logger.info(f"Added client for processing {processing_id}")
        return client_queue
    
    def remove_client(self, processing_id: str, client_queue: asyncio.Queue = None):
        """Remove a client from receiving updates"""
        if client_queue:
            self.clients[processing_id].discard(client_queue)
        
        if not self.clients[processing_id]:
            self.active_streams.discard(processing_id)
            del self.clients[processing_id]
            logger.info(f"Removed all clients for processing {processing_id}")
    
    async def send_to_client(self, processing_id: str, message: str):
        """Send a message to all clients listening to a processing job"""
        if processing_id not in self.clients:
            logger.warning(f"No clients for processing {processing_id}")
            return
        
        # Send to all clients for this processing job
        for client_queue in self.clients[processing_id].copy():
            try:
                await client_queue.put(message)
            except Exception as e:
                logger.error(f"Failed to send message to client: {e}")
                self.clients[processing_id].discard(client_queue)
    
    async def stream_for_client(self, processing_id: str) -> AsyncGenerator[str, None]:
        """Stream messages for a specific client"""
        client_queue = self.add_client(processing_id)
        
        try:
            while True:
                try:
                    # Wait for message with timeout
                    message = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                    yield message
                except asyncio.TimeoutError:
                    # Send heartbeat to keep connection alive
                    yield '{"type": "heartbeat"}'
                except asyncio.CancelledError:
                    logger.info(f"Stream cancelled for processing {processing_id}")
                    break
                except Exception as e:
                    logger.error(f"Error in stream for processing {processing_id}: {e}")
                    break
        finally:
            self.remove_client(processing_id, client_queue)
    
    def get_active_streams(self) -> Set[str]:
        """Get list of active processing streams"""
        return self.active_streams.copy()
    
    def cleanup_processing(self, processing_id: str):
        """Clean up all clients for a processing job"""
        if processing_id in self.clients:
            del self.clients[processing_id]
            self.active_streams.discard(processing_id)
            logger.info(f"Cleaned up processing {processing_id}")

# Global SSE manager instance
sse_manager = SSEManager()