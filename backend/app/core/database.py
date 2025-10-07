from prisma import Prisma
import structlog
import asyncio
from app.core.config import settings

logger = structlog.get_logger()

# Global database instance - Prisma will auto-load DATABASE_URL from .env
db = Prisma()


async def init_db():
    """Initialize database connection with Neon cold-start handling"""
    try:
        logger.info("Connecting to database...")
        logger.info(f"Database URL configured: {'YES' if settings.DATABASE_URL else 'NO'}")
        
        # Retry connection with exponential backoff for Neon cold starts
        for attempt in range(3):
            try:
                await asyncio.wait_for(db.connect(), timeout=30.0)  # Longer timeout for Neon
                logger.info("Database connected successfully")
                
                # Test the connection with execute_raw (query_raw is deprecated)
                await db.execute_raw("SELECT 1")
                logger.info("Database connection verified")
                return
                
            except asyncio.TimeoutError:
                logger.warning(f"Database connection timeout (attempt {attempt + 1}/3) - Neon may be cold starting")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
            except Exception as e:
                logger.warning(f"Database connection failed (attempt {attempt + 1}/3): {str(e)[:100]}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)
                else:
                    raise
                    
    except Exception as e:
        logger.error("Failed to connect to database", error=str(e))
        logger.warning("Continuing without database connection (development mode)")
        logger.info("Check DATABASE_URL and ensure Neon database is accessible")
        # Don't raise in development - allow API to start without DB


async def close_db():
    """Close database connection"""
    try:
        if db.is_connected():
            await db.disconnect()
            logger.info("Database connection closed")
    except Exception as e:
        logger.error("Error closing database connection", error=str(e))


async def get_db() -> Prisma:
    """
    Dependency to get database instance.
    
    Returns the global Prisma database instance. This instance is initialized
    during application startup via init_db().
    
    Usage in routes:
        @router.get("/example")
        async def example_route(db: Prisma = Depends(get_db)):
            questions = await db.question.find_many()
            return questions
    """
    if not db.is_connected():
        logger.warning("Database not connected - attempting to connect")
        try:
            await db.connect()
        except Exception as e:
            logger.error(f"Failed to connect database in get_db: {e}")
            raise
    
    return db


async def health_check() -> dict:
    """Check database health and connection status"""
    try:
        # Check if connected
        if not db.is_connected():
            return {
                "status": "disconnected",
                "message": "Database not connected"
            }
        
        # Test database connectivity with execute_raw
        await asyncio.wait_for(db.execute_raw("SELECT 1"), timeout=5.0)
        
        # Get basic counts for health check
        try:
            question_count = await db.question.count()
            topic_count = await db.topic.count()
            cache_count = await db.embeddingcache.count()
            
            return {
                "status": "healthy",
                "message": "Database connection active",
                "database_url_configured": bool(settings.DATABASE_URL),
                "stats": {
                    "questions": question_count,
                    "topics": topic_count,
                    "cached_embeddings": cache_count
                }
            }
        except Exception as stats_error:
            # Return healthy status even if counts fail (tables might not exist yet)
            logger.warning(f"Could not fetch database stats: {stats_error}")
            return {
                "status": "healthy",
                "message": "Database connection active (stats unavailable)",
                "database_url_configured": bool(settings.DATABASE_URL)
            }
        
    except asyncio.TimeoutError:
        return {
            "status": "timeout",
            "message": "Database query timeout (Neon may be sleeping)"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}"
        }


async def ensure_connection():
    """
    Ensure database connection is active.
    Helper function for long-running operations.
    """
    if not db.is_connected():
        logger.info("Reconnecting to database...")
        await db.connect()
        logger.info("Database reconnected")