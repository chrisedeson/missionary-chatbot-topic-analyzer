from prisma import Prisma
import structlog
from app.core.config import settings

logger = structlog.get_logger()

# Global database instance
db = Prisma()


async def init_db():
    """Initialize database connection"""
    try:
        logger.info("Connecting to database...")
        await db.connect()
        logger.info("Database connected successfully")
    except Exception as e:
        logger.error("Failed to connect to database", error=str(e))
        logger.warning("Continuing without database connection (development mode)")
        # Don't raise in development - allow API to start without DB


async def close_db():
    """Close database connection"""
    try:
        await db.disconnect()
        logger.info("Database connection closed")
    except Exception as e:
        logger.error("Error closing database connection", error=str(e))


async def get_db():
    """Dependency to get database instance"""
    return db