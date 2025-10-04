from prisma import Prisma
import structlog
from app.core.config import settings

logger = structlog.get_logger()

# Global database instance
db = Prisma()


async def init_db():
    """Initialize database connection and run migrations"""
    try:
        logger.info("Connecting to database...")
        await db.connect()
        logger.info("Database connected successfully")
        
        # The Prisma client will automatically handle migrations
        # when using prisma generate and prisma db push
        
    except Exception as e:
        logger.error("Failed to connect to database", error=str(e))
        raise


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