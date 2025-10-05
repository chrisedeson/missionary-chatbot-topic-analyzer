from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import structlog

from app.core.config import settings
from app.core.database import init_db, health_check
from app.api.routes import auth, dashboard, upload, analysis, sheets

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    logger.info("Starting BYU Pathway Topic Analyzer API")
    
    # Initialize database
    await init_db()
    
    # Initialize services after database connection
    try:
        from app.services.questions import init_questions_service
        await init_questions_service()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize services: {e}")
        logger.info("Continuing without service initialization (development mode)")
    
    yield
    
    logger.info("Shutting down BYU Pathway Topic Analyzer API")


app = FastAPI(
    title="BYU Pathway Topic Analyzer API",
    description="API for analyzing and managing student chatbot questions",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add trusted host middleware for security
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=settings.ALLOWED_HOSTS
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(sheets.router, prefix="/api/sheets", tags=["sheets"])


@app.get("/")
async def root():
    return {
        "message": "BYU Pathway Topic Analyzer API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check_endpoint():
    """Health check endpoint with database status"""
    try:
        db_health = await health_check()
        return {
            "status": "healthy",
            "database": db_health,
            "environment": {
                "database_url_configured": bool(settings.DATABASE_URL),
                "debug": settings.DEBUG
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "database": {"status": "error", "message": str(e)}
        }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
