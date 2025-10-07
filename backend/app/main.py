from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import structlog

from app.core.config import settings, validate_settings
from app.core.database import init_db, close_db, health_check

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if not settings.DEBUG else structlog.dev.ConsoleRenderer()
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
    logger.info("Starting BYU Pathway Topic Analyzer API", version="1.0.0", debug=settings.DEBUG)
    
    # Validate settings
    try:
        validate_settings()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error("Configuration validation failed", error=str(e))
        logger.warning("Some features may not work correctly")
    
    # Initialize database
    await init_db()
    
    # Initialize services after database connection
    try:
        from app.services.questions import init_questions_service
        await init_questions_service()
        logger.info("Services initialized successfully")
    except Exception as e:
        logger.warning("Failed to initialize services", error=str(e))
        logger.info("Continuing without service initialization (development mode)")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down BYU Pathway Topic Analyzer API")
    await close_db()


app = FastAPI(
    title="BYU Pathway Topic Analyzer API",
    description="API for analyzing and managing Missionary chatbot questions with hybrid topic discovery",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.DEBUG else None,  # Only expose docs in debug mode
    redoc_url="/api/redoc" if settings.DEBUG else None
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
from app.api.routes import auth, dashboard, upload, analysis, sheets

app.include_router(auth.router, prefix="/api/auth", tags=["authentication"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(sheets.router, prefix="/api/sheets", tags=["sheets"])


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "BYU Pathway Topic Analyzer API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/api/docs" if settings.DEBUG else "disabled",
        "endpoints": {
            "health": "/health",
            "auth": "/api/auth",
            "dashboard": "/api/dashboard",
            "upload": "/api/upload",
            "analysis": "/api/analysis",
            "sheets": "/api/sheets"
        }
    }


@app.get("/health", tags=["health"])
async def health_check_endpoint():
    """Health check endpoint with database and service status"""
    try:
        db_health = await health_check()
        
        # Check services
        services_status = {}
        try:
            from app.services.questions import questions_service
            services_status["questions_service"] = "initialized" if questions_service else "not_initialized"
        except Exception as e:
            services_status["questions_service"] = f"error: {str(e)}"
        
        return {
            "status": "healthy" if db_health["status"] == "healthy" else "degraded",
            "database": db_health,
            "services": services_status,
            "environment": {
                "database_url_configured": bool(settings.DATABASE_URL),
                "openai_api_key_configured": bool(settings.OPENAI_API_KEY),
                "google_sheets_configured": bool(settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL),
                "debug": settings.DEBUG
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
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
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )