from pydantic_settings import BaseSettings
from typing import List, Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "BYU Pathway Topic Analyzer"
    
    # Database Configuration  
    DATABASE_URL: str
    
    # Security
    DEV_PASSWORD: str
    SECRET_KEY: str
    
    # CORS and Security
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "*"]
    
    # OpenAI Configuration (from hybrid analysis requirements)
    OPENAI_API_KEY: str
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536
    CHAT_MODEL: str = "gpt-5-nano"
    
    # Analysis Configuration (exact settings from hybrid script)
    SIMILARITY_THRESHOLD: float = 0.70
    REPRESENTATIVE_QUESTION_METHOD: str = "centroid" 
    PROCESSING_MODE: str = "sample"
    SAMPLE_SIZE: int = 2000
    UMAP_N_COMPONENTS: int = 5
    MIN_CLUSTER_SIZE: int = 3
    RANDOM_SEED: int = 42
    CACHE_EMBEDDINGS: bool = True
    CACHE_DIR: str = "embeddings_cache"
    
    # Google Sheets Configuration
    GOOGLE_SERVICE_ACCOUNT_TYPE: str = "service_account"
    GOOGLE_SERVICE_ACCOUNT_PROJECT_ID: str
    GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID: str  
    GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY: str
    GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL: str
    GOOGLE_SERVICE_ACCOUNT_CLIENT_ID: str
    GOOGLE_SERVICE_ACCOUNT_AUTH_URI: str = "https://accounts.google.com/o/oauth2/auth"
    GOOGLE_SERVICE_ACCOUNT_TOKEN_URI: str = "https://oauth2.googleapis.com/token"
    GOOGLE_SERVICE_ACCOUNT_AUTH_PROVIDER_CERT_URL: str = "https://www.googleapis.com/oauth2/v1/certs"
    GOOGLE_SERVICE_ACCOUNT_CLIENT_CERT_URL: str
    GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN: str = "googleapis.com"
    
    # Google Sheets IDs
    QUESTIONS_SHEET_ID: str = "1KIu4W9-BYRpZKxrpoWy6qpCBXjSDeRRmKek6q71wTRE"
    TOPICS_SHEET_ID: Optional[str] = None  # Will be configurable by developers
    
    # Queue Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Rate Limiting
    MAX_CONCURRENT_REQUESTS: int = 5
    ENABLE_ASYNC_PROCESSING: bool = True
    
    # File Upload
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    UPLOAD_DIR: str = "uploads"
    
    # Batch Processing (for embedding generation)
    EMBEDDING_BATCH_SIZE: int = 1000  # Match reference implementation
    EMBEDDING_RATE_LIMIT_PAUSE: int = 1  # Seconds to pause every N batches
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


# Initialize settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings instance"""
    return settings


def get_google_credentials_dict() -> dict:
    """
    Generate Google service account credentials dictionary from environment variables.
    Used for Google Sheets API authentication.
    """
    return {
        "type": settings.GOOGLE_SERVICE_ACCOUNT_TYPE,
        "project_id": settings.GOOGLE_SERVICE_ACCOUNT_PROJECT_ID,
        "private_key_id": settings.GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY_ID,
        "private_key": settings.GOOGLE_SERVICE_ACCOUNT_PRIVATE_KEY.replace('\\n', '\n'),  # Handle escaped newlines
        "client_email": settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL,
        "client_id": settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_ID,
        "auth_uri": settings.GOOGLE_SERVICE_ACCOUNT_AUTH_URI,
        "token_uri": settings.GOOGLE_SERVICE_ACCOUNT_TOKEN_URI,
        "auth_provider_x509_cert_url": settings.GOOGLE_SERVICE_ACCOUNT_AUTH_PROVIDER_CERT_URL,
        "client_x509_cert_url": settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_CERT_URL,
        "universe_domain": settings.GOOGLE_SERVICE_ACCOUNT_UNIVERSE_DOMAIN
    }


def validate_settings():
    """
    Validate critical settings are properly configured.
    Called during application startup.
    """
    errors = []
    
    # Check required OpenAI settings
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "your-api-key-here":
        errors.append("OPENAI_API_KEY is not configured")
    
    # Check database URL
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL is not configured")
    
    # Check Google Sheets credentials
    if not settings.GOOGLE_SERVICE_ACCOUNT_CLIENT_EMAIL:
        errors.append("Google Sheets credentials not configured")
    
    # Check security settings
    if not settings.SECRET_KEY or settings.SECRET_KEY == "your-secret-key-here":
        errors.append("SECRET_KEY is not configured")
    
    if not settings.DEV_PASSWORD or settings.DEV_PASSWORD == "your-dev-password":
        errors.append("DEV_PASSWORD is not configured")
    
    if errors:
        error_msg = "\n".join([f"  - {error}" for error in errors])
        raise ValueError(f"Configuration errors:\n{error_msg}")
    
    return True