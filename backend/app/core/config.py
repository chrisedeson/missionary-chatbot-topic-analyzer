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
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

def get_settings() -> Settings:
    """Get application settings instance"""
    return settings