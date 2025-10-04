from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import structlog
from app.core.config import settings

logger = structlog.get_logger()
security = HTTPBearer(auto_error=False)


class SimpleAuth:
    """Simple password-based authentication for developers"""
    
    @staticmethod
    def verify_developer_password(password: str) -> bool:
        """Verify developer password"""
        return password == settings.DEV_PASSWORD
    
    @staticmethod
    def create_developer_session() -> dict:
        """Create a simple session token for developers"""
        return {
            "authenticated": True,
            "role": "developer",
            "message": "Developer authentication successful"
        }


async def get_current_developer(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Dependency to verify developer authentication"""
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # For simple auth, we'll use the token as the password
    if not SimpleAuth.verify_developer_password(credentials.credentials):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid developer password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return {"role": "developer", "authenticated": True}


# Optional dependency for endpoints that work for both authenticated and unauthenticated users
async def get_optional_developer(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Optional dependency for developer authentication"""
    
    if not credentials:
        return None
    
    try:
        return await get_current_developer(credentials)
    except HTTPException:
        return None


# Alias for backward compatibility and clear naming
require_developer_auth = get_current_developer