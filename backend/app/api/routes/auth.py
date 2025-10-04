from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
import structlog

from app.core.auth import SimpleAuth

logger = structlog.get_logger()
router = APIRouter()


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    authenticated: bool
    role: str
    message: str
    token: str


@router.post("/login", response_model=LoginResponse)
async def developer_login(request: LoginRequest):
    """Authenticate developer with password"""
    
    logger.info("Developer login attempt")
    
    if not SimpleAuth.verify_developer_password(request.password):
        logger.warning("Invalid developer login attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid developer password"
        )
    
    logger.info("Developer login successful")
    
    # For simple auth, we'll return the password as the token
    # In production, you'd want to use JWT or similar
    return LoginResponse(
        authenticated=True,
        role="developer",
        message="Developer authentication successful",
        token=request.password
    )


@router.post("/logout")
async def developer_logout():
    """Logout developer (simple acknowledgment)"""
    logger.info("Developer logout")
    return {"message": "Logout successful"}


@router.get("/status")
async def auth_status():
    """Get authentication status endpoint"""
    return {"auth_enabled": True, "auth_type": "simple_password"}
