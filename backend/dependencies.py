"""Dependency injection functions for routes."""
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from auth.utils import verify_token

security = HTTPBearer()


async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    """Extract and validate JWT token from request header."""
    data = verify_token(creds.credentials)
    if not data:
        raise HTTPException(401, "Invalid or expired token")
    return data


async def require_admin(user=Depends(get_current_user)):
    """Enforce admin-only access."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin access required")
    return user


def require_authority(user=Depends(get_current_user)):
    """Enforce authority or admin access."""
    if user.get("role") not in ("admin", "authority"):
        raise HTTPException(403, "Authority access required")
    return user
