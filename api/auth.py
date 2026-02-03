"""
API key auth (Phase 2): when API_KEY env is set, require X-API-Key header.
"""
from fastapi import Header, HTTPException
from core.config import settings


def require_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Dependency: require X-API-Key when API_KEY is configured."""
    if not settings.api_key:
        return None
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return x_api_key
