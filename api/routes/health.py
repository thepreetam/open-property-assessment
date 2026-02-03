"""
Health and readiness endpoints.
"""
from fastapi import APIRouter, Depends
from api.models import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health():
    """Liveness: service is up."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
def ready():
    """Readiness: service can accept work. DB/Redis optional for Phase 1."""
    from core.config import settings
    db = "connected" if settings.database_configured else "not_configured"
    redis = "connected" if settings.redis_configured else "not_configured"
    status = "ready" if True else "not_ready"  # Always ready in Phase 1; can require DB later
    return ReadinessResponse(status=status, database=db, redis=redis)
