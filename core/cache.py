"""
Redis cache for pipeline results and rate limiting (Phase 4).
"""
import hashlib
import json
from typing import Any, Optional

from core.config import settings


def _redis_client():
    """Lazy Redis client."""
    if not settings.redis_url:
        return None
    try:
        import redis
        return redis.from_url(settings.redis_url, decode_responses=True)
    except Exception:
        return None


def cache_key(prefix: str, *parts: str) -> str:
    """Build cache key from prefix and parts."""
    raw = ":".join(str(p) for p in parts)
    h = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"repair_opt:{prefix}:{h}"


def get_cached_result(key: str) -> Optional[dict]:
    """Get cached JSON result. Returns None if miss or Redis unavailable."""
    client = _redis_client()
    if not client:
        return None
    try:
        data = client.get(key)
        return json.loads(data) if data else None
    except Exception:
        return None


def set_cached_result(key: str, value: dict, ttl_seconds: int = 3600) -> bool:
    """Set cached result. TTL default 1 hour."""
    client = _redis_client()
    if not client:
        return False
    try:
        client.setex(key, ttl_seconds, json.dumps(value))
        return True
    except Exception:
        return False
