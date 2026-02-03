"""
Structured config from environment for API, worker, and DB.
"""
import os
from typing import Optional


def _str(value: Optional[str], default: str = "") -> str:
    return (value or "").strip() or default


def _int(value: Optional[str], default: int = 0) -> int:
    try:
        return int(value) if value else default
    except ValueError:
        return default


class Settings:
    """Application settings from environment."""

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database (PostgreSQL)
    database_url: str = ""

    # Redis (Celery broker + optional cache)
    redis_url: str = ""

    # ML / pipeline (for worker)
    use_llava: bool = False
    free_tier: bool = True

    # Optional base URL for status_url in responses
    api_base_url: str = ""

    # API key for external access (Phase 2). If set, X-API-Key header required.
    api_key: str = ""

    def __init__(self) -> None:
        self.api_host = _str(os.environ.get("API_HOST"), "0.0.0.0")
        self.api_port = _int(os.environ.get("API_PORT"), 8000)
        self.database_url = _str(os.environ.get("DATABASE_URL"), "")
        self.redis_url = _str(os.environ.get("REDIS_URL"), "")
        self.use_llava = os.environ.get("USE_LLAVA", "false").lower() == "true"
        self.free_tier = os.environ.get("FREE_TIER", "true").lower() == "true"
        self.api_base_url = _str(os.environ.get("API_BASE_URL"), "")
        self.api_key = _str(os.environ.get("API_KEY"), "")

    @property
    def database_configured(self) -> bool:
        return bool(self.database_url)

    @property
    def redis_configured(self) -> bool:
        return bool(self.redis_url)


settings = Settings()
