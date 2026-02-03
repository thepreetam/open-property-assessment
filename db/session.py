"""
Database session dependency for FastAPI.
"""
from typing import Generator, Optional
from sqlalchemy.orm import Session
from core.config import settings
from db.models import Base, get_session_factory, get_engine

_session_factory: Optional[any] = None


def init_db() -> None:
    """Initialize session factory from settings when DATABASE_URL is set. Call at app startup."""
    global _session_factory
    if getattr(settings, "database_url", None):
        _session_factory = get_session_factory(settings.database_url)


def get_db() -> Generator[Session, None, None]:
    """Yield a DB session for FastAPI dependency injection."""
    if _session_factory is None:
        init_db()
    if _session_factory is None:
        raise RuntimeError("Database not configured (DATABASE_URL)")
    session = _session_factory()
    try:
        yield session
    finally:
        session.close()


def get_session_factory_or_none():
    """Return session factory if DB is configured, else None."""
    if _session_factory is None:
        init_db()
    return _session_factory
