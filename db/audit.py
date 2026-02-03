"""
Audit log helper (Phase 3).
"""
import uuid
from typing import Any, Optional

from sqlalchemy.orm import Session
from db.models import AuditLog


def log(
    session: Session,
    action: str,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    payload: Optional[dict] = None,
) -> None:
    """Append an audit log entry."""
    entry = AuditLog(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        user_id=user_id,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        payload=payload,
    )
    session.add(entry)
