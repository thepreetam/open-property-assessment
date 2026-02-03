"""
RBAC (Phase 3): workspace membership and role check.
"""
from typing import Optional

from fastapi import HTTPException, Header
from sqlalchemy.orm import Session
from db.models import WorkspaceMember
from db.session import get_session_factory_or_none


def get_current_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> Optional[str]:
    """Dependency: optional user id from header (stub; real impl would use OAuth)."""
    return x_user_id


def require_workspace_role(
    workspace_id: str,
    min_role: str = "member",  # admin | member | viewer
    session: Optional[Session] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Check user has at least min_role in workspace. Role order: viewer < member < admin."""
    if not session or not user_id:
        return False
    member = session.query(WorkspaceMember).filter(
        WorkspaceMember.workspace_id == workspace_id,
        WorkspaceMember.user_id == user_id,
    ).first()
    if not member:
        return False
    order = {"viewer": 0, "member": 1, "admin": 2}
    return order.get(member.role, 0) >= order.get(min_role, 0)


def get_workspace_role_dependency(min_role: str = "member"):
    """Return a dependency that checks X-User-Id and X-Workspace-Id and role."""
    def _check(
        x_workspace_id: Optional[str] = Header(None, alias="X-Workspace-Id"),
        x_user_id: Optional[str] = Header(None, alias="X-User-Id"),
    ):
        if not x_workspace_id or not x_user_id:
            raise HTTPException(status_code=403, detail="X-Workspace-Id and X-User-Id required for this action")
        factory = get_session_factory_or_none()
        if not factory:
            raise HTTPException(status_code=503, detail="Database not configured")
        session = factory()
        try:
            if not require_workspace_role(x_workspace_id, min_role, session, x_user_id):
                raise HTTPException(status_code=403, detail="Insufficient role in workspace")
            return {"workspace_id": x_workspace_id, "user_id": x_user_id}
        finally:
            session.close()
    return _check
