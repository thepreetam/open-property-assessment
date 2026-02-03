"""
Teams and workspaces (Phase 3 multi-tenant).
"""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from api.auth import require_api_key
from db.models import Team, Workspace, WorkspaceMember
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


def get_db_session():
    factory = get_session_factory_or_none()
    if factory is None:
        yield None
        return
    session = factory()
    try:
        yield session
    finally:
        session.close()


class TeamCreate(BaseModel):
    name: str


class WorkspaceCreate(BaseModel):
    team_id: str
    name: str


class MemberCreate(BaseModel):
    user_id: str
    role: str = "member"  # admin | member | viewer


@router.post("/teams")
def create_team(
    body: TeamCreate,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    team = Team(id=str(uuid.uuid4()), name=body.name)
    session.add(team)
    session.commit()
    return {"id": team.id, "name": team.name}


@router.post("")
def create_workspace(
    body: WorkspaceCreate,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    ws = Workspace(id=str(uuid.uuid4()), team_id=body.team_id, name=body.name)
    session.add(ws)
    session.commit()
    return {"id": ws.id, "team_id": ws.team_id, "name": ws.name}


@router.post("/{workspace_id}/members")
def add_member(
  workspace_id: str,
  body: MemberCreate,
  session: Optional[Session] = Depends(get_db_session),
  _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    m = WorkspaceMember(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        user_id=body.user_id,
        role=body.role,
    )
    session.add(m)
    session.commit()
    return {"id": m.id, "workspace_id": m.workspace_id, "user_id": m.user_id, "role": m.role}
