"""
Properties and portfolio (Phase 2).
"""
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.auth import require_api_key
from api.models import PropertyCreate, PropertyResponse
from db.models import Property
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/properties", tags=["properties"])


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


@router.post("", response_model=PropertyResponse)
def create_property(
    body: PropertyCreate,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    prop = Property(
        id=str(uuid.uuid4()),
        workspace_id=body.workspace_id,
        address=body.address,
        zip_code=body.zip_code,
        home_value=str(body.home_value) if body.home_value is not None else None,
    )
    session.add(prop)
    session.commit()
    session.refresh(prop)
    return PropertyResponse(
        id=prop.id,
        workspace_id=prop.workspace_id,
        address=prop.address,
        zip_code=prop.zip_code,
        home_value=prop.home_value,
        created_at=prop.created_at.isoformat() if prop.created_at else None,
    )


@router.get("", response_model=List[PropertyResponse])
def list_properties(
    workspace_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    q = session.query(Property)
    if workspace_id:
        q = q.filter(Property.workspace_id == workspace_id)
    q = q.order_by(Property.created_at.desc()).limit(limit)
    rows = q.all()
    return [
        PropertyResponse(
            id=r.id,
            workspace_id=r.workspace_id,
            address=r.address,
            zip_code=r.zip_code,
            home_value=r.home_value,
            created_at=r.created_at.isoformat() if r.created_at else None,
        )
        for r in rows
    ]


@router.get("/{property_id}", response_model=PropertyResponse)
def get_property(
    property_id: str,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    prop = session.query(Property).filter(Property.id == property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found.")
    return PropertyResponse(
        id=prop.id,
        workspace_id=prop.workspace_id,
        address=prop.address,
        zip_code=prop.zip_code,
        home_value=prop.home_value,
        created_at=prop.created_at.isoformat() if prop.created_at else None,
    )
