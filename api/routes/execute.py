"""
Execute strategy stub (Phase 2): dispatch repair execution.
"""
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.auth import require_api_key
from api.models import ExecuteRequest, ExecuteResponse
from core.repair_option_generator import build_timeline_tasks
from db.models import Execution, Job, Property
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/execute", tags=["execute"])


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


@router.post("", response_model=ExecuteResponse)
def execute_strategy(
    body: ExecuteRequest,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    prop = session.query(Property).filter(Property.id == body.property_id).first()
    if not prop:
        raise HTTPException(status_code=404, detail="Property not found.")
    strategies = None
    if body.job_id:
        job = session.query(Job).filter(Job.id == body.job_id).first()
        if job and job.result:
            strategies = job.result.get("strategies", {})
    if not strategies:
        job = session.query(Job).filter(Job.property_id == body.property_id).order_by(Job.created_at.desc()).first()
        if job and job.result:
            strategies = job.result.get("strategies", {})
    timeline = None
    if strategies and body.strategy_key in strategies:
        repairs = strategies[body.strategy_key].get("repairs", [])
        timeline = {"tasks": build_timeline_tasks(repairs)}
    execution = Execution(
        id=str(uuid.uuid4()),
        property_id=body.property_id,
        job_id=body.job_id,
        strategy_key=body.strategy_key,
        status="dispatched",
        payload=timeline,
    )
    session.add(execution)
    session.commit()
    return ExecuteResponse(
        execution_id=execution.id,
        status="dispatched",
        property_id=body.property_id,
        strategy_key=body.strategy_key,
        timeline=timeline,
    )
