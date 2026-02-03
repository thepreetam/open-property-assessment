"""
Contractor API surface (Phase 3): capture photo, get repairs, submit completion.
REST/API-first; same endpoints can power a mobile SDK.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from pydantic import BaseModel

from api.auth import require_api_key
from core.repair_option_generator import build_timeline_tasks
from db.models import Execution, Job
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/contractor", tags=["contractor"])


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


class WorkCompletionBody(BaseModel):
    execution_id: str
    status: str = "completed"  # completed | in_progress | cancelled
    notes: Optional[str] = None


@router.get("/repairs")
def contractor_get_repairs(
    property_id: Optional[str] = Query(None),
    job_id: Optional[str] = Query(None),
    strategy_key: str = Query("value_add"),
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
) -> dict:
    """Get recommended repairs for a property/job (contractor-facing)."""
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    if not property_id and not job_id:
        raise HTTPException(status_code=400, detail="Provide property_id or job_id.")
    if job_id:
        job = session.query(Job).filter(Job.id == job_id).first()
    else:
        job = session.query(Job).filter(Job.property_id == property_id).order_by(Job.created_at.desc()).first()
    if not job or job.status != "completed" or not job.result:
        raise HTTPException(status_code=404, detail="No completed job result found.")
    strategies = job.result.get("strategies", {})
    recommendation = job.result.get("recommendation", {})
    timeline = None
    if strategy_key in strategies:
        repairs = strategies[strategy_key].get("repairs", [])
        timeline = build_timeline_tasks(repairs)
    return {
        "property_id": job.property_id,
        "job_id": job.id,
        "strategy_key": strategy_key,
        "recommendation": recommendation,
        "timeline": timeline,
        "strategies": strategies,
    }


@router.post("/completion")
def submit_work_completion(
    body: WorkCompletionBody,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    """Submit work completion (contractor). Stub: updates execution status."""
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    execution = session.query(Execution).filter(Execution.id == body.execution_id).first()
    if not execution:
        raise HTTPException(status_code=404, detail="Execution not found.")
    execution.status = body.status
    session.commit()
    return {"execution_id": execution.id, "status": execution.status}
