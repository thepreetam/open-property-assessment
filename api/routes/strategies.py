"""
Strategies: get strategies by job_id or property_id (Phase 2).
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from api.auth import require_api_key
from api.models import StrategiesResponse
from core.repair_option_generator import build_timeline_tasks
from db.models import Job, Property
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/strategies", tags=["strategies"])


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


@router.get("", response_model=StrategiesResponse)
def get_strategies(
    job_id: Optional[str] = Query(None, description="Get strategies from this job result"),
    property_id: Optional[str] = Query(None, description="Latest job for this property"),
    strategy_key: Optional[str] = Query(None, description="Include timeline for this strategy"),
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    if not job_id and not property_id:
        raise HTTPException(status_code=400, detail="Provide job_id or property_id.")
    if job_id:
        job = session.query(Job).filter(Job.id == job_id).first()
    else:
        job = session.query(Job).filter(Job.property_id == property_id).order_by(Job.created_at.desc()).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "completed" or not job.result:
        raise HTTPException(status_code=404, detail="Job not completed or no result.")
    strategies = job.result.get("strategies", {})
    recommendation = job.result.get("recommendation", {})
    timeline = None
    if strategy_key and strategy_key in strategies:
        repairs = strategies[strategy_key].get("repairs", [])
        timeline = build_timeline_tasks(repairs)
    return StrategiesResponse(
        job_id=job.id,
        property_id=job.property_id,
        strategies=strategies,
        recommendation=recommendation,
        timeline=timeline,
    )
