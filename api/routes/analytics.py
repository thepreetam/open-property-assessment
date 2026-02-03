"""
Analytics and BI metrics (Phase 3). Stub: aggregate counts and executive summary.
"""
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from api.auth import require_api_key
from db.models import Job, Property, Execution
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/analytics", tags=["analytics"])


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


@router.get("/metrics")
def get_metrics(
    workspace_id: Optional[str] = Query(None),
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
) -> Dict[str, Any]:
    """Aggregate metrics: properties count, jobs completed, executions."""
    if session is None:
        return {"error": "Database not configured", "properties": 0, "jobs_completed": 0, "executions": 0}
    q_props = session.query(func.count(Property.id))
    q_jobs = session.query(func.count(Job.id)).filter(Job.status == "completed")
    q_exec = session.query(func.count(Execution.id))
    if workspace_id:
        q_props = q_props.filter(Property.workspace_id == workspace_id)
        q_jobs = q_jobs.filter(Job.workspace_id == workspace_id)
    return {
        "properties": q_props.scalar() or 0,
        "jobs_completed": q_jobs.scalar() or 0,
        "executions": q_exec.scalar() or 0,
    }


@router.get("/executive-summary")
def executive_summary(
    workspace_id: Optional[str] = Query(None),
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
) -> Dict[str, Any]:
    """Stub: executive report placeholder."""
    if session is None:
        return {"summary": "Database not configured", "metrics": {}, "insights": [], "period": "all_time"}
    q_props = session.query(func.count(Property.id))
    q_jobs = session.query(func.count(Job.id)).filter(Job.status == "completed")
    q_exec = session.query(func.count(Execution.id))
    if workspace_id:
        q_props = q_props.filter(Property.workspace_id == workspace_id)
        q_jobs = q_jobs.filter(Job.workspace_id == workspace_id)
    metrics = {
        "properties": q_props.scalar() or 0,
        "jobs_completed": q_jobs.scalar() or 0,
        "executions": q_exec.scalar() or 0,
    }
    return {
        "summary": "Repair ROI Optimizer metrics (stub)",
        "metrics": metrics,
        "insights": [],
        "period": "all_time",
    }
