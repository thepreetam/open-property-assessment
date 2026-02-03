"""
Export (Phase 4): CSV, PDF, Excel, Sheets-compatible.
"""
import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from api.auth import require_api_key
from db.models import Job
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/export", tags=["export"])


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


@router.get("/job/{job_id}/csv")
def export_job_csv(
    job_id: str,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    """Export job result as CSV (per-photo data + summary)."""
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    job = session.query(Job).filter(Job.id == job_id).first()
    if not job or job.status != "completed" or not job.result:
        raise HTTPException(status_code=404, detail="Job not found or no result.")
    import csv
    all_data = job.result.get("all_data", [])
    if not all_data:
        raise HTTPException(status_code=404, detail="No per-photo data to export.")
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Photo", "Room", "Quality excerpt", "Adjustment $", "% Adj", "Notes"])
    writer.writeheader()
    writer.writerows(all_data)
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}.csv"},
    )


@router.get("/job/{job_id}/summary")
def export_job_summary_text(
    job_id: str,
    session: Optional[Session] = Depends(get_db_session),
    _api_key: Optional[str] = Depends(require_api_key),
):
    """Export job result as plain text summary (strategies + recommendation)."""
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    job = session.query(Job).filter(Job.id == job_id).first()
    if not job or job.status != "completed" or not job.result:
        raise HTTPException(status_code=404, detail="Job not found or no result.")
    rec = job.result.get("recommendation", {})
    strategies = job.result.get("strategies", {})
    lines = [
        f"Job {job_id}",
        f"Recommendation: {rec.get('strategy_name', 'N/A')} – {rec.get('reason', '')}",
        "",
    ]
    for key, s in strategies.items():
        lines.append(f"{s.get('name', key)}: ${s.get('cost', 0):,} cost, {s.get('roi_pct', 0)}% ROI, {s.get('timeline_days', 0)} days")
    body = "\n".join(lines)
    return StreamingResponse(
        iter([body]),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename=job_{job_id}_summary.txt"},
    )
