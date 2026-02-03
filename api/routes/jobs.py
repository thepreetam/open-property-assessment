"""
Job creation and status endpoints.
"""
import asyncio
import base64
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session

from api.models import BulkJobRequest, BulkJobResponse, BulkJobError, JobCreateResponse, JobStatusResponse
from core.config import settings
from db.models import Job, Property
from db.session import get_session_factory_or_none

router = APIRouter(prefix="/jobs", tags=["jobs"])


def get_db_session():
    """Dependency: yield DB session if configured, else None."""
    factory = get_session_factory_or_none()
    if factory is None:
        yield None
        return
    session = factory()
    try:
        yield session
    finally:
        session.close()


@router.post("", response_model=JobCreateResponse)
async def create_job(
    home_value: int = Form(..., ge=200_000, le=10_000_000),
    zip_code: Optional[str] = Form(None),
    property_id: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    session: Optional[Session] = Depends(get_db_session),
):
    """
    Create an analysis job: upload 1–10 photos + home value + optional zip + optional property_id.
    Returns job_id and status_url. Processing runs asynchronously via Celery.
    """
    if session is None:
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL and REDIS_URL for async jobs.",
        )
    if not files or len(files) > 10:
        raise HTTPException(status_code=400, detail="Upload 1–10 photos (jpg, png).")

    photo_bytes_list: List[bytes] = []
    for f in files:
        if f.content_type and "image" not in f.content_type:
            continue
        photo_bytes_list.append(await f.read())
    if not photo_bytes_list:
        raise HTTPException(status_code=400, detail="No valid image files.")

    job_id = str(uuid.uuid4())
    job = Job(id=job_id, status="pending", property_id=property_id)
    session.add(job)
    session.commit()

    try:
        from core.metrics import jobs_started
        jobs_started.inc()
    except Exception:
        pass

    # Enqueue Celery task (photos as base64 for JSON broker); if Celery unavailable, run sync
    try:
        from worker.tasks import run_analysis_task
        photo_b64_list = [base64.b64encode(b).decode("utf-8") for b in photo_bytes_list]
        run_analysis_task.delay(
            job_id=job_id,
            photo_b64_list=photo_b64_list,
            home_value=home_value,
            zip_code=zip_code or "",
        )
    except ImportError:
        from core.pipeline import run_pipeline
        result = run_pipeline(
            photo_bytes_list,
            home_value,
            zip_code,
            free_tier=settings.free_tier,
            use_llava=settings.use_llava,
        )
        job.status = "completed"
        job.result = {
            "all_data": result["all_data"],
            "strategies": result["strategies"],
            "recommendation": result["recommendation"],
        }
        session.commit()
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Failed to enqueue job: {e}")

    base = getattr(settings, "api_base_url", "") or "http://localhost:8000"
    return JobCreateResponse(
        job_id=job_id,
        status="pending",
        status_url=f"{base.rstrip('/')}/api/v1/jobs/{job_id}",
        estimated_completion="2 minutes",
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    session: Optional[Session] = Depends(get_db_session),
):
    """Get job status and result."""
    if session is None:
        raise HTTPException(status_code=503, detail="Database not configured.")
    job = session.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(
        job_id=job.id,
        status=job.status,
        created_at=job.created_at.isoformat() if job.created_at else None,
        updated_at=job.updated_at.isoformat() if job.updated_at else None,
        result=job.result,
        error=job.error,
    )


@router.websocket("/{job_id}/stream")
async def job_stream(websocket: WebSocket, job_id: str):
    """
    WebSocket: push job status (pending → processing → completed/failed) every 1–2 s until terminal or client disconnect.
    """
    await websocket.accept()
    factory = get_session_factory_or_none()
    if factory is None:
        await websocket.send_json({"error": "Database not configured"})
        await websocket.close()
        return
    try:
        while True:
            session = factory()
            try:
                job = session.query(Job).filter(Job.id == job_id).first()
                if not job:
                    await websocket.send_json({"error": "Job not found"})
                    break
                await websocket.send_json({
                    "job_id": job.id,
                    "status": job.status,
                    "result": job.result,
                    "error": job.error,
                })
                if job.status in ("completed", "failed"):
                    break
            finally:
                session.close()
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass


async def _fetch_image_bytes(url: str, timeout_sec: float = 15.0) -> Optional[bytes]:
    """Download image bytes from URL; return None on failure."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as resp:
                if resp.status != 200:
                    return None
                ct = (resp.headers.get("Content-Type") or "").lower()
                if "image" not in ct and "octet-stream" not in ct:
                    return None
                return await resp.read()
    except Exception:
        return None


@router.post("/bulk", response_model=BulkJobResponse)
async def create_jobs_bulk(
    body: BulkJobRequest,
    session: Optional[Session] = Depends(get_db_session),
):
    """
    Bulk job creation: one Property per row (reused if address+zip match), one Job per row, enqueued via Celery.
    Each job's photo_urls are downloaded; if any download fails, that row is skipped and reported in errors.
    """
    if session is None:
        raise HTTPException(
            status_code=503,
            detail="Database not configured. Set DATABASE_URL and REDIS_URL for async jobs.",
        )
    job_ids: List[str] = []
    errors: List[BulkJobError] = []
    for idx, item in enumerate(body.jobs):
        photo_bytes_list: List[bytes] = []
        for url in item.photo_urls:
            b = await _fetch_image_bytes(url)
            if b is None:
                errors.append(BulkJobError(index=idx, reason=f"Failed to download image: {url[:80]}"))
                break
            photo_bytes_list.append(b)
        if len(photo_bytes_list) != len(item.photo_urls):
            continue
        if not photo_bytes_list:
            errors.append(BulkJobError(index=idx, reason="No valid images"))
            continue
        # Get or create property by address + zip_code
        property_id: Optional[str] = None
        addr = (item.address or "").strip() or None
        zip_ = (item.zip_code or "").strip() or None
        if addr is not None or zip_ is not None:
            prop = session.query(Property).filter(Property.address == addr, Property.zip_code == zip_).first()
            if prop:
                property_id = prop.id
            else:
                property_id = str(uuid.uuid4())
                session.add(
                    Property(id=property_id, address=addr, zip_code=zip_, home_value=str(item.home_value))
                )
                session.commit()
        job_id = str(uuid.uuid4())
        job = Job(id=job_id, status="pending", property_id=property_id)
        session.add(job)
        session.commit()
        try:
            from core.metrics import jobs_started
            jobs_started.inc()
        except Exception:
            pass
        try:
            from worker.tasks import run_analysis_task
            photo_b64_list = [base64.b64encode(b).decode("utf-8") for b in photo_bytes_list]
            run_analysis_task.delay(
                job_id=job_id,
                photo_b64_list=photo_b64_list,
                home_value=item.home_value,
                zip_code=item.zip_code or "",
            )
        except Exception as e:
            errors.append(BulkJobError(index=idx, reason=f"Enqueue failed: {e}"))
            continue
        job_ids.append(job_id)
    return BulkJobResponse(job_ids=job_ids, errors=errors)
