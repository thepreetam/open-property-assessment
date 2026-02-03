"""
Celery task: run ML pipeline and update job in DB.
Serializes photo bytes as base64 for JSON-safe broker payload.
"""
import base64
from typing import List

from celery import Celery
from core.config import settings
from core.pipeline import run_pipeline

# Celery app using Redis from config
celery_app = Celery(
    "repair_optimizer",
    broker=settings.redis_url,
    backend=settings.redis_url,
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
)


@celery_app.task(bind=True, name="worker.tasks.run_analysis_task")
def run_analysis_task(
    self,
    job_id: str,
    photo_b64_list: List[str],
    home_value: int,
    zip_code: str = "",
):
    """
    Decode base64 photos, run pipeline, update job in PostgreSQL.
    """
    from sqlalchemy.orm import Session
    from db.models import Job, get_session_factory

    photo_bytes_list = [base64.b64decode(s) for s in photo_b64_list]

    # Update status to processing
    factory = get_session_factory(settings.database_url)
    session: Session = factory()
    try:
        job = session.query(Job).filter(Job.id == job_id).first()
        if not job:
            return {"error": "Job not found"}
        job.status = "processing"
        session.commit()
    finally:
        session.close()

    try:
        result = run_pipeline(
            photo_bytes_list,
            home_value,
            zip_code or None,
            free_tier=settings.free_tier,
            use_llava=settings.use_llava,
        )
    except Exception as e:
        factory = get_session_factory(settings.database_url)
        session = factory()
        try:
            job = session.query(Job).filter(Job.id == job_id).first()
            if job:
                job.status = "failed"
                job.error = str(e)
                session.commit()
        finally:
            session.close()
        raise

    session = factory()
    try:
        job = session.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = "completed"
            job.result = {
                "all_data": result["all_data"],
                "strategies": result["strategies"],
                "recommendation": result["recommendation"],
            }
            session.commit()
    finally:
        session.close()

    return {"job_id": job_id, "status": "completed"}
