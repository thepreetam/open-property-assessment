# Repair ROI Optimizer – Architecture

## Overview

- **Presentation**: Streamlit (MVP), API Gateway (FastAPI).
- **Business logic**: Strategy engine, repair option generator, ROI calculator, market intelligence stub.
- **Data & ML**: PostgreSQL (jobs, properties, workspaces, audit), Redis (Celery broker + optional cache), in-process ML pipeline (room classifier, BLIP/LLaVA, YOLO).

## Components

| Layer | Components |
|-------|------------|
| API | FastAPI v1: health, jobs, properties, strategies, execute, workspaces, analytics, contractor, export. Optional API key (API_KEY env). |
| Worker | Celery + Redis: run_analysis_task (pipeline, update job). |
| Core | pipeline.py (ML), repair_engine.py (findings → strategies), repair_option_generator.py (options by defect, timeline), market_intelligence.py (zip-based cost/uplift stub), cache.py (Redis), webhooks.py (notify on completion). |
| DB | SQLAlchemy: Team, Workspace, WorkspaceMember, AuditLog, Property, Job, Execution. |

## Data Flow

1. **Job creation**: POST /api/v1/jobs (photos, home_value, zip_code, optional property_id) → job_id; Celery enqueues pipeline.
2. **Pipeline**: Photos → room classification, condition description, YOLO → per-photo data → findings → three strategies (budget/value/premium) + recommendation; result stored in Job.result.
3. **Strategies**: GET /api/v1/strategies?job_id= or property_id= → strategies + optional timeline.
4. **Execute**: POST /api/v1/execute (property_id, strategy_key) → execution record (stub dispatched).
5. **Multi-tenant**: Workspaces and teams; RBAC via X-Workspace-Id and X-User-Id; audit_logs for actions.

## Deployment

- **Streamlit-only**: `streamlit run app.py` (no DB/Redis).
- **Full stack**: Docker Compose (db, redis, api, worker, streamlit). Set DATABASE_URL, REDIS_URL; worker needs 2GB+ for ML.
- **Production**: Use TLS, API_KEY, encryption at rest for DB/Redis; see docs/SECURITY.md.
