# Runbook – Repair ROI Optimizer

## Local development

### Streamlit only
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Full stack (Docker)
```bash
docker-compose up -d
# Streamlit http://localhost:8501, API http://localhost:8000
```

### API + worker (no Docker)
```bash
export DATABASE_URL=postgresql://user:pass@localhost:5432/repair_optimizer
export REDIS_URL=redis://localhost:6379/0
# Terminal 1
uvicorn api.main:app --reload --port 8000
# Terminal 2
celery -A worker.tasks worker --loglevel=info
```

## Environment

| Variable | Description |
|----------|-------------|
| DATABASE_URL | PostgreSQL URL (required for jobs/properties) |
| REDIS_URL | Redis URL (required for Celery) |
| FREE_TIER | true = placeholder analysis; false = full ML |
| USE_LLAVA | true = LLaVA condition analysis (needs 2GB+) |
| API_KEY | If set, X-API-Key required on protected routes |
| API_BASE_URL | Base URL for status_url and webhooks |
| PROMETHEUS_MULTIPROC_DIR | Dir for Prometheus multiprocess metrics (set in API + worker for /metrics aggregation) |
| MLFLOW_TRACKING_URI | MLflow server URI; set to "false" to disable pipeline logging |

## Health checks and observability

- `GET /api/v1/health` – liveness  
- `GET /api/v1/ready` – readiness (DB/Redis status)  
- `GET /metrics` – Prometheus metrics (jobs started/completed/failed, pipeline duration)  
- `WS /api/v1/jobs/{job_id}/stream` – WebSocket job progress (status pushed every ~1.5 s until completed/failed)  

## Common issues

- **503 on jobs**: Set DATABASE_URL and REDIS_URL (or run sync fallback when Celery not installed).  
- **OOM on worker**: Use Standard (2GB) or higher; set FREE_TIER=false only when sufficient RAM.  
- **401 on properties/strategies**: Set X-API-Key header when API_KEY is set.  

## Logs

- API: uvicorn stdout.  
- Worker: Celery stdout.  
- DB: PostgreSQL logs.  

## Load testing

- **Locust**: Scenario under `tests/load/locustfile.py` – POSTs a job (multipart with small fixture image), then polls GET `/api/v1/jobs/{job_id}` until completed or 5 min timeout.
- **Run**: `locust -f tests/load/locustfile.py --host=http://localhost:8000` (optional: `--users 2 --spawn-rate 0.5 --run-time 1h`).
- **Target**: 12–15 jobs/hour sustained with 1–2 workers to validate ~750 assessments/week.

## Database indexes (high-volume job queries)

- **Job**: `(status, created_at)` for recent pending/completed lists; `(property_id, created_at)` for property history (see `db/models.py`).
- **Execution**: `(status)` and `(property_id)` for filtering. New indexes are created on `create_all`; for existing DBs, add via migration or `CREATE INDEX` if needed.

## Deployment automation

- **Render**: Use a **Blueprint** (or separate services) for API + worker + Redis + PostgreSQL. Create a **Web Service** for the API (`uvicorn api.main:app --host 0.0.0.0 --port $PORT`), a **Background Worker** for Celery (`celery -A worker.tasks worker --loglevel=info`), and attach **Redis** and **PostgreSQL** from the Render dashboard. Set `DATABASE_URL`, `REDIS_URL`, `API_BASE_URL`, and optionally `PROMETHEUS_MULTIPROC_DIR` (e.g. a writable path or leave unset for single-process metrics). See `render.yaml` for the Streamlit app; for API+worker, duplicate and set `startCommand` to uvicorn/celery as above.
- **Docker**: `docker-compose up` runs API, worker, Streamlit, Postgres, and Redis. For production, use a single `docker-compose.prod.yml` (or override) with no Streamlit if only API+worker are needed.
- **Terraform/Ansible**: Not included in-repo; choose one for your cloud (e.g. Terraform for AWS/GCP, Ansible for VM orchestration). Document in `deploy/` or in this runbook: provision Postgres + Redis, deploy API (uvicorn) and worker (Celery) with same env vars, and optionally a reverse proxy (nginx) and Prometheus scraper for `GET /metrics`.

## Backup

- PostgreSQL: `pg_dump`; back up Redis if storing cache critical to recovery.  
- Audit logs: retain per compliance policy.  
