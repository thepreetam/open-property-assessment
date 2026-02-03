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

## Health checks

- `GET /api/v1/health` – liveness  
- `GET /api/v1/ready` – readiness (DB/Redis status)  

## Common issues

- **503 on jobs**: Set DATABASE_URL and REDIS_URL (or run sync fallback when Celery not installed).  
- **OOM on worker**: Use Standard (2GB) or higher; set FREE_TIER=false only when sufficient RAM.  
- **401 on properties/strategies**: Set X-API-Key header when API_KEY is set.  

## Logs

- API: uvicorn stdout.  
- Worker: Celery stdout.  
- DB: PostgreSQL logs.  

## Backup

- PostgreSQL: `pg_dump`; back up Redis if storing cache critical to recovery.  
- Audit logs: retain per compliance policy.  
