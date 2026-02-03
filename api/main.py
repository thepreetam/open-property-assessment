"""
FastAPI application for Repair ROI Optimizer.
Phase 1: health, jobs. Phase 2: properties, strategies, execute. Phase 3: multi-tenant, analytics.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from api.auth import require_api_key
from api.routes import health, jobs, properties, strategies, execute, workspaces, analytics, contractor, export
from core.config import settings
from core.metrics import metrics_output
from db.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Repair ROI Optimizer API",
    version="2.0.0",
    description="Property repair strategies: jobs, properties, strategies, execute. API key optional (API_KEY env).",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health (no auth)
app.include_router(health.router, prefix="/api/v1")
# v1 routes (optional API key via dependency on protected routes if desired; for simplicity we don't enforce on all)
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(properties.router, prefix="/api/v1")
app.include_router(strategies.router, prefix="/api/v1")
app.include_router(execute.router, prefix="/api/v1")
app.include_router(workspaces.router, prefix="/api/v1")
app.include_router(analytics.router, prefix="/api/v1")
app.include_router(contractor.router, prefix="/api/v1")
app.include_router(export.router, prefix="/api/v1")


@app.get("/")
def root():
    return {"service": "Repair ROI Optimizer API", "docs": "/docs"}


@app.get("/metrics")
def metrics():
    """Prometheus-compatible metrics (jobs started/completed/failed, pipeline duration)."""
    body, content_type = metrics_output()
    return Response(content=body, media_type=content_type)
